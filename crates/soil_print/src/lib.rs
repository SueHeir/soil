//! Output systems for SOIL simulations: thermo printing, CSV/binary dump files,
//! restart file serialization, and VTP (ParaView) visualization output.
//!
//! # Overview
//!
//! This crate provides four output subsystems, each controlled by its own TOML
//! configuration section:
//!
//! | System  | TOML section  | Description                                    |
//! |---------|---------------|------------------------------------------------|
//! | Thermo  | `[thermo]`    | Periodic console output of simulation metrics  |
//! | Dump    | `[dump]`      | Per-atom CSV or binary snapshots                |
//! | Restart | `[restart]`   | Checkpoint files for resuming simulations       |
//! | VTP     | `[vtp]`       | ParaView-compatible `.vtp` visualization files  |
//!
//! All systems are registered automatically when [`PrintPlugin`] is added to the app.
//!
//! # TOML Configuration
//!
//! ```toml
//! [thermo]
//! # Columns to display (optional — defaults to step, atoms, ke, neighbors, walltime, stepps).
//! # Use "compute/group" syntax for group-filtered values, e.g. "ke/mobile".
//! # Built-in columns: step, atoms, ke, temp, neighbors, walltime, stepps.
//! # Any name pushed via Thermo::set() is also available.
//! columns = ["step", "atoms", "ke", "temp", "walltime", "stepps"]
//!
//! [dump]
//! # Write dump every N steps (0 = disabled, default: 0)
//! interval = 1000
//! # Output format: "text" (CSV) or "binary" (little-endian f64/u32)
//! format = "text"
//!
//! [restart]
//! # Write restart every N steps (0 = disabled, default: 0)
//! interval = 5000
//! # File format: "bincode" (compact binary, default) or "json" (human-readable)
//! format = "bincode"
//! # Read the latest restart file at startup (default: false)
//! read = false
//!
//! [vtp]
//! # Write VTP (ParaView) output every N steps (0 = disabled, default: 0)
//! interval = 500
//! ```
//!
//! # Extending Dump / VTP Output
//!
//! Plugins register additional per-atom columns via [`DumpRegistry`]. The
//! registry uses interior mutability, so a plugin takes a shared ref from its
//! `build()`:
//!
//! ```rust,ignore
//! let dump_reg = app.get_resource_ref::<DumpRegistry>().unwrap();
//! dump_reg.register_scalar("pressure", |atoms, registry| {
//!     // Return Vec<f64> of length atoms.nlocal
//!     vec![0.0; atoms.nlocal as usize]
//! });
//! ```
//!
//! See `examples/output_wiring.rs` for a runnable end-to-end demonstration that
//! registers a scalar column, a vector column, a custom dump format, and a
//! thermo value.

use std::{
    cell::RefCell,
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Write},
    time::Instant,
};

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

use soil_core::{compute_ke, Atom, AtomDataRegistry, CommResource, Config, Domain, GroupRegistry, Input, RunConfig, RunState, ParticleSimScheduleSet, ScheduleSetupSet, VirialStress};
use soil_core::Neighbor;

// ── Thermo config ───────────────────────────────────────────────────────────

/// TOML `[thermo]` — configures which columns appear in thermo console output.
///
/// # TOML Fields
///
/// | Field     | Type             | Default                                             | Description                        |
/// |-----------|------------------|------------------------------------------------------|------------------------------------|
/// | `columns` | `[String]` (opt) | `["step","atoms","ke","neighbors","walltime","stepps"]` | Column names to display         |
///
/// Column names can use `"compute/group"` syntax (e.g. `"ke/mobile"`) to filter
/// by a named atom group. Built-in compute names: `step`, `atoms`, `ke`, `temp`,
/// `neighbors`, `walltime`, `stepps`. Any value pushed via [`Thermo::set`] is also
/// available as a column.
#[derive(Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct ThermoConfig {
    /// Column names to display. If `None`, uses the default set.
    #[serde(default)]
    pub columns: Option<Vec<String>>,
}

// ── Thermo column ───────────────────────────────────────────────────────────

/// A parsed thermo column specification, produced from a raw string like `"ke/mobile"`.
pub struct ThermoColumn {
    /// The original column string from config (e.g. `"ke/mobile"`).
    pub raw: String,
    /// The compute name portion (e.g. `"ke"`).
    pub compute_name: String,
    /// Optional group name filter (e.g. `Some("mobile")`).
    pub group_name: Option<String>,
    /// Formatted header string for console display (e.g. `"Ke/mobile"`).
    pub header: String,
    /// Column display width in characters (minimum 12).
    pub width: usize,
}

/// Parse a raw column spec string (e.g. `"ke/mobile"`) into a [`ThermoColumn`].
fn parse_thermo_column(raw: &str) -> ThermoColumn {
    let parts: Vec<&str> = raw.splitn(2, '/').collect();
    let compute_name = parts[0].to_string();
    let group_name = parts.get(1).map(|s| s.to_string());

    let header = if let Some(ref g) = group_name {
        format!("{}/{}", capitalize(&compute_name), g)
    } else {
        capitalize(&compute_name)
    };

    let width = header.len().max(12);

    ThermoColumn {
        raw: raw.to_string(),
        compute_name,
        group_name,
        header,
        width,
    }
}

/// Capitalize the first character of a string.
fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// Returns the default thermo column names when no `[thermo] columns` is specified.
fn default_columns() -> Vec<String> {
    vec![
        "step".into(),
        "atoms".into(),
        "ke".into(),
        "neighbors".into(),
        "walltime".into(),
        "stepps".into(),
    ]
}

// ── Thermo ──────────────────────────────────────────────────────────────────

/// Runtime state for thermo console output.
///
/// Tracks the print interval, wall-clock timing for steps-per-second calculation,
/// parsed column specifications, and user-pushed values from other plugins.
///
/// Other plugins can push custom values via [`Thermo::set`], which become available
/// as thermo columns if listed in the `[thermo] columns` config.
pub struct Thermo {
    /// Print thermo output every N steps.
    pub interval: usize,
    /// Wall-clock timestamp of the last thermo print (for steps/sec calculation).
    pub start_time: Instant,
    /// The step number at which thermo was last printed.
    pub last_printed_step: usize,
    /// Parsed column specifications for output formatting.
    pub columns: Vec<ThermoColumn>,
    /// User-pushed named values (e.g. `"pe"` → `42.0`), available as thermo columns.
    pub values: HashMap<String, f64>,
}

impl Default for Thermo {
    fn default() -> Self {
        Self::new()
    }
}

impl Thermo {
    /// Create a new `Thermo` with default interval of 100 steps.
    pub fn new() -> Self {
        Thermo {
            interval: 100,
            start_time: Instant::now(),
            last_printed_step: 0,
            columns: Vec::new(),
            values: HashMap::new(),
        }
    }

    /// Push a named value into the thermo value map.
    ///
    /// The value becomes available as a thermo column if its name is listed in
    /// `[thermo] columns`. Values are overwritten on each call, so plugins should
    /// call this every thermo interval to keep values current.
    pub fn set(&mut self, name: &str, value: f64) {
        self.values.insert(name.to_string(), value);
    }
}

// ── Dump config ─────────────────────────────────────────────────────────────

/// Default dump format: CSV text.
fn default_dump_format() -> String {
    "text".to_string()
}

/// TOML `[dump]` — per-atom dump file output settings.
///
/// # TOML Fields
///
/// | Field      | Type    | Default  | Description                                      |
/// |------------|---------|----------|--------------------------------------------------|
/// | `interval` | `usize` | `0`      | Write dump every N steps (0 = disabled)          |
/// | `format`   | `String`| `"text"` | `"text"` / `"binary"` / `"lammps"` / plugin name |
/// | `per_rank` | `bool`  | `false`  | One file per MPI rank vs. one gathered file      |
/// | `ghost`    | `bool`  | `false`  | Include ghost (halo) atoms                       |
///
/// # Output Files
///
/// With `per_rank = false` (default) ranks gather to rank 0, which writes a
/// single `dump/dump_{step}.{ext}`. With `per_rank = true` each rank writes its
/// own `dump/dump_{step}_rank{rank}.{ext}` (no gather).
#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct DumpConfig {
    /// Write dump every N steps (0 = disabled).
    #[serde(default)]
    pub interval: usize,
    /// Output format: `"text"` (CSV), `"binary"` (little-endian), `"lammps"`
    /// (`.lammpstrj`, OVITO-native), or any format a plugin registered via
    /// [`DumpRegistry::register_format`].
    #[serde(default = "default_dump_format")]
    pub format: String,
    /// `false` (default): MPI-gather all ranks' atoms to rank 0 and write one
    /// file per step. `true`: each rank writes its own `_rank{rank}` file.
    #[serde(default)]
    pub per_rank: bool,
    /// Include ghost (halo) atoms, flagged with an `IsGhost` column.
    #[serde(default)]
    pub ghost: bool,
}

impl Default for DumpConfig {
    fn default() -> Self {
        DumpConfig {
            interval: 0,
            format: "text".to_string(),
            per_rank: false,
            ghost: false,
        }
    }
}

// ── Restart config ──────────────────────────────────────────────────────────

/// Default restart format: bincode.
fn default_restart_format() -> String {
    "bincode".to_string()
}

/// TOML `[restart]` — restart (checkpoint) file write/read settings.
///
/// # TOML Fields
///
/// | Field      | Type    | Default      | Description                                  |
/// |------------|---------|--------------|----------------------------------------------|
/// | `interval` | `usize` | `0`         | Write restart every N steps (0 = disabled)   |
/// | `format`   | `String`| `"bincode"` | `"bincode"` (compact) or `"json"` (readable) |
/// | `read`     | `bool`  | `false`     | Read latest restart file at startup          |
///
/// When `read = true`, the system scans the restart directory for the highest-numbered
/// restart file matching this rank and format, then restores atom state from it.
#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct RestartConfig {
    /// Write restart every N steps (0 = disabled).
    #[serde(default)]
    pub interval: usize,
    /// File format: `"bincode"` (compact binary) or `"json"` (human-readable).
    #[serde(default = "default_restart_format")]
    pub format: String,
    /// Whether to read the latest restart file at startup.
    #[serde(default)]
    pub read: bool,
}

impl Default for RestartConfig {
    fn default() -> Self {
        RestartConfig {
            interval: 0,
            format: "bincode".to_string(),
            read: false,
        }
    }
}

// ── RestartData ─────────────────────────────────────────────────────────────

/// Serializable snapshot of all atom state for restart files.
///
/// Positions, velocities, and forces are stored as separate x/y/z vectors for
/// serialization compatibility. Legacy rotational fields (`omega_*`, `torque_*`,
/// etc.) are kept for backwards compatibility with old restart files but are no
/// longer written — rotational data now lives in `atom_data_buffers` via `AtomData`.
#[derive(Serialize, Deserialize)]
struct RestartData {
    natoms: u64,
    total_cycle: usize,
    dt: f64,
    tag: Vec<u32>,
    atom_type: Vec<u32>,
    pos_x: Vec<f64>,
    pos_y: Vec<f64>,
    pos_z: Vec<f64>,
    vel_x: Vec<f64>,
    vel_y: Vec<f64>,
    vel_z: Vec<f64>,
    force_x: Vec<f64>,
    force_y: Vec<f64>,
    force_z: Vec<f64>,
    mass: Vec<f64>,
    cutoff_radius: Vec<f64>,
    atom_data_buffers: Vec<Vec<f64>>,
    // Legacy fields for backwards compatibility with old restart files
    #[serde(default)]
    omega_x: Vec<f64>,
    #[serde(default)]
    omega_y: Vec<f64>,
    #[serde(default)]
    omega_z: Vec<f64>,
    #[serde(default)]
    torque_x: Vec<f64>,
    #[serde(default)]
    torque_y: Vec<f64>,
    #[serde(default)]
    torque_z: Vec<f64>,
    #[serde(default)]
    ang_mom_x: Vec<f64>,
    #[serde(default)]
    ang_mom_y: Vec<f64>,
    #[serde(default)]
    ang_mom_z: Vec<f64>,
    #[serde(default)]
    quaternion: Vec<[f64; 4]>,
}

impl RestartData {
    /// Build a `RestartData` snapshot from the current atom state.
    ///
    /// Only local atoms (indices `0..nlocal`) are included; ghost atoms are excluded.
    fn from_atoms(atoms: &Atom, registry: &AtomDataRegistry, step: usize) -> Self {
        let nlocal = atoms.nlocal as usize;
        RestartData {
            natoms: atoms.natoms,
            total_cycle: step,
            dt: atoms.dt,
            tag: atoms.tag[..nlocal].to_vec(),
            atom_type: atoms.atom_type[..nlocal].to_vec(),
            pos_x: atoms.pos[..nlocal].iter().map(|p| p[0]).collect(),
            pos_y: atoms.pos[..nlocal].iter().map(|p| p[1]).collect(),
            pos_z: atoms.pos[..nlocal].iter().map(|p| p[2]).collect(),
            vel_x: atoms.vel[..nlocal].iter().map(|v| v[0]).collect(),
            vel_y: atoms.vel[..nlocal].iter().map(|v| v[1]).collect(),
            vel_z: atoms.vel[..nlocal].iter().map(|v| v[2]).collect(),
            force_x: atoms.force[..nlocal].iter().map(|v| v[0]).collect(),
            force_y: atoms.force[..nlocal].iter().map(|v| v[1]).collect(),
            force_z: atoms.force[..nlocal].iter().map(|v| v[2]).collect(),
            mass: atoms.mass[..nlocal].to_vec(),
            cutoff_radius: atoms.cutoff_radius[..nlocal].to_vec(),
            atom_data_buffers: registry.pack_all_for_restart(nlocal),
            // Legacy fields left empty — rotational data now in atom_data_buffers via DemAtom
            omega_x: Vec::new(),
            omega_y: Vec::new(),
            omega_z: Vec::new(),
            torque_x: Vec::new(),
            torque_y: Vec::new(),
            torque_z: Vec::new(),
            ang_mom_x: Vec::new(),
            ang_mom_y: Vec::new(),
            ang_mom_z: Vec::new(),
            quaternion: Vec::new(),
        }
    }
}

// ── DumpRegistry ────────────────────────────────────────────────────────

/// Registry of user-defined per-atom data callbacks for dump and VTP output.
///
/// Plugins register callbacks during their `build()` phase. These callbacks are
/// only invoked on steps when dump/VTP output is actually written — zero overhead
/// on non-output steps.
///
/// # Example
///
/// ```rust,ignore
/// // In a plugin's build() — `register_scalar` takes `&self`, so a shared ref:
/// let dump_reg = app.get_resource_ref::<DumpRegistry>().unwrap();
/// dump_reg.register_scalar("pressure", |atoms, registry| {
///     let dem = registry.expect::<DemAtom>("pressure");
///     (0..atoms.nlocal as usize).map(|i| /* ... */ 0.0).collect()
/// });
/// ```
///
/// See `soil_print`'s `examples/output_wiring.rs` for a compiling end-to-end use.
pub struct DumpRegistry {
    scalar_fns: RefCell<Vec<(String, Box<dyn Fn(&Atom, &AtomDataRegistry) -> Vec<f64>>)>>,
    vector_fns: RefCell<Vec<(String, Box<dyn Fn(&Atom, &AtomDataRegistry) -> Vec<[f64; 3]>>)>>,
    format_fns: RefCell<Vec<(String, Box<dyn Fn(&DumpFrame) -> std::io::Result<()>>)>>,
    /// Global box bounds + periodicity, synced from [`Domain`] each step so format
    /// writers can emit a box header without threading `Domain` through every caller.
    box_info: RefCell<BoxInfo>,
}

/// Global simulation-box snapshot for dump headers.
#[derive(Clone, Copy)]
pub struct BoxInfo {
    pub low: [f64; 3],
    pub high: [f64; 3],
    pub periodic: [bool; 3],
}

/// One frame of per-atom data ready to serialize, decoupled from [`Atom`] so the
/// same writer handles per-rank, MPI-gathered, and ghost-inclusive output.
///
/// Columns are parallel vectors all of the same length (`self.n()`). Registered
/// scalar/vector columns are pre-evaluated and padded to that length (ghost atoms
/// get `0`). Built per rank, optionally gathered to rank 0, then handed to a
/// format writer registered via [`DumpRegistry::register_format`].
#[derive(Default)]
pub struct DumpFrame {
    /// Current timestep (total cycle count).
    pub step: usize,
    /// Global simulation box (bounds + per-axis periodicity).
    pub box_info: BoxInfo,
    /// Output path WITHOUT extension (e.g. `dump/dump_500` or `dump/dump_500_rank2`);
    /// the writer appends its own format-specific suffix.
    pub path_stem: String,
    pub tag: Vec<u64>,
    pub atom_type: Vec<u64>,
    pub pos: Vec<[f64; 3]>,
    pub vel: Vec<[f64; 3]>,
    pub force: Vec<[f64; 3]>,
    pub radius: Vec<f64>,
    /// Registered scalar columns `(name, values)`, each of length `n()`.
    pub scalars: Vec<(String, Vec<f64>)>,
    /// Registered vector columns `(name, values)`, each of length `n()`.
    pub vectors: Vec<(String, Vec<[f64; 3]>)>,
    /// `Some` when ghost atoms are included: `true` marks a ghost (halo) atom.
    pub is_ghost: Option<Vec<bool>>,
}

impl Default for BoxInfo {
    fn default() -> Self {
        BoxInfo { low: [0.0; 3], high: [1.0; 3], periodic: [true; 3] }
    }
}

impl DumpFrame {
    /// Number of atoms (rows) in this frame.
    pub fn n(&self) -> usize {
        self.tag.len()
    }

    /// Flatten one atom's fields into the transport buffer (must mirror [`push_unpacked`]).
    fn stride(&self) -> usize {
        // tag,type,radius + pos,vel,force(9) + scalars + 3*vectors + ghost flag
        3 + 9 + self.scalars.len() + 3 * self.vectors.len() + self.is_ghost.is_some() as usize
    }

    /// Pack all atoms into a flat `f64` buffer for MPI transport.
    fn pack(&self) -> Vec<f64> {
        let mut buf = Vec::with_capacity(self.n() * self.stride());
        for i in 0..self.n() {
            buf.push(self.tag[i] as f64);
            buf.push(self.atom_type[i] as f64);
            buf.push(self.radius[i]);
            buf.extend_from_slice(&self.pos[i]);
            buf.extend_from_slice(&self.vel[i]);
            buf.extend_from_slice(&self.force[i]);
            for (_, v) in &self.scalars {
                buf.push(v[i]);
            }
            for (_, v) in &self.vectors {
                buf.extend_from_slice(&v[i]);
            }
            if let Some(g) = &self.is_ghost {
                buf.push(g[i] as u8 as f64);
            }
        }
        buf
    }

    /// Append atoms from a packed buffer produced by [`pack`](Self::pack) on a
    /// peer rank. Column names/structure are taken from `self`, so `self` must
    /// already carry the (possibly empty) column layout.
    fn push_unpacked(&mut self, buf: &[f64]) {
        let stride = self.stride();
        if stride == 0 {
            return;
        }
        for chunk in buf.chunks_exact(stride) {
            self.tag.push(chunk[0] as u64);
            self.atom_type.push(chunk[1] as u64);
            self.radius.push(chunk[2]);
            self.pos.push([chunk[3], chunk[4], chunk[5]]);
            self.vel.push([chunk[6], chunk[7], chunk[8]]);
            self.force.push([chunk[9], chunk[10], chunk[11]]);
            let mut k = 12;
            for (_, v) in &mut self.scalars {
                v.push(chunk[k]);
                k += 1;
            }
            for (_, v) in &mut self.vectors {
                v.push([chunk[k], chunk[k + 1], chunk[k + 2]]);
                k += 3;
            }
            if let Some(g) = &mut self.is_ghost {
                g.push(chunk[k] != 0.0);
            }
        }
    }
}

impl Default for DumpRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DumpRegistry {
    /// Create an empty `DumpRegistry` with no registered callbacks.
    pub fn new() -> Self {
        DumpRegistry {
            scalar_fns: RefCell::new(Vec::new()),
            vector_fns: RefCell::new(Vec::new()),
            format_fns: RefCell::new(Vec::new()),
            box_info: RefCell::new(BoxInfo {
                low: [0.0; 3],
                high: [1.0; 3],
                periodic: [true; 3],
            }),
        }
    }

    /// Register a per-atom scalar column for dump/VTP output.
    ///
    /// The callback receives the current [`Atom`] and [`AtomDataRegistry`] and should
    /// return a `Vec<f64>` of length `atoms.nlocal`. The column appears in CSV/LAMMPS
    /// dumps and as a VTP `Float32` data array.
    ///
    /// Takes `&self` (interior mutability) so plugins can register from `build()`
    /// via `app.get_resource_ref::<DumpRegistry>()`.
    pub fn register_scalar(
        &self,
        name: impl Into<String>,
        f: impl Fn(&Atom, &AtomDataRegistry) -> Vec<f64> + 'static,
    ) {
        self.scalar_fns.borrow_mut().push((name.into(), Box::new(f)));
    }

    /// Register a per-atom 3-component vector column for dump/VTP output.
    ///
    /// The callback should return a `Vec<[f64; 3]>` of length `atoms.nlocal`.
    /// In CSV/LAMMPS dumps, the vector is split into `{name}_x`, `{name}_y`,
    /// `{name}_z` columns. In VTP output, it appears as a 3-component `Float32` array.
    pub fn register_vector(
        &self,
        name: impl Into<String>,
        f: impl Fn(&Atom, &AtomDataRegistry) -> Vec<[f64; 3]> + 'static,
    ) {
        self.vector_fns.borrow_mut().push((name.into(), Box::new(f)));
    }

    /// Register a named dump output format.
    ///
    /// The writer receives a [`DumpFrame`] (already MPI-gathered when
    /// `per_rank = false`) and creates/writes one file, appending its own
    /// extension to `frame.path_stem`. Selected at runtime by `[dump] format =
    /// "<name>"`. A later registration of the same name wins, so plugins may
    /// override a built-in format.
    ///
    /// Core registers `"text"`, `"binary"`, and `"lammps"`; any plugin can add
    /// its own from its `build()`:
    ///
    /// ```rust,ignore
    /// let dump_reg = app.get_resource_ref::<DumpRegistry>().unwrap();
    /// dump_reg.register_format("xyz", |frame| {
    ///     let f = std::fs::File::create(format!("{}.xyz", frame.path_stem))?;
    ///     // ... write frame.pos / frame.scalars / frame.vectors ...
    ///     Ok(())
    /// });
    /// ```
    pub fn register_format(
        &self,
        name: impl Into<String>,
        f: impl Fn(&DumpFrame) -> std::io::Result<()> + 'static,
    ) {
        self.format_fns.borrow_mut().push((name.into(), Box::new(f)));
    }

    /// Update the cached global box (called each step from [`Domain`]).
    pub fn set_box(&self, info: BoxInfo) {
        *self.box_info.borrow_mut() = info;
    }

    /// Current cached global box.
    pub(crate) fn box_info(&self) -> BoxInfo {
        *self.box_info.borrow()
    }

    /// Run the writer for `name` against `frame` (last registration wins).
    /// Returns `None` if no format with that name is registered.
    pub(crate) fn write_format(
        &self,
        name: &str,
        frame: &DumpFrame,
    ) -> Option<std::io::Result<()>> {
        let formats = self.format_fns.borrow();
        formats
            .iter()
            .rev()
            .find(|(n, _)| n == name)
            .map(|(_, f)| f(frame))
    }

    /// Names of all registered dump formats, in registration order.
    pub fn format_names(&self) -> Vec<String> {
        self.format_fns.borrow().iter().map(|(n, _)| n.clone()).collect()
    }

    /// Returns `true` if any scalar or vector callbacks are registered.
    pub fn has_callbacks(&self) -> bool {
        !self.scalar_fns.borrow().is_empty() || !self.vector_fns.borrow().is_empty()
    }
}

// ── VTP config ──────────────────────────────────────────────────────────────

/// TOML `[vtp]` — ParaView `.vtp` visualization output settings.
///
/// # TOML Fields
///
/// | Field      | Type    | Default | Description                              |
/// |------------|---------|---------|------------------------------------------|
/// | `interval` | `usize` | `0`    | Write VTP every N steps (0 = disabled)   |
///
/// VTP files are written to `{output_dir}/vtp/{step}CYCLE_{rank}RANK.vtp` and
/// include particle positions, radii, velocity magnitudes, ghost flags, and any
/// fields registered via [`DumpRegistry`].
#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct VtpConfig {
    /// Write VTP every N steps (0 = disabled).
    #[serde(default)]
    pub interval: usize,
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Main output plugin — registers thermo, dump, restart, and VTP systems.
///
/// Add this plugin to the app to enable all output subsystems. Each subsystem
/// is independently configured via its TOML section and only produces output
/// when its interval is non-zero.
pub struct PrintPlugin;

impl Plugin for PrintPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[dump]
# Dump output interval (0 = disabled)
interval = 0
# Dump format: "text" (CSV), "binary", or "lammps" (.lammpstrj, OVITO-native)
format = "text"
# false: MPI-gather all ranks into one file/step; true: one file per rank
per_rank = false
# include ghost (halo) atoms, flagged with an IsGhost column
ghost = false

[restart]
# Restart file write interval (0 = disabled)
interval = 0
# Restart format: "bincode" or "json"
format = "bincode"
# Whether to read restart files at startup
read = false

[vtp]
# VTP (ParaView) output interval (0 = disabled)
interval = 0"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<DumpConfig>(app, "dump");
        Config::load::<RestartConfig>(app, "restart");
        Config::load::<VtpConfig>(app, "vtp");
        Config::load::<ThermoConfig>(app, "thermo");

        // Built-in dump formats, registered through the same public extension
        // point (`register_format`) any plugin uses — core just ships three.
        let dump_reg = DumpRegistry::new();
        dump_reg.register_format("text", write_dump_text);
        dump_reg.register_format("binary", write_dump_binary);
        dump_reg.register_format("lammps", write_dump_lammps);
        app.add_resource(dump_reg);

        app.add_resource(Thermo::new())
            .add_setup_system(setup_thermo, ScheduleSetupSet::PostSetup)
            .add_setup_system(read_restart.run_if(first_stage_only()), ScheduleSetupSet::PostSetup)
            .add_setup_system(sync_dump_box, ScheduleSetupSet::PostSetup)
            .add_update_system(output_virial_to_thermo, ParticleSimScheduleSet::PostForce)
            .add_update_system(sync_dump_box, ParticleSimScheduleSet::PostForce)
            .add_update_system(print_vtp, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(print_thermo, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(dump_atoms, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(write_restart, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(check_stage_end_save.before("update_cycle"), ParticleSimScheduleSet::PostFinalIntegration);
    }
}

// ── Helper: restart directory ───────────────────────────────────────────────

/// Compute the restart base directory from the output directory setting.
fn restart_base_dir(input: &Input) -> String {
    match input.output_dir.as_deref() {
        Some(dir) => format!("{}/restart", dir),
        None => "restart".to_string(),
    }
}

// ── Thermo systems ──────────────────────────────────────────────────────────

/// Setup system for thermo output: parses column specs and prints the header.
///
/// Runs at the start of each stage to update the print interval and reset the
/// wall-clock timer. Column specifications are parsed only once (on the first stage).
pub fn setup_thermo(
    config: Res<RunConfig>,
    thermo_config: Res<ThermoConfig>,
    scheduler_manager: Res<SchedulerManager>,
    comm: Res<CommResource>,
    run_state: Res<RunState>,
    mut thermo: ResMut<Thermo>,
    mut virial: Option<ResMut<VirialStress>>,
) {
    let index = scheduler_manager.index;
    if index >= config.num_stages() {
        return;
    }
    thermo.interval = config.current_stage(index)
        .overrides.get("thermo")
        .and_then(|v| v.as_integer())
        .map(|i| i as usize)
        .unwrap_or(100);
    if let Some(ref mut v) = virial {
        v.set_interval(thermo.interval);
    }
    thermo.start_time = Instant::now();
    thermo.last_printed_step = run_state.total_cycle;

    // Parse column specifications (only on first stage, or if columns empty).
    if thermo.columns.is_empty() {
        let col_names = thermo_config
            .columns
            .clone()
            .unwrap_or_else(default_columns);
        thermo.columns = col_names.iter().map(|s| parse_thermo_column(s)).collect();
    }

    if comm.rank() == 0 {
        println!();
        let header: String = thermo
            .columns
            .iter()
            .map(|c| format!("{:<width$}", c.header, width = c.width))
            .collect::<Vec<_>>()
            .join(" ");
        let total_width: usize = thermo.columns.iter().map(|c| c.width).sum::<usize>()
            + thermo.columns.len().saturating_sub(1);
        println!("{}", header);
        println!("{}", "-".repeat(total_width));
    }
}

/// Print thermo output to console at the configured interval.
///
/// All MPI ranks participate in allreduce operations (for KE, atom counts, etc.),
/// but only rank 0 prints the formatted output line.
#[allow(clippy::too_many_arguments)]
pub fn print_thermo(
    mut atoms: ResMut<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    neighbor: Res<Neighbor>,
    groups: Res<GroupRegistry>,
    mut thermo: ResMut<Thermo>,
) {
    let step = run_state.total_cycle;
    if !step.is_multiple_of(thermo.interval) {
        return;
    }

    // Pre-compute values that need allreduce (all ranks must participate).
    // We compute these eagerly so all ranks call allreduce together.
    let local_ke_all = compute_ke(&atoms, None);
    let global_ke_all = comm.all_reduce_sum_f64(local_ke_all);
    // Refresh the global atom count here (reporting-only; the ghost-rebuild path no
    // longer maintains it). Piggybacks on this thermo collective — no new barrier.
    let global_atoms_all = comm.all_reduce_sum_f64(atoms.nlocal as f64);
    atoms.natoms = global_atoms_all as u64;
    let local_neighbors = neighbor.neighbor_indices.len() as f64;
    let global_neighbors = comm.all_reduce_sum_f64(local_neighbors);

    // Pre-compute group-filtered KE and atom counts for any group columns.
    // Each group that appears needs its own allreduce.
    let mut group_ke: HashMap<String, f64> = HashMap::new();
    let mut group_count: HashMap<String, f64> = HashMap::new();
    for col in thermo.columns.iter() {
        if let Some(ref gname) = col.group_name {
            if group_ke.contains_key(gname) {
                continue;
            }
            if let Some(group) = groups.get(gname) {
                let local_ke = compute_ke(&atoms, Some(&group.mask));
                let local_count = group.count as f64;
                group_ke.insert(gname.clone(), comm.all_reduce_sum_f64(local_ke));
                group_count.insert(gname.clone(), comm.all_reduce_sum_f64(local_count));
            }
        }
    }

    if comm.rank() == 0 {
        let elapsed = thermo.start_time.elapsed().as_secs_f64();
        let steps_since = (step - thermo.last_printed_step) as f64;
        let steps_per_sec = if elapsed > 1e-9 {
            steps_since / elapsed
        } else {
            0.0
        };

        let mut parts: Vec<String> = Vec::new();
        for col in thermo.columns.iter() {
            let val_str = match col.compute_name.as_str() {
                "step" => format!("{:<width$}", step, width = col.width),
                "atoms" => {
                    if let Some(ref gname) = col.group_name {
                        let n = group_count.get(gname).copied().unwrap_or(0.0) as u64;
                        format!("{:<width$}", n, width = col.width)
                    } else {
                        format!("{:<width$}", atoms.natoms, width = col.width)
                    }
                }
                "ke" => {
                    let ke = if let Some(ref gname) = col.group_name {
                        group_ke.get(gname).copied().unwrap_or(0.0)
                    } else {
                        global_ke_all
                    };
                    format!("{:<width$.6e}", ke, width = col.width)
                }
                "temp" => {
                    let (ke, n) = if let Some(ref gname) = col.group_name {
                        (
                            group_ke.get(gname).copied().unwrap_or(0.0),
                            group_count.get(gname).copied().unwrap_or(0.0),
                        )
                    } else {
                        (global_ke_all, global_atoms_all)
                    };
                    let ndof = 3.0 * n - 3.0;
                    let temp = if ndof > 0.0 { 2.0 * ke / ndof } else { 0.0 };
                    format!("{:<width$.6}", temp, width = col.width)
                }
                "neighbors" => {
                    format!("{:<width$}", global_neighbors as usize, width = col.width)
                }
                "walltime" => {
                    format!("{:<width$.4}", elapsed, width = col.width)
                }
                "stepps" => {
                    format!("{:<width$.1}", steps_per_sec, width = col.width)
                }
                other => {
                    // User-pushed value from Thermo::set()
                    if let Some(&v) = thermo.values.get(other) {
                        format!("{:<width$.6e}", v, width = col.width)
                    } else {
                        format!("{:<width$}", "N/A", width = col.width)
                    }
                }
            };
            parts.push(val_str);
        }
        println!("{}", parts.join(" "));
        thermo.start_time = Instant::now();
        thermo.last_printed_step = step;
    }
}

// ── Virial stress output ────────────────────────────────────────────────────

/// MPI-reduce each virial stress component and push to thermo values.
///
/// Publishes `virial_xx`, `virial_yy`, `virial_zz`, `virial_xy`, `virial_xz`,
/// `virial_yz` as thermo columns. Only runs on thermo output steps.
pub fn output_virial_to_thermo(
    virial: Option<Res<VirialStress>>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut thermo: ResMut<Thermo>,
) {
    if !run_state.total_cycle.is_multiple_of(thermo.interval) {
        return;
    }
    let virial = match virial {
        Some(v) => v,
        None => return,
    };
    let xx = comm.all_reduce_sum_f64(virial.xx);
    let yy = comm.all_reduce_sum_f64(virial.yy);
    let zz = comm.all_reduce_sum_f64(virial.zz);
    let xy = comm.all_reduce_sum_f64(virial.xy);
    let xz = comm.all_reduce_sum_f64(virial.xz);
    let yz = comm.all_reduce_sum_f64(virial.yz);
    thermo.set("virial_xx", xx);
    thermo.set("virial_yy", yy);
    thermo.set("virial_zz", zz);
    thermo.set("virial_xy", xy);
    thermo.set("virial_xz", xz);
    thermo.set("virial_yz", yz);
}

// ── VTP output ──────────────────────────────────────────────────────────────

/// Write a single VTP `<DataArray>` element with per-point scalar data.
fn write_vtp_data_array(
    file: &mut File,
    vtp_type: &str,
    name: &str,
    n: usize,
    value_fn: impl Fn(usize) -> String,
) -> std::io::Result<()> {
    writeln!(
        file,
        "<DataArray type=\"{}\" Name=\"{}\" format=\"ascii\">",
        vtp_type, name
    )?;
    for i in 0..n {
        writeln!(file, "{}", value_fn(i))?;
    }
    write!(file, "</DataArray>")?;
    Ok(())
}

/// Write ParaView VTP output at the configured interval.
///
/// Each MPI rank writes its own `.vtp` file containing local + ghost atoms.
/// Output includes positions, radii, velocity magnitude, ghost flags, and any
/// fields registered via [`DumpRegistry`].
#[allow(clippy::too_many_arguments)]
pub fn print_vtp(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    vtp_config: Res<VtpConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    dump_registry: Res<DumpRegistry>,
) {
    let count = run_state.total_cycle;
    let rank = comm.rank();
    let stage = run_config.current_stage(scheduler_manager.index);
    let interval = stage.overrides.get("vtp_interval")
        .and_then(|v| v.as_integer())
        .map(|i| i as usize)
        .unwrap_or(vtp_config.interval);
    if interval == 0 || !count.is_multiple_of(interval) {
        return;
    }
    if let Err(e) = print_vtp_inner(&atoms, &registry, count, rank, &input, &dump_registry) {
        eprintln!("WARNING: VTP write failed at step {}: {}", count, e);
    }
}

/// Inner VTP write logic, separated for error handling via `?`.
fn print_vtp_inner(
    atoms: &Atom,
    registry: &AtomDataRegistry,
    count: usize,
    rank: i32,
    input: &Input,
    dump_reg: &DumpRegistry,
) -> std::io::Result<()> {
    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("{}/vtp", dir),
        None => "vtp".to_string(),
    };
    let filename = format!("{}/{}CYCLE_{}RANK.vtp", base_dir, count, rank);
    fs::create_dir_all(&base_dir)?;
    let mut file = File::create(&filename)?;

    let n = atoms.len();
    let nlocal = atoms.nlocal as usize;

    // XML header and PolyData opening
    write!(&mut file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n<PolyData>\n")?;
    writeln!(&mut file, "<Piece NumberOfPoints=\"{}\">", n)?;

    // Points (positions)
    write!(&mut file, "<Points><DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">")?;
    for i in 0..n {
        writeln!(&mut file, "{} {} {}", atoms.pos[i][0], atoms.pos[i][1], atoms.pos[i][2])?;
    }
    write!(&mut file, "</DataArray>\n</Points>\n")?;

    // Per-point data arrays
    writeln!(&mut file, "<PointData Scalars=\"\" Vectors=\"\">")?;
    write_vtp_data_array(&mut file, "Float32", "Radius", n, |i| format!("{}", atoms.cutoff_radius[i]))?;
    write_vtp_data_array(&mut file, "Float32", "Vel_Mag", n, |i| {
        let vmag = (atoms.vel[i][0].powi(2) + atoms.vel[i][1].powi(2) + atoms.vel[i][2].powi(2)).sqrt();
        format!("{vmag}")
    })?;
    write_vtp_data_array(&mut file, "Int32", "IsGhost", n, |i| {
        if i >= nlocal { "1".to_string() } else { "0".to_string() }
    })?;

    // Registered scalar callbacks
    for (name, f) in dump_reg.scalar_fns.borrow().iter() {
        let data = f(atoms, registry);
        write_vtp_data_array(&mut file, "Float32", name, n, |i| {
            if i < data.len() {
                format!("{}", data[i])
            } else {
                "0".to_string()
            }
        })?;
    }

    // Registered vector callbacks
    for (name, f) in dump_reg.vector_fns.borrow().iter() {
        let data = f(atoms, registry);
        writeln!(
            &mut file,
            "<DataArray type=\"Float32\" Name=\"{}\" NumberOfComponents=\"3\" format=\"ascii\">",
            name
        )?;
        for i in 0..n {
            if i < data.len() {
                writeln!(&mut file, "{} {} {}", data[i][0], data[i][1], data[i][2])?;
            } else {
                writeln!(&mut file, "0 0 0")?;
            }
        }
        write!(&mut file, "</DataArray>")?;
    }

    write!(&mut file, "</PointData>\n</Piece>\n</PolyData>\n</VTKFile>\n")?;
    Ok(())
}

// ── Dump output ─────────────────────────────────────────────────────────────

/// Write per-atom dump files (CSV or binary) at the configured interval.
///
/// Each MPI rank writes its own file containing only local atoms. The dump
/// includes core fields (tag, type, position, velocity, force, radius) plus
/// any columns registered via [`DumpRegistry`].
/// Copy the global box bounds + periodicity from [`Domain`] into the
/// [`DumpRegistry`] so format writers can emit a box header (e.g. LAMMPS
/// `BOX BOUNDS`) without `Domain` being threaded through every dump call site.
pub fn sync_dump_box(domain: Res<Domain>, dump_registry: Res<DumpRegistry>) {
    dump_registry.set_box(BoxInfo {
        low: domain.boundaries_low,
        high: domain.boundaries_high,
        periodic: [
            domain.is_periodic(0),
            domain.is_periodic(1),
            domain.is_periodic(2),
        ],
    });
}

#[allow(clippy::too_many_arguments)]
pub fn dump_atoms(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    dump_config: Res<DumpConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    dump_registry: Res<DumpRegistry>,
) {
    let stage = run_config.current_stage(scheduler_manager.index);
    let interval = stage.overrides.get("dump_interval")
        .and_then(|v| v.as_integer())
        .map(|i| i as usize)
        .unwrap_or(dump_config.interval);
    if interval == 0 {
        return;
    }
    let step = run_state.total_cycle;
    if !step.is_multiple_of(interval) {
        return;
    }

    if let Err(e) = dump_atoms_inner(&atoms, &registry, &comm, step, &input, &dump_config, &dump_registry) {
        eprintln!("WARNING: Dump write failed at step {}: {}", step, e);
    }
}

/// Inner dump write logic, shared between periodic dumps and stage-end saves.
///
/// Builds a [`DumpFrame`] from local atoms (plus ghosts if `ghost = true`),
/// evaluating registered columns once. With `per_rank = false` (default) the
/// frame is MPI-gathered to rank 0, which writes one file; otherwise each rank
/// writes its own. Dispatches to the [`DumpRegistry`] format writer named by
/// `[dump] format`; unknown formats warn rather than fail.
pub(crate) fn dump_atoms_inner(
    atoms: &Atom,
    registry: &AtomDataRegistry,
    comm: &CommResource,
    step: usize,
    input: &Input,
    dump_config: &DumpConfig,
    dump_reg: &DumpRegistry,
) -> std::io::Result<()> {
    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("{}/dump", dir),
        None => "dump".to_string(),
    };

    // Evaluate registered callbacks (only when dump is actually written). Each
    // returns one value per LOCAL atom; ghost atoms (if included) are padded.
    let scalar_data: Vec<(String, Vec<f64>)> = dump_reg
        .scalar_fns
        .borrow()
        .iter()
        .map(|(name, f)| (name.clone(), f(atoms, registry)))
        .collect();
    let vector_data: Vec<(String, Vec<[f64; 3]>)> = dump_reg
        .vector_fns
        .borrow()
        .iter()
        .map(|(name, f)| (name.clone(), f(atoms, registry)))
        .collect();

    // Build this rank's frame. With ghosts, range is all atoms; the IsGhost
    // column flags indices >= nlocal.
    let nlocal = atoms.nlocal as usize;
    let count = if dump_config.ghost { atoms.len() } else { nlocal };
    let mut frame = DumpFrame {
        step,
        box_info: dump_reg.box_info(),
        is_ghost: dump_config.ghost.then(|| Vec::with_capacity(count)),
        ..Default::default()
    };
    frame.tag.reserve(count);
    for i in 0..count {
        frame.tag.push(atoms.tag[i] as u64);
        frame.atom_type.push(atoms.atom_type[i] as u64);
        frame.pos.push(atoms.pos[i]);
        frame.vel.push(atoms.vel[i]);
        frame.force.push(atoms.force[i]);
        frame.radius.push(atoms.cutoff_radius[i]);
        if let Some(g) = &mut frame.is_ghost {
            g.push(i >= nlocal);
        }
    }
    let pad_scalar = |data: &[f64]| (0..count).map(|i| data.get(i).copied().unwrap_or(0.0)).collect();
    let pad_vector = |data: &[[f64; 3]]| (0..count).map(|i| data.get(i).copied().unwrap_or([0.0; 3])).collect();
    frame.scalars = scalar_data.iter().map(|(n, d)| (n.clone(), pad_scalar(d))).collect();
    frame.vectors = vector_data.iter().map(|(n, d)| (n.clone(), pad_vector(d))).collect();

    if dump_config.per_rank {
        // One file per rank — no gather.
        frame.path_stem = format!("{}/dump_{}_rank{}", base_dir, step, comm.rank());
        fs::create_dir_all(&base_dir)?;
        write_frame(dump_reg, dump_config, &frame)?;
    } else {
        // Gather every rank's atoms to rank 0, which writes a single file.
        let is_root = gather_frame_to_root(comm, &mut frame);
        if is_root {
            frame.path_stem = format!("{}/dump_{}", base_dir, step);
            fs::create_dir_all(&base_dir)?;
            write_frame(dump_reg, dump_config, &frame)?;
        }
    }
    Ok(())
}

/// Dispatch a built frame to the configured format writer.
fn write_frame(
    dump_reg: &DumpRegistry,
    dump_config: &DumpConfig,
    frame: &DumpFrame,
) -> std::io::Result<()> {
    match dump_reg.write_format(&dump_config.format, frame) {
        Some(res) => res?,
        None => eprintln!(
            "WARNING: unknown dump format '{}' — no writer registered (available: {})",
            dump_config.format,
            dump_reg.format_names().join(", ")
        ),
    }
    Ok(())
}

/// Gather every rank's atoms into `frame` on rank 0 via point-to-point
/// communication (works with or without the MPI backend). Returns `true` on the
/// root rank (which now holds all atoms and should write); `false` elsewhere.
fn gather_frame_to_root(comm: &CommResource, frame: &mut DumpFrame) -> bool {
    let size = comm.size();
    if size <= 1 {
        return true; // single rank already holds everything
    }
    if comm.rank() == 0 {
        // Receive each peer's packed atoms in rank order and append.
        for src in 1..size {
            let buf = comm.0.recv_f64(src);
            frame.push_unpacked(&buf);
        }
        true
    } else {
        comm.0.send_f64(0, &frame.pack());
        false
    }
}

// ── Built-in dump-format writers ─────────────────────────────────────────────
// Registered by `PrintPlugin::build` via the same public `register_format` API
// that plugins use, so core's formats and plugin formats are peers. All operate
// on a `DumpFrame`, so they are agnostic to per-rank vs. gathered output.

/// Append registered scalar/vector/ghost column NAMES with the given separator
/// and per-vector-component suffixes — shared by the text and LAMMPS headers.
fn append_column_names(header: &mut String, frame: &DumpFrame, sep: char) {
    for (name, _) in &frame.scalars {
        header.push(sep);
        header.push_str(name);
    }
    for (name, _) in &frame.vectors {
        header.push_str(&format!("{sep}{name}_x{sep}{name}_y{sep}{name}_z"));
    }
    if frame.is_ghost.is_some() {
        header.push(sep);
        header.push_str("IsGhost");
    }
}

/// Text/CSV dump: one row per atom, header line of column names.
fn write_dump_text(frame: &DumpFrame) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(format!("{}.csv", frame.path_stem))?);

    let mut header = "tag,type,x,y,z,vx,vy,vz,fx,fy,fz,radius".to_string();
    append_column_names(&mut header, frame, ',');
    writeln!(w, "{}", header)?;

    for i in 0..frame.n() {
        write!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            frame.tag[i], frame.atom_type[i],
            frame.pos[i][0], frame.pos[i][1], frame.pos[i][2],
            frame.vel[i][0], frame.vel[i][1], frame.vel[i][2],
            frame.force[i][0], frame.force[i][1], frame.force[i][2],
            frame.radius[i],
        )?;
        for (_, data) in &frame.scalars {
            write!(w, ",{}", data[i])?;
        }
        for (_, data) in &frame.vectors {
            write!(w, ",{},{},{}", data[i][0], data[i][1], data[i][2])?;
        }
        if let Some(g) = &frame.is_ghost {
            write!(w, ",{}", g[i] as u8)?;
        }
        writeln!(w)?;
    }
    Ok(())
}

/// Binary dump: little-endian u32 count, then packed per-atom records
/// (`tag`/`type` as `u32`, everything else `f64`; IsGhost as a trailing `u8`).
fn write_dump_binary(frame: &DumpFrame) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(format!("{}.bin", frame.path_stem))?);

    w.write_all(&(frame.n() as u32).to_le_bytes())?;
    for i in 0..frame.n() {
        w.write_all(&(frame.tag[i] as u32).to_le_bytes())?;
        w.write_all(&(frame.atom_type[i] as u32).to_le_bytes())?;
        for c in 0..3 {
            w.write_all(&frame.pos[i][c].to_le_bytes())?;
        }
        for c in 0..3 {
            w.write_all(&frame.vel[i][c].to_le_bytes())?;
        }
        for c in 0..3 {
            w.write_all(&frame.force[i][c].to_le_bytes())?;
        }
        w.write_all(&frame.radius[i].to_le_bytes())?;
        for (_, data) in &frame.scalars {
            w.write_all(&data[i].to_le_bytes())?;
        }
        for (_, data) in &frame.vectors {
            for c in 0..3 {
                w.write_all(&data[i][c].to_le_bytes())?;
            }
        }
        if let Some(g) = &frame.is_ghost {
            w.write_all(&[g[i] as u8])?;
        }
    }
    Ok(())
}

/// LAMMPS dump (`.lammpstrj`) — OVITO reads this natively: `radius` renders
/// spheres, `type` colors by species, and registered scalar/vector columns
/// (e.g. a `mode` flag) become per-particle properties you can color by.
fn write_dump_lammps(frame: &DumpFrame) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(format!("{}.lammpstrj", frame.path_stem))?);

    // Global box bounds (same on every rank). Periodic -> "pp", else "ff".
    let bx = &frame.box_info;
    let tok = |d: usize| if bx.periodic[d] { "pp" } else { "ff" };
    writeln!(w, "ITEM: TIMESTEP")?;
    writeln!(w, "{}", frame.step)?;
    writeln!(w, "ITEM: NUMBER OF ATOMS")?;
    writeln!(w, "{}", frame.n())?;
    writeln!(w, "ITEM: BOX BOUNDS {} {} {}", tok(0), tok(1), tok(2))?;
    for d in 0..3 {
        writeln!(w, "{} {}", bx.low[d], bx.high[d])?;
    }

    let mut cols = "id type x y z vx vy vz fx fy fz radius".to_string();
    append_column_names(&mut cols, frame, ' ');
    writeln!(w, "ITEM: ATOMS {}", cols)?;

    for i in 0..frame.n() {
        write!(
            w,
            "{} {} {} {} {} {} {} {} {} {} {} {}",
            frame.tag[i], frame.atom_type[i],
            frame.pos[i][0], frame.pos[i][1], frame.pos[i][2],
            frame.vel[i][0], frame.vel[i][1], frame.vel[i][2],
            frame.force[i][0], frame.force[i][1], frame.force[i][2],
            frame.radius[i],
        )?;
        for (_, data) in &frame.scalars {
            write!(w, " {}", data[i])?;
        }
        for (_, data) in &frame.vectors {
            write!(w, " {} {} {}", data[i][0], data[i][1], data[i][2])?;
        }
        if let Some(g) = &frame.is_ghost {
            write!(w, " {}", g[i] as u8)?;
        }
        writeln!(w)?;
    }
    Ok(())
}

// ── Restart write ───────────────────────────────────────────────────────────

/// Write restart (checkpoint) files at the configured interval.
///
/// Each MPI rank writes its own restart file containing only its local atoms.
/// The file format is determined by `[restart] format` (bincode or JSON).
#[allow(clippy::too_many_arguments)]
pub fn write_restart(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    restart_config: Res<RestartConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
) {
    let stage = run_config.current_stage(scheduler_manager.index);
    let interval = stage.overrides.get("restart_interval")
        .and_then(|v| v.as_integer())
        .map(|i| i as usize)
        .unwrap_or(restart_config.interval);
    if interval == 0 {
        return;
    }
    let step = run_state.total_cycle;
    if !step.is_multiple_of(interval) {
        return;
    }

    let rank = comm.rank();
    let base_dir = restart_base_dir(&input);
    fs::create_dir_all(&base_dir).ok();

    let data = RestartData::from_atoms(&atoms, &registry, step);

    if let Err(e) = write_restart_inner(&data, &base_dir, step, rank, &restart_config) {
        eprintln!("WARNING: Restart write failed at step {}: {}", step, e);
    }
}

/// Serialize restart data to disk in the configured format (bincode or JSON).
pub(crate) fn write_restart_inner(
    data: &RestartData,
    base_dir: &str,
    step: usize,
    rank: i32,
    restart_config: &RestartConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    match restart_config.format.as_str() {
        "json" => {
            let filename = format!("{}/restart_{}_rank{}.json", base_dir, step, rank);
            let file = File::create(&filename)?;
            serde_json::to_writer(BufWriter::new(file), data)?;
        }
        _ => {
            let filename = format!("{}/restart_{}_rank{}.bin", base_dir, step, rank);
            let file = File::create(&filename)?;
            bincode::serialize_into(BufWriter::new(file), data)?;
        }
    }
    Ok(())
}

// ── Stage-end save ──────────────────────────────────────────────────────────

/// Write dump + restart files when a stage with `save_at_end = true` finishes.
///
/// Runs before `update_cycle` so the stage index is still valid. This ensures
/// that the final state of each stage is captured even if the regular dump/restart
/// intervals don't align with the stage boundary.
#[allow(clippy::too_many_arguments)]
pub fn check_stage_end_save(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    dump_config: Res<DumpConfig>,
    restart_config: Res<RestartConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    dump_registry: Res<DumpRegistry>,
) {
    let index = scheduler_manager.index;
    if index >= run_config.num_stages() {
        return;
    }
    let stage = run_config.current_stage(index);
    if !stage.save_at_end {
        return;
    }

    let remaining = run_state.cycle_remaining[index];
    // Don't save for skipped stages (remaining == 0)
    if remaining == 0 {
        return;
    }

    // check_stage_end_save runs .before("update_cycle"), so the cycle counter
    // hasn't been incremented for the current step yet.  After update_cycle runs,
    // count will be count+1, which equals remaining on the final step.
    let count = run_state.cycle_count[index];
    let is_last_step = count + 1 == remaining;
    let is_advancing = scheduler_manager.advance_requested;

    if !is_last_step && !is_advancing {
        return;
    }

    let step = run_state.total_cycle;
    let rank = comm.rank();
    let stage_label = stage.name.as_deref().unwrap_or("(unnamed)");

    if rank == 0 {
        println!("Stage {} [{}] finished — saving dump + restart at step {}", index, stage_label, step);
    }

    // Write dump
    if let Err(e) = dump_atoms_inner(&atoms, &registry, &comm, step, &input, &dump_config, &dump_registry) {
        eprintln!("WARNING: Stage-end dump write failed at step {}: {}", step, e);
    }

    // Write restart
    let base_dir = restart_base_dir(&input);
    fs::create_dir_all(&base_dir).ok();

    let data = RestartData::from_atoms(&atoms, &registry, step);

    if let Err(e) = write_restart_inner(&data, &base_dir, step, rank, &restart_config) {
        eprintln!("WARNING: Stage-end restart write failed at step {}: {}", step, e);
    }
}

// ── Restart read ────────────────────────────────────────────────────────────

/// Read the latest restart file at startup and restore atom state.
///
/// Scans the restart directory for files matching the current rank and format,
/// selects the one with the highest step number, and deserializes it to restore
/// all atom data (positions, velocities, forces, mass, radius, and any `AtomData`
/// extensions stored in `atom_data_buffers`).
///
/// Only runs when `[restart] read = true` and only on the first stage.
pub fn read_restart(
    restart_config: Res<RestartConfig>,
    comm: Res<CommResource>,
    input: Res<Input>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    mut run_state: ResMut<RunState>,
) {
    if !restart_config.read {
        return;
    }

    let rank = comm.rank();
    let base_dir = restart_base_dir(&input);

    // Find the latest restart file for this rank
    let ext = match restart_config.format.as_str() {
        "json" => "json",
        _ => "bin",
    };

    let prefix = "restart_";
    let suffix = format!("_rank{}.{}", rank, ext);

    let mut latest_step: Option<usize> = None;
    if let Ok(entries) = fs::read_dir(&base_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(prefix) && name.ends_with(&suffix) {
                let mid = &name[prefix.len()..name.len() - suffix.len()];
                if let Ok(step) = mid.parse::<usize>() {
                    if latest_step.map_or(true, |prev| step > prev) {
                        latest_step = Some(step);
                    }
                }
            }
        }
    }

    let step = match latest_step {
        Some(s) => s,
        None => {
            if rank == 0 {
                println!("Restart: no restart files found in {}", base_dir);
            }
            return;
        }
    };

    let filename = format!("{}/restart_{}_rank{}.{}", base_dir, step, rank, ext);
    if rank == 0 {
        println!("Restart: reading from {}", filename);
    }

    let file = File::open(&filename).unwrap_or_else(|e| {
        eprintln!("ERROR: Failed to open restart file '{}': {}", filename, e);
        std::process::exit(1);
    });
    let data: RestartData = match ext {
        "json" => serde_json::from_reader(std::io::BufReader::new(file)).unwrap_or_else(|e| {
            eprintln!("ERROR: Failed to parse restart JSON '{}': {}", filename, e);
            std::process::exit(1);
        }),
        _ => bincode::deserialize_from(std::io::BufReader::new(file)).unwrap_or_else(|e| {
            eprintln!("ERROR: Failed to deserialize restart bincode '{}': {}", filename, e);
            std::process::exit(1);
        }),
    };

    let n = data.tag.len();

    // Clear existing atoms and repopulate from restart data
    atoms.natoms = data.natoms;
    atoms.nlocal = n as u32;
    atoms.nghost = 0;
    atoms.dt = data.dt;

    atoms.tag = data.tag;
    atoms.atom_type = data.atom_type;
    atoms.origin_index = vec![0; n];
    atoms.is_ghost = vec![false; n];
    atoms.pos = data.pos_x.iter().zip(data.pos_y.iter()).zip(data.pos_z.iter())
        .map(|((&x, &y), &z)| [x, y, z]).collect();
    atoms.vel = data.vel_x.iter().zip(data.vel_y.iter()).zip(data.vel_z.iter())
        .map(|((&x, &y), &z)| [x, y, z]).collect();
    atoms.force = data.force_x.iter().zip(data.force_y.iter()).zip(data.force_z.iter())
        .map(|((&x, &y), &z)| [x, y, z]).collect();
    atoms.mass = data.mass;
    atoms.inv_mass = atoms.mass.iter().map(|&m| 1.0 / m).collect();
    atoms.cutoff_radius = data.cutoff_radius;

    // Restore AtomData (DemAtom, etc.) from generic buffers
    if !data.atom_data_buffers.is_empty() {
        registry.unpack_all_from_restart(&data.atom_data_buffers);
    }

    run_state.total_cycle = data.total_cycle;
    if rank == 0 {
        println!("Restart: loaded {} atoms from step {}", n, data.total_cycle);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_columns_backward_compat() {
        let config = ThermoConfig::default();
        assert!(config.columns.is_none());
        let cols = config.columns.unwrap_or_else(default_columns);
        assert_eq!(cols, vec!["step", "atoms", "ke", "neighbors", "walltime", "stepps"]);
    }

    #[test]
    fn test_column_parsing() {
        let col = parse_thermo_column("temp/mobile");
        assert_eq!(col.compute_name, "temp");
        assert_eq!(col.group_name.as_deref(), Some("mobile"));
        assert_eq!(col.header, "Temp/mobile");

        let col2 = parse_thermo_column("step");
        assert_eq!(col2.compute_name, "step");
        assert!(col2.group_name.is_none());
        assert_eq!(col2.header, "Step");
    }

    #[test]
    fn test_user_value_set_and_read() {
        let mut thermo = Thermo::new();
        assert!(thermo.values.get("pe").is_none());
        thermo.set("pe", 42.0);
        assert_eq!(*thermo.values.get("pe").unwrap(), 42.0);
        thermo.set("pe", 99.0);
        assert_eq!(*thermo.values.get("pe").unwrap(), 99.0);
    }

    #[test]
    fn test_capitalize() {
        assert_eq!(capitalize("step"), "Step");
        assert_eq!(capitalize("ke"), "Ke");
        assert_eq!(capitalize(""), "");
        assert_eq!(capitalize("a"), "A");
    }

    #[test]
    fn test_dump_registry_has_callbacks() {
        let reg = DumpRegistry::new();
        assert!(!reg.has_callbacks());
        reg.register_scalar("test", |_atoms, _reg| vec![]);
        assert!(reg.has_callbacks());
    }

    #[test]
    fn test_column_width_minimum() {
        let col = parse_thermo_column("ke");
        // "Ke" is 2 chars, but minimum width is 12
        assert_eq!(col.width, 12);
    }

    #[test]
    fn test_column_width_long_header() {
        let col = parse_thermo_column("virial_xx/long_group_name");
        // "Virial_xx/long_group_name" is 25 chars > 12
        assert!(col.width >= 12);
        assert_eq!(col.width, col.header.len());
    }
}
