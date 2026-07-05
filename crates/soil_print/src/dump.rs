use std::{
    fs,
    fs::File,
    io::{BufWriter, Write},
};

use grass_scheduler::prelude::*;
use serde::{Deserialize, Serialize};
use soil_core::{Atom, AtomDataRegistry, CommResource, Domain, Input, RunConfig, RunState};

use crate::registry::{BoxInfo, DumpFrame, DumpRegistry};

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

/// Output system: build a [`DumpFrame`] from the current atom state (optionally
/// MPI-gathering to rank 0) and write it via the registered format writers when
/// the configured dump interval fires.
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
    let interval = stage
        .overrides
        .get("dump_interval")
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

    if let Err(e) = dump_atoms_inner(
        &atoms,
        &registry,
        &comm,
        step,
        &input,
        &dump_config,
        &dump_registry,
    ) {
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
    let count = if dump_config.ghost {
        atoms.len()
    } else {
        nlocal
    };
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
        // DumpFrame keeps an f64 output format independent of build precision.
        frame.pos.push([
            atoms.pos[i][0] as f64,
            atoms.pos[i][1] as f64,
            atoms.pos[i][2] as f64,
        ]);
        frame.vel.push([
            atoms.vel[i][0] as f64,
            atoms.vel[i][1] as f64,
            atoms.vel[i][2] as f64,
        ]);
        frame.force.push([
            atoms.force[i][0] as f64,
            atoms.force[i][1] as f64,
            atoms.force[i][2] as f64,
        ]);
        frame.radius.push(atoms.cutoff_radius[i] as f64);
        if let Some(g) = &mut frame.is_ghost {
            g.push(i >= nlocal);
        }
    }
    let pad_scalar = |data: &[f64]| {
        (0..count)
            .map(|i| data.get(i).copied().unwrap_or(0.0))
            .collect()
    };
    let pad_vector = |data: &[[f64; 3]]| {
        (0..count)
            .map(|i| data.get(i).copied().unwrap_or([0.0; 3]))
            .collect()
    };
    frame.scalars = scalar_data
        .iter()
        .map(|(n, d)| (n.clone(), pad_scalar(d)))
        .collect();
    frame.vectors = vector_data
        .iter()
        .map(|(n, d)| (n.clone(), pad_vector(d)))
        .collect();

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
pub(crate) fn write_dump_text(frame: &DumpFrame) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(format!("{}.csv", frame.path_stem))?);

    let mut header = "tag,type,x,y,z,vx,vy,vz,fx,fy,fz,radius".to_string();
    append_column_names(&mut header, frame, ',');
    writeln!(w, "{}", header)?;

    for i in 0..frame.n() {
        write!(
            w,
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            frame.tag[i],
            frame.atom_type[i],
            frame.pos[i][0],
            frame.pos[i][1],
            frame.pos[i][2],
            frame.vel[i][0],
            frame.vel[i][1],
            frame.vel[i][2],
            frame.force[i][0],
            frame.force[i][1],
            frame.force[i][2],
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
pub(crate) fn write_dump_binary(frame: &DumpFrame) -> std::io::Result<()> {
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
pub(crate) fn write_dump_lammps(frame: &DumpFrame) -> std::io::Result<()> {
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
            frame.tag[i],
            frame.atom_type[i],
            frame.pos[i][0],
            frame.pos[i][1],
            frame.pos[i][2],
            frame.vel[i][0],
            frame.vel[i][1],
            frame.vel[i][2],
            frame.force[i][0],
            frame.force[i][1],
            frame.force[i][2],
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
