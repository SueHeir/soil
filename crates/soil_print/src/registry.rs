use std::cell::RefCell;

use soil_core::{Atom, AtomDataRegistry};

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
    pub(crate) scalar_fns:
        RefCell<Vec<(String, Box<dyn Fn(&Atom, &AtomDataRegistry) -> Vec<f64>>)>>,
    pub(crate) vector_fns: RefCell<
        Vec<(
            String,
            Box<dyn Fn(&Atom, &AtomDataRegistry) -> Vec<[f64; 3]>>,
        )>,
    >,
    pub(crate) format_fns: RefCell<Vec<(String, Box<dyn Fn(&DumpFrame) -> std::io::Result<()>>)>>,
    /// Global box bounds + periodicity, synced from [`Domain`] each step so format
    /// writers can emit a box header without threading `Domain` through every caller.
    box_info: RefCell<BoxInfo>,
}

/// Global simulation-box snapshot for dump headers.
#[derive(Clone, Copy)]
pub struct BoxInfo {
    /// Global lower box corner `[x, y, z]`.
    pub low: [f64; 3],
    /// Global upper box corner `[x, y, z]`.
    pub high: [f64; 3],
    /// Per-axis periodicity flags `[x, y, z]`.
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
    /// Per-atom global tag (unique id).
    pub tag: Vec<u64>,
    /// Per-atom type index.
    pub atom_type: Vec<u64>,
    /// Per-atom position `[x, y, z]`.
    pub pos: Vec<[f64; 3]>,
    /// Per-atom velocity `[vx, vy, vz]`.
    pub vel: Vec<[f64; 3]>,
    /// Per-atom force `[fx, fy, fz]`.
    pub force: Vec<[f64; 3]>,
    /// Per-atom radius (from `cutoff_radius`; `0` for point particles).
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
        BoxInfo {
            low: [0.0; 3],
            high: [1.0; 3],
            periodic: [true; 3],
        }
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
    pub(crate) fn pack(&self) -> Vec<f64> {
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
    pub(crate) fn push_unpacked(&mut self, buf: &[f64]) {
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
        self.scalar_fns
            .borrow_mut()
            .push((name.into(), Box::new(f)));
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
        self.vector_fns
            .borrow_mut()
            .push((name.into(), Box::new(f)));
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
        self.format_fns
            .borrow_mut()
            .push((name.into(), Box::new(f)));
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
        self.format_fns
            .borrow()
            .iter()
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// Returns `true` if any scalar or vector callbacks are registered.
    pub fn has_callbacks(&self) -> bool {
        !self.scalar_fns.borrow().is_empty() || !self.vector_fns.borrow().is_empty()
    }
}
