//! Neighbor-list runtime state and state-only helpers.

use super::NeighborDistanceKernel;

/// Neighbor list state: CSR indices, bin grid, and rebuild tracking.
///
/// The primary output is the CSR neighbor list stored in [`neighbor_offsets`](Self::neighbor_offsets)
/// and [`neighbor_indices`](Self::neighbor_indices). Use [`pairs()`](Self::pairs) to iterate
/// over `(i, j)` neighbor pairs.
pub struct Neighbor {
    /// Multiplier on pairwise cutoff: pair cutoff = `(r_i + r_j) * skin_fraction`.
    /// Values > 1.0 add a "skin" buffer to reduce rebuild frequency.
    pub skin_fraction: f64,
    /// CSR row offsets: `neighbor_offsets[i]..neighbor_offsets[i+1]` gives the range
    /// of neighbor indices for local atom `i`. Length = `nlocal + 1`.
    pub neighbor_offsets: Vec<u32>,
    /// CSR column indices: flat array of neighbor atom indices (local or ghost).
    pub neighbor_indices: Vec<u32>,
    /// User-configured minimum bin size (may be increased to match cutoff).
    pub bin_min_size: f64,
    /// Actual bin dimensions in each axis `[bx, by, bz]`, computed from domain / bin count.
    pub bin_size: [f64; 3],
    /// Number of bins in each axis `[nx, ny, nz]` (including ghost layers).
    pub bin_count: [i32; 3],
    /// Saved atom positions from the last neighbor build, for displacement checking.
    pub last_build_pos: Vec<[f64; 3]>,
    /// Saved local atom cutoff radii from the last neighbor build. Displacement
    /// budgets are tied to the radius used to build the current list.
    pub last_build_radius: Vec<f64>,
    /// Number of timesteps since the last neighbor list rebuild.
    pub steps_since_build: usize,
    /// Total atom count (local + ghost) at the last neighbor build.
    pub last_build_total: usize,
    /// Full simulation box dimensions `[Lx, Ly, Lz]` for minimum-image displacement checks.
    pub pbc_box: [f64; 3],
    /// Which axes have periodic boundary conditions.
    pub pbc_flags: [bool; 3],
    /// Full stencil: flat cell offsets `dx*ny*nz + dy*nz + dz` for all neighbor cells
    /// within cutoff distance (both forward and backward).
    pub bin_stencil: Vec<i32>,
    /// Forward-only stencil: cell offsets with `offset > 0`, used for half-neighbor-list
    /// construction to avoid counting each pair twice.
    pub bin_stencil_forward: Vec<i32>,
    /// Full-list stencil excluding the self-cell offset. Used when `newton = false`
    /// so the offset-zero case is handled by the dedicated self-cell scan instead
    /// of being checked in the neighbor-cell loop.
    pub bin_stencil_nonself: Vec<i32>,
    /// Whether the self-cell (offset 0) passes the stencil distance test.
    pub bin_stencil_self: bool,
    /// Lower-left corner of the bin grid, offset by ghost layers from `sub_domain_low`.
    pub bin_origin: [f64; 3],
    /// Total number of bin cells: `nx * ny * nz` (including ghost layers).
    pub bin_total_cells: usize,
    /// Rebuild every N steps (0 = displacement-based only).
    pub every: usize,
    /// When true and `every > 0`, also check displacement threshold each step.
    pub check: bool,
    /// Communication cutoff for ghost atoms: `pair_cutoff + 2 * displacement_buffer`.
    pub ghost_cutoff: f64,
    /// Smallest cutoff radius among local atoms, cached at rebuild time for
    /// diagnostics and uniform-cutoff fast-path setup.
    pub cached_min_skin: f64,
    /// Reorder atoms by spatial bin every N steps for cache locality (0 = disabled).
    pub sort_every: usize,
    /// Steps elapsed since the last spatial sort.
    pub sort_counter: usize,
    /// Per-atom bin cell index (reused across rebuilds to avoid allocation).
    pub bin_atom_cell: Vec<u32>,
    /// Per-cell atom count, then reused as write cursor during CSR construction.
    pub bin_count_arr: Vec<u32>,
    /// CSR bin offsets: `bin_start[c]..bin_start[c+1]` gives sorted atom range for cell `c`.
    pub bin_start: Vec<u32>,
    /// Atoms sorted by bin cell (indices into the atom arrays).
    pub bin_sorted_atoms: Vec<u32>,
    /// Inverse of `bin_sorted_atoms`: position of atom `i` within `bin_sorted_atoms`.
    /// Used for self-cell skip optimization (start scanning after atom `i`).
    pub bin_atom_sorted_idx: Vec<u32>,
    /// Positions reordered by bin for cache-friendly inner loop access.
    pub bin_sorted_pos: Vec<[f64; 3]>,
    /// Scratch for the O(N) counting sort in `sort_atoms_by_bin`, reused across
    /// rebuilds. Dedicated (not shared with the `bin_*` build buffers) so the
    /// spatial sort stays decoupled from the neighbor build.
    pub sort_cell: Vec<u32>,
    /// Per-cell atom counts for the spatial-sort counting sort (companion to `sort_cell`).
    pub sort_counts: Vec<u32>,
    /// Permutation produced by the spatial sort, applied to the atom arrays.
    pub sort_perm: Vec<usize>,
    /// Scratch positions used while building the spatial-sort permutation.
    pub sort_pos_scratch: Vec<[f64; 3]>,
    /// Scratch cutoff radii used while applying the same spatial-sort permutation
    /// to the saved rebuild state.
    pub sort_radius_scratch: Vec<f64>,
    /// Cycle-visited scratch reused while applying the spatial-sort permutation
    /// to `Atom` fields and all registered `AtomData` extension columns.
    pub sort_permutation_visited: Vec<bool>,
    /// N2: post-permute local cell indices written by `sort_atoms_by_bin`, so the
    /// neighbor build doesn't recompute the cell for every local atom (it already
    /// did, identically, when sorting). Only the boundary ghosts are binned in the
    /// build. `local_cells_valid` gates reuse: it's set true only when the sort ran
    /// and populated this for the current `nlocal`, and cleared once the build
    /// consumes it (so a build with no preceding sort falls back to computing all).
    pub sorted_local_cell: Vec<u32>,
    /// `true` when `sorted_local_cell` was populated by a sort for the current
    /// `nlocal` and may be reused by the build; cleared once the build consumes it.
    pub local_cells_valid: bool,
    /// When all atoms share the same cutoff radius, cache `(2 * r * skin_fraction)²`
    /// to skip per-pair cutoff computation. `None` for polydisperse systems.
    pub cached_uniform_cutoff_sq: Option<f64>,
    /// Max pairwise cutoff from `neighbor_setup`, needed when recomputing bins
    /// after shrink-wrap domain changes.
    pub cached_max_cutoff: f64,
    /// Newton's third law optimization: `true` = half neighbor list + reverse comm,
    /// `false` = full neighbor list, skip reverse comm.
    pub newton: bool,
    /// Distance-filter kernel selected for neighbor construction.
    pub distance_kernel: NeighborDistanceKernel,
}

impl Default for Neighbor {
    fn default() -> Self {
        Self::new()
    }
}

impl Neighbor {
    /// Creates a new `Neighbor` with default values.
    ///
    /// All arrays start empty; the bin grid and stencil are computed during
    /// [`neighbor_setup`] after the simulation domain and atom data are available.
    pub fn new() -> Self {
        Neighbor {
            skin_fraction: 1.0,
            neighbor_offsets: Vec::new(),
            neighbor_indices: Vec::new(),
            bin_min_size: 1.0,
            bin_size: [1.0; 3],
            bin_count: [1, 1, 1],
            last_build_pos: Vec::new(),
            last_build_radius: Vec::new(),
            steps_since_build: 0,
            last_build_total: 0,
            pbc_box: [0.0; 3],
            pbc_flags: [false; 3],
            bin_stencil: Vec::new(),
            bin_stencil_forward: Vec::new(),
            bin_stencil_nonself: Vec::new(),
            bin_stencil_self: false,
            bin_origin: [0.0; 3],
            bin_total_cells: 0,
            every: 0,
            check: true,
            ghost_cutoff: 0.0,
            cached_min_skin: f64::MAX,
            sort_every: 1000,
            sort_counter: 0,
            bin_atom_cell: Vec::new(),
            bin_count_arr: Vec::new(),
            bin_start: Vec::new(),
            bin_sorted_atoms: Vec::new(),
            bin_atom_sorted_idx: Vec::new(),
            bin_sorted_pos: Vec::new(),
            sort_cell: Vec::new(),
            sort_counts: Vec::new(),
            sort_perm: Vec::new(),
            sort_pos_scratch: Vec::new(),
            sort_radius_scratch: Vec::new(),
            sort_permutation_visited: Vec::new(),
            sorted_local_cell: Vec::new(),
            local_cells_valid: false,
            cached_uniform_cutoff_sq: None,
            cached_max_cutoff: 0.0,
            newton: true,
            distance_kernel: NeighborDistanceKernel::Scalar,
        }
    }
}

/// Plugin that registers neighbor list construction and rebuild systems.
///
/// Uses spatial binning with CSR layout for O(N) expected neighbor finding.

/// Classify each LOCAL atom as **boundary** (has at least one ghost neighbour,
/// index >= `nlocal`) or **interior** (all neighbours local). Building block for
/// interior/boundary overlap (roadmap step 4): interior forces need no fresh
/// ghosts, so they can be computed while the halo exchange is in flight; boundary
/// forces are finished once the ghosts land. Returns a `nlocal`-length mask
/// (`true` = boundary).
pub fn classify_boundary_atoms(neighbor: &Neighbor, nlocal: usize) -> Vec<bool> {
    let off = &neighbor.neighbor_offsets;
    let idx = &neighbor.neighbor_indices;
    (0..nlocal)
        .map(|i| {
            let (s, e) = (off[i] as usize, off[i + 1] as usize);
            idx[s..e].iter().any(|&j| j as usize >= nlocal)
        })
        .collect()
}
