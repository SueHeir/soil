//! Neighbor list construction for particle simulations.
//!
//! Uses spatial binning with a precomputed stencil of neighbor cells for O(N)
//! expected neighbor finding. Produces a **half neighbor list** in CSR
//! (Compressed Sparse Row) format: each local atom `i` stores its neighbor
//! indices `j > i` (or ghost atoms) in a flat array, with offsets delimiting
//! each atom's neighbors. Use [`Neighbor::pairs()`] to iterate over `(i, j)`
//! pairs efficiently.
//!
//! # Rebuild strategies
//!
//!
//! Neighbor lists are rebuilt based on configurable criteria:
//!
//! - **Displacement-based** (`every = 0`): rebuilds when any atom moves more than
//!   `(skin_fraction - 1) * cutoff_radius_i` since the last build.
//! - **Periodic** (`every = N`): rebuilds every N steps.
//! - **Hybrid** (`every = N, check = true`): rebuilds every N steps OR on displacement,
//!   whichever comes first (like LAMMPS `neigh_modify every N check yes`).
//!
//! # Configuration
//!
//! Configure via the `[neighbor]` TOML section (see [`NeighborConfig`]).

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    Atom, CommResource, CommState, Config, Domain, ParticleSimScheduleSet, ScheduleSetupSet,
};

mod binning;
mod csr;
mod rebuild;

pub use binning::{bin_neighbor_list, sort_atoms_by_bin};
pub use csr::PairIter;
pub use rebuild::decide_rebuild;

fn default_one_f64() -> f64 {
    1.0
}
fn default_zero_usize() -> usize {
    0
}
fn default_true() -> bool {
    true
}
fn default_sort_every() -> usize {
    1000
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[neighbor]` — neighbor list rebuild and binning settings.
pub struct NeighborConfig {
    /// Multiplier on pairwise cutoff for neighbor skin distance.
    #[serde(default = "default_one_f64")]
    pub skin_fraction: f64,
    /// Minimum bin size for bin-based neighbor lists.
    #[serde(default = "default_one_f64")]
    pub bin_size: f64,
    /// Rebuild every N steps (0 = displacement-based only)
    #[serde(default = "default_zero_usize")]
    pub every: usize,
    /// When true and every > 0, also check displacement threshold (like LAMMPS "check yes")
    #[serde(default = "default_true")]
    pub check: bool,
    /// Sort atoms by spatial bin every N steps for cache locality (0 = disabled).
    #[serde(default = "default_sort_every")]
    pub sort_every: usize,
    /// Newton's third law optimization: true = half neighbor list + reverse comm,
    /// false = full neighbor list, no reverse comm.
    #[serde(default = "default_true")]
    pub newton: bool,
}

impl Default for NeighborConfig {
    fn default() -> Self {
        NeighborConfig {
            skin_fraction: 1.0,
            bin_size: 1.0,
            every: 0,
            check: true,
            sort_every: 1000,
            newton: true,
        }
    }
}

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
        }
    }
}

/// Plugin that registers neighbor list construction and rebuild systems.
///
/// Uses spatial binning with CSR layout for O(N) expected neighbor finding.
///
/// # Systems registered
///
/// - **Setup**: [`neighbor_read_input`] (reads `[neighbor]` config) and
///   [`neighbor_setup`] (computes bin grid, ghost cutoff).
/// - **Update**: [`decide_rebuild`] (displacement check),
///   [`sort_atoms_by_bin`] (cache-locality reordering),
///   [`bin_neighbor_list`] (bin-based neighbor construction).
pub struct NeighborPlugin;

impl Plugin for NeighborPlugin {
    fn provides(&self) -> Vec<&str> {
        vec!["neighbor_list"]
    }

    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[neighbor]
# Skin fraction multiplier for neighbor list cutoff
skin_fraction = 1.0
# Bin size for bin-based neighbor list
bin_size = 1.0
# Rebuild every N steps (0 = displacement-based only)
every = 0
# Also check displacement when every > 0 (like LAMMPS "check yes")
check = true
# Sort atoms by spatial bin every N steps for cache locality (0 = disabled)
sort_every = 1000
# Newton's third law optimization (true = half list + reverse comm, false = full list)
newton = true"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<NeighborConfig>(app, "neighbor");

        app.add_resource(Neighbor::new())
            .add_resource(CurrentState(CommState::FullRebuild))
            .add_setup_system(neighbor_read_input, ScheduleSetupSet::Setup)
            .add_setup_system(
                neighbor_setup.label("neighbor_setup"),
                ScheduleSetupSet::PostSetup,
            );

        app.add_update_system(
            decide_rebuild
                .label("decide_rebuild")
                .before(crate::remove_ghost_atoms)
                .run_if(in_state(CommState::CommunicateOnly)),
            ParticleSimScheduleSet::PostInitialIntegration,
        );
        app.add_update_system(
            sort_atoms_by_bin.label("sort_atoms").before(crate::borders),
            ParticleSimScheduleSet::PreNeighbor,
        );
        app.add_update_system(
            bin_neighbor_list.run_if(in_state(CommState::FullRebuild)),
            ParticleSimScheduleSet::Neighbor,
        );
    }
}

/// Setup system: reads `[neighbor]` config values into the [`Neighbor`] resource.
///
/// Runs at [`ScheduleSetupSet::Setup`]. Prints the rebuild strategy on rank 0.
pub fn neighbor_read_input(
    config: Res<NeighborConfig>,
    mut neighbor: ResMut<Neighbor>,
    comm: Res<CommResource>,
) {
    neighbor.skin_fraction = config.skin_fraction;
    neighbor.bin_min_size = config.bin_size;
    neighbor.every = config.every;
    neighbor.check = config.check;
    neighbor.sort_every = config.sort_every;
    neighbor.newton = config.newton;
    if comm.rank() == 0 {
        let rebuild_str = if config.every == 0 {
            "displacement".to_string()
        } else if config.check {
            format!("every {} + check", config.every)
        } else {
            format!("every {}", config.every)
        };
        println!(
            "Neighbor: skin_fraction={} bin_size={} rebuild={} newton={}",
            config.skin_fraction,
            config.bin_size,
            rebuild_str,
            if config.newton { "on" } else { "off" }
        );
    }
}

/// Setup system: computes bin grid, ghost cutoff, and stencil from atom cutoff radii.
///
/// Runs at [`ScheduleSetupSet::PostSetup`] after atoms are created. Determines:
/// - `max_cutoff = 2 * max_skin * skin_fraction` (largest pairwise neighbor distance)
/// - `ghost_cutoff = max_cutoff + displacement_buffer` (communication distance for ghosts)
/// - Bin grid dimensions, stencil offsets, and PBC flags
pub fn neighbor_setup(
    _config: Res<NeighborConfig>,
    mut neighbor: ResMut<Neighbor>,
    mut domain: ResMut<Domain>,
    atoms: ResMut<Atom>,
    comm: Res<CommResource>,
) {
    // Compute max neighbor cutoff = (skin_i + skin_j) * skin_fraction = 2 * max_skin * skin_fraction
    // Use global reduction: at PostSetup, atoms may only be on rank 0 (before exchange).
    let local_max_skin = atoms
        .cutoff_radius
        .iter()
        .map(|&c| c as f64)
        .fold(0.0f64, f64::max);
    let max_skin = -comm.all_reduce_min_f64(-local_max_skin); // global max via negated min
                                                              // When no particles exist yet (e.g. rate-based insertion), fall back to bin_size
                                                              // so ghost_cutoff is sensible. The cutoff will be updated on first neighbor rebuild.
    let max_cutoff = if max_skin > 0.0 {
        2.0 * max_skin * neighbor.skin_fraction
    } else {
        neighbor.bin_min_size
    };
    // Add displacement buffer to ghost_cutoff so atoms don't drift in/out of the
    // ghost zone between neighbor rebuilds. Without this padding, ghost count
    // fluctuates every step, forcing unnecessary neighbor rebuilds.
    // Max per-atom displacement before rebuild = (skin_fraction - 1) * radius_i.
    // Two largest atoms can each move by the largest budget, so pad the globally
    // largest skinned pair cutoff by 2 * max_budget.
    let displacement_buffer = if max_skin > 0.0 {
        (neighbor.skin_fraction - 1.0) * max_skin
    } else {
        0.0
    };
    let ghost_cut = (max_cutoff + 2.0 * displacement_buffer).max(domain.ghost_cutoff);
    neighbor.ghost_cutoff = ghost_cut;
    neighbor.cached_max_cutoff = max_cutoff;
    domain.ghost_cutoff = ghost_cut;
    if comm.rank() == 0 {
        println!(
            "Neighbor: ghost_cutoff={:.4} (pair_cutoff={:.4} + buffer={:.4})",
            ghost_cut,
            max_cutoff,
            2.0 * displacement_buffer
        );
    }
    // Single-process: bin_size >= cutoff gives stencil range=1, keeping bin_start
    // small enough (< 64KB) for L1 cache and only 13 forward stencil cells.
    // Multi-process: use cutoff/2 so subdomains have enough bins for correct stencil.
    let required_bin = if comm.size() > 1 {
        max_cutoff * 0.5
    } else {
        max_cutoff
    };
    if neighbor.bin_min_size < required_bin {
        neighbor.bin_min_size = required_bin;
    }

    let min_bin = neighbor.bin_min_size;
    if (domain.sub_length[0] / min_bin < 1.0
        || domain.sub_length[1] / min_bin < 1.0
        || domain.sub_length[2] / min_bin < 1.0)
        && comm.rank() == 0
    {
        println!(
            "WARNING: subdomain smaller than bin_size in at least one dimension, clamping to 1 bin"
        );
    }

    // Compute bin grid, stencil, and PBC box using shared helper
    binning::compute_bin_grid(&mut neighbor, &domain, comm.size());
    neighbor.pbc_flags = domain.periodic_flags();

    if comm.rank() == 0 {
        println!(
            "Neighbor: bins {}x{}x{} (with ghost layers), {} forward stencil cells",
            neighbor.bin_count[0],
            neighbor.bin_count[1],
            neighbor.bin_count[2],
            neighbor.bin_stencil_forward.len()
        );
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Atom;

    #[test]
    fn classify_boundary_atoms_flags_ghost_neighbours() {
        // 3 local atoms (0,1,2) + 1 ghost (3). CSR neighbour lists:
        //   atom 0 -> {1}        interior
        //   atom 1 -> {0, 3}     boundary (3 is a ghost)
        //   atom 2 -> {}         interior (no neighbours)
        let mut nb = Neighbor::new();
        nb.neighbor_offsets = vec![0, 1, 3, 3];
        nb.neighbor_indices = vec![1, 0, 3];
        let mask = classify_boundary_atoms(&nb, 3);
        assert_eq!(mask, vec![false, true, false]);
    }

    fn push_atom(atom: &mut Atom, tag: u32, pos: [f64; 3], radius: f64) {
        atom.push_test_atom(tag, pos, radius, 1.0);
    }

    #[test]
    fn bin_neighbor_list_finds_close_pair() {
        let mut app = App::new();
        let mut atom = Atom::new();
        push_atom(&mut atom, 0, [0.5, 0.5, 0.5], 0.5);
        push_atom(&mut atom, 1, [1.0, 0.5, 0.5], 0.5);
        atom.nlocal = 2;
        atom.natoms = 2;

        let mut domain = crate::Domain::new();
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [2.0, 2.0, 2.0];
        domain.sub_length = [2.0, 2.0, 2.0];

        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = 1.0;
        neighbor.bin_min_size = 1.0;

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(domain);
        app.add_resource(NeighborConfig::default());
        app.add_resource(CurrentState(CommState::FullRebuild));
        app.add_resource(crate::CommResource(Box::new(
            crate::SingleProcessComm::new(),
        )));
        app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
        app.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
        app.organize_systems();
        app.setup();
        app.run();

        let n = app.get_resource_ref::<Neighbor>().unwrap();
        assert!(
            n.neighbor_indices.len() >= 1,
            "bin neighbor list should find the close pair"
        );
        // Check CSR: atom 0's neighbors should contain 1
        let start = n.neighbor_offsets[0] as usize;
        let end = n.neighbor_offsets[1] as usize;
        let has_pair = n.neighbor_indices[start..end].contains(&1u32);
        assert!(has_pair, "pair (0,1) should be in CSR neighbors");
    }

    #[test]
    fn pair_iter_matches_manual() {
        let mut neighbor = Neighbor::new();
        // 3 local atoms: atom 0 -> [1, 2], atom 1 -> [2], atom 2 -> []
        neighbor.neighbor_offsets = vec![0, 2, 3, 3];
        neighbor.neighbor_indices = vec![1, 2, 2];

        let pairs: Vec<(usize, usize)> = neighbor.pairs(3).collect();
        assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn pair_iter_empty() {
        let mut neighbor = Neighbor::new();
        neighbor.neighbor_offsets = vec![0, 0, 0];
        neighbor.neighbor_indices = vec![];

        let pairs: Vec<(usize, usize)> = neighbor.pairs(2).collect();
        assert!(pairs.is_empty());

        // Also test zero local atoms
        let pairs2: Vec<(usize, usize)> = neighbor.pairs(0).collect();
        assert!(pairs2.is_empty());
    }

    // ── Reference vs bin-based neighbor list comparison ──────────────────

    /// Reference O(N²) all-pairs neighbor finder that doesn't use ghost atoms
    /// or any accelerated algorithm. Pure distance-based cutoff check.
    fn reference_all_pairs(
        positions: &[[f64; 3]],
        cutoff_radii: &[f64],
        skin_fraction: f64,
        nlocal: usize,
    ) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..nlocal {
            for j in (i + 1)..positions.len() {
                let dx = positions[j][0] - positions[i][0];
                let dy = positions[j][1] - positions[i][1];
                let dz = positions[j][2] - positions[i][2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let cutoff = (cutoff_radii[i] + cutoff_radii[j]) * skin_fraction;
                if dist < cutoff {
                    pairs.push((i, j));
                }
            }
        }
        pairs.sort();
        pairs
    }

    #[test]
    fn bin_neighbor_list_matches_reference() {
        // Create a system and verify that bin-based neighbor list finds
        // exactly the same pairs as the reference all-pairs calculation.
        let n = 30;
        let skin_fraction = 1.0;
        let radius = 0.3;

        let mut positions = Vec::new();
        for i in 0..n {
            let x = (i as f64 * 0.31).sin() * 1.5 + 2.0;
            let y = (i as f64 * 0.67).cos() * 1.5 + 2.0;
            let z = (i as f64 * 0.97).sin() * 1.5 + 2.0;
            positions.push([x, y, z]);
        }

        // Reference all-pairs
        let cutoffs: Vec<f64> = vec![radius; n];
        let ref_pairs = reference_all_pairs(&positions, &cutoffs, skin_fraction, n);

        // Bin-based neighbor list
        let mut app_bin = App::new();
        let mut atom_bin = Atom::new();
        for i in 0..n {
            push_atom(&mut atom_bin, i as u32, positions[i], radius);
        }
        atom_bin.nlocal = n as u32;
        atom_bin.natoms = n as u64;

        let mut domain = crate::Domain::new();
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [4.0, 4.0, 4.0];
        domain.sub_length = [4.0, 4.0, 4.0];

        let mut neighbor_bin = Neighbor::new();
        neighbor_bin.skin_fraction = skin_fraction;
        neighbor_bin.bin_min_size = 0.5;

        app_bin.add_resource(atom_bin);
        app_bin.add_resource(neighbor_bin);
        app_bin.add_resource(domain);
        app_bin.add_resource(NeighborConfig::default());
        app_bin.add_resource(CurrentState(CommState::FullRebuild));
        app_bin.add_resource(crate::CommResource(Box::new(
            crate::SingleProcessComm::new(),
        )));
        app_bin.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
        app_bin.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
        app_bin.organize_systems();
        app_bin.setup();
        app_bin.run();

        let neigh_bin = app_bin.get_resource_ref::<Neighbor>().unwrap();
        // Extract pairs from CSR format
        let nlocal = n;
        let mut bin_pairs = Vec::new();
        for i in 0..nlocal {
            if i + 1 >= neigh_bin.neighbor_offsets.len() {
                break;
            }
            let start = neigh_bin.neighbor_offsets[i] as usize;
            let end = neigh_bin.neighbor_offsets[i + 1] as usize;
            for k in start..end {
                let j = neigh_bin.neighbor_indices[k] as usize;
                let pair = if i < j { (i, j) } else { (j, i) };
                bin_pairs.push(pair);
            }
        }
        bin_pairs.sort();
        bin_pairs.dedup();

        // Verify: bin-based should find all pairs that reference finds
        for pair in &ref_pairs {
            assert!(
                bin_pairs.contains(pair),
                "Bin neighbor list missed pair {:?} found by reference",
                pair
            );
        }
    }

    #[test]
    fn full_bin_neighbor_list_scans_self_cell_without_self_pairs() {
        // Dense same-cell case for the newton=false path: every atom is inside
        // the self-cell cutoff, so the full neighbor list should contain both
        // directions of each pair and no (i, i) self entries.
        let mut app = App::new();
        let mut atom = Atom::new();
        push_atom(&mut atom, 0, [0.50, 0.50, 0.50], 0.5);
        push_atom(&mut atom, 1, [0.75, 0.50, 0.50], 0.5);
        push_atom(&mut atom, 2, [0.50, 0.75, 0.50], 0.5);
        atom.nlocal = 3;
        atom.natoms = 3;

        let mut domain = crate::Domain::new();
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [2.0, 2.0, 2.0];
        domain.sub_length = [2.0, 2.0, 2.0];

        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = 1.0;
        neighbor.bin_min_size = 1.0;
        neighbor.newton = false;

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(domain);
        app.add_resource(NeighborConfig {
            newton: false,
            ..NeighborConfig::default()
        });
        app.add_resource(CurrentState(CommState::FullRebuild));
        app.add_resource(crate::CommResource(Box::new(
            crate::SingleProcessComm::new(),
        )));
        app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
        app.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
        app.organize_systems();
        app.setup();
        app.run();

        let neigh = app.get_resource_ref::<Neighbor>().unwrap();
        let mut directed_pairs = Vec::new();
        for i in 0..3 {
            let start = neigh.neighbor_offsets[i] as usize;
            let end = neigh.neighbor_offsets[i + 1] as usize;
            for &j in &neigh.neighbor_indices[start..end] {
                directed_pairs.push((i, j as usize));
            }
        }
        directed_pairs.sort();
        assert_eq!(
            directed_pairs,
            vec![(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        );
    }

    fn full_neighbor_directed_pairs(neighbor: &Neighbor, nlocal: usize) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..nlocal {
            let start = neighbor.neighbor_offsets[i] as usize;
            let end = neighbor.neighbor_offsets[i + 1] as usize;
            for &j in &neighbor.neighbor_indices[start..end] {
                pairs.push((i, j as usize));
            }
        }
        pairs.sort();
        pairs
    }

    fn legacy_full_neighbor_directed_pairs(
        atom: &Atom,
        neighbor: &Neighbor,
        nlocal: usize,
        cutoffs: &[f64],
    ) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        let skin_fraction_sq = neighbor.skin_fraction * neighbor.skin_fraction;
        for i in 0..nlocal {
            let my_cell = neighbor.bin_atom_cell[i] as usize;
            let pi = atom.pos[i];
            let pi = [pi[0] as f64, pi[1] as f64, pi[2] as f64];
            for &offset in &neighbor.bin_stencil {
                let c = (my_cell as i32 + offset) as usize;
                let start = neighbor.bin_start[c] as usize;
                let end = neighbor.bin_start[c + 1] as usize;
                for m in start..end {
                    let j = neighbor.bin_sorted_atoms[m] as usize;
                    if offset == 0 && j == i {
                        continue;
                    }
                    let pj = neighbor.bin_sorted_pos[m];
                    let dx = pj[0] - pi[0];
                    let dy = pj[1] - pi[1];
                    let dz = pj[2] - pi[2];
                    let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                    let sum_skin = cutoffs[i] + cutoffs[j];
                    if r2 < sum_skin * sum_skin * skin_fraction_sq {
                        pairs.push((i, j));
                    }
                }
            }
        }
        pairs.sort();
        pairs
    }

    #[test]
    fn full_bin_neighbor_list_matches_legacy_self_cell_scan() {
        // Regression for the newton=false full-list path: production code scans
        // the self cell separately and uses a stencil with offset 0 removed. This
        // must produce the same directed pair set as the legacy full-stencil scan
        // with hot-loop `offset == 0 && j == i` filtering.
        let n = 12;
        let skin_fraction = 1.0;
        let positions = [
            [0.20, 0.20, 0.20],
            [0.55, 0.22, 0.20],
            [0.20, 0.58, 0.20],
            [0.62, 0.60, 0.20],
            [1.05, 0.24, 0.20],
            [1.38, 0.22, 0.20],
            [0.22, 1.05, 0.20],
            [0.58, 1.35, 0.20],
            [1.08, 1.08, 0.20],
            [1.42, 1.38, 0.20],
            [2.20, 0.20, 0.20],
            [2.55, 0.22, 0.20],
        ];
        let cutoffs = [
            0.30, 0.36, 0.28, 0.34, 0.32, 0.27, 0.35, 0.31, 0.29, 0.37, 0.30, 0.33,
        ];

        let mut app = App::new();
        let mut atom = Atom::new();
        for i in 0..n {
            push_atom(&mut atom, i as u32, positions[i], cutoffs[i]);
        }
        atom.nlocal = n as u32;
        atom.natoms = n as u64;

        let mut domain = crate::Domain::new();
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [3.0, 2.0, 2.0];
        domain.sub_length = [3.0, 2.0, 2.0];

        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = skin_fraction;
        neighbor.bin_min_size = 0.5;
        neighbor.newton = false;

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(domain);
        app.add_resource(NeighborConfig {
            newton: false,
            ..NeighborConfig::default()
        });
        app.add_resource(CurrentState(CommState::FullRebuild));
        app.add_resource(crate::CommResource(Box::new(
            crate::SingleProcessComm::new(),
        )));
        app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
        app.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
        app.organize_systems();
        app.setup();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        let neighbor = app.get_resource_ref::<Neighbor>().unwrap();
        let optimized_pairs = full_neighbor_directed_pairs(&neighbor, n);
        let legacy_pairs = legacy_full_neighbor_directed_pairs(&atom, &neighbor, n, &cutoffs);

        assert_eq!(optimized_pairs, legacy_pairs);
        assert!(!optimized_pairs.iter().any(|(i, j)| i == j));
        assert!(
            optimized_pairs.iter().any(|(i, j)| i > j)
                && optimized_pairs.iter().any(|(i, j)| i < j),
            "newton=false full list should contain both pair directions"
        );
    }

    #[test]
    fn triclinic_bin_neighbor_list_matches_reference() {
        // Same cross-check as above but with a TILTED (triclinic) box. Atoms are
        // kept in the interior so no periodic/ghost pairs are needed — this isolates
        // the lamda binning + triclinic stencil: it must still find every real-space
        // pair within cutoff (no misses from a too-narrow sheared stencil).
        let n = 40;
        let skin_fraction = 1.0;
        let radius = 0.3;
        let xy = 1.0; // box tilt (0.25*Lx, within the box-flip bound)

        let mut positions = Vec::new();
        for i in 0..n {
            // Spread through the interior [0.6, 3.4]^3 so no wrap is required.
            let x = (i as f64 * 0.31).sin() * 1.4 + 2.0;
            let y = (i as f64 * 0.67).cos() * 1.4 + 2.0;
            let z = (i as f64 * 0.97).sin() * 1.4 + 2.0;
            positions.push([x, y, z]);
        }

        let cutoffs: Vec<f64> = vec![radius; n];
        let ref_pairs = reference_all_pairs(&positions, &cutoffs, skin_fraction, n);

        let mut app = App::new();
        let mut atom = Atom::new();
        for i in 0..n {
            push_atom(&mut atom, i as u32, positions[i], radius);
        }
        atom.nlocal = n as u32;
        atom.natoms = n as u64;

        let mut domain = crate::Domain::new();
        domain.boundaries_low = [0.0, 0.0, 0.0];
        domain.boundaries_high = [4.0, 4.0, 4.0];
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [4.0, 4.0, 4.0];
        domain.size = [4.0, 4.0, 4.0];
        domain.sub_length = [4.0, 4.0, 4.0];
        domain.sub_lamda_low = [0.0, 0.0, 0.0];
        domain.sub_lamda_high = [1.0, 1.0, 1.0];
        domain.triclinic = true;
        domain.tilt = [xy, 0.0, 0.0];

        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = skin_fraction;
        neighbor.bin_min_size = 0.5;

        app.add_resource(atom);
        app.add_resource(neighbor);
        app.add_resource(domain);
        app.add_resource(NeighborConfig::default());
        app.add_resource(CurrentState(CommState::FullRebuild));
        app.add_resource(crate::CommResource(Box::new(
            crate::SingleProcessComm::new(),
        )));
        app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
        app.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
        app.organize_systems();
        app.setup();
        app.run();

        let neigh = app.get_resource_ref::<Neighbor>().unwrap();
        let mut bin_pairs = Vec::new();
        for i in 0..n {
            if i + 1 >= neigh.neighbor_offsets.len() {
                break;
            }
            let start = neigh.neighbor_offsets[i] as usize;
            let end = neigh.neighbor_offsets[i + 1] as usize;
            for k in start..end {
                let j = neigh.neighbor_indices[k] as usize;
                bin_pairs.push(if i < j { (i, j) } else { (j, i) });
            }
        }
        bin_pairs.sort();
        bin_pairs.dedup();

        for pair in &ref_pairs {
            assert!(
                bin_pairs.contains(pair),
                "Triclinic bin neighbor list missed pair {:?} found by reference",
                pair
            );
        }
    }
}
