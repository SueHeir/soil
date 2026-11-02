//! Neighbor plugin registration and setup systems.

use grass_app::prelude::*;
use grass_scheduler::prelude::*;

use super::{bin_neighbor_list, decide_rebuild, sort_atoms_by_bin, Neighbor, NeighborConfig};
use crate::{
    Atom, CommResource, CommState, Config, Domain, ParticleSimScheduleSet, ScheduleSetupSet,
};

/// Type-level label for [`neighbor_setup`].
pub struct NeighborSetupSystem;
impl SystemLabel for NeighborSetupSystem {
    const NAME: &'static str = "neighbor_setup";
}
/// Stable scheduler key for [`neighbor_setup`].
pub const NEIGHBOR_SETUP: SystemKey<NeighborSetupSystem> = SystemKey::new();

/// Type-level label for [`decide_rebuild`].
pub struct DecideRebuildSystem;
impl SystemLabel for DecideRebuildSystem {
    const NAME: &'static str = "decide_rebuild";
}
/// Stable scheduler key for [`decide_rebuild`].
pub const DECIDE_REBUILD: SystemKey<DecideRebuildSystem> = SystemKey::new();

/// Type-level label for [`sort_atoms_by_bin`].
pub struct SortAtomsSystem;
impl SystemLabel for SortAtomsSystem {
    const NAME: &'static str = "sort_atoms";
}
/// Stable scheduler key for [`sort_atoms_by_bin`].
pub const SORT_ATOMS: SystemKey<SortAtomsSystem> = SystemKey::new();

/// Particle binning without a pair list: the bin grid, per-bin CSR
/// (`bin_start` / `bin_sorted_atoms`), spatial sorting, the rebuild decision,
/// and with it SOIL's exchange and ghost borders. Provides `"particle_bins"`.
///
/// Use this alone for methods whose particles interact through cells or a
/// grid rather than in pairs (DSMC, PIC, MPM). Pairs are built only when
/// [`PairListPlugin`] is also present, so a code never pays for a list it
/// does not read. [`NeighborPlugin`] adds both.
///
/// # Systems registered
///
/// - **Setup**: [`neighbor_read_input`] (reads `[neighbor]` config) and
///   [`neighbor_setup`] (computes bin grid, ghost cutoff).
/// - **Update**: [`decide_rebuild`] (displacement check),
///   [`sort_atoms_by_bin`] (cache-locality reordering),
///   [`bin_neighbor_list`] (binning, plus pairs when requested).
pub struct BinningPlugin;

/// Marker resource installed by [`PairListPlugin`]: build the pair list.
#[derive(Debug, Default, Clone, Copy)]
pub struct PairListRequested;

/// Opts in to the CSR pair list on top of [`BinningPlugin`] and provides
/// `"neighbor_list"`. Any plugin that reads `Neighbor::pairs()`,
/// `neighbors_of` or the raw CSR needs this (directly or via
/// [`NeighborPlugin`]).
pub struct PairListPlugin;

impl Plugin for PairListPlugin {
    fn provides(&self) -> Vec<&str> {
        vec!["neighbor_list"]
    }

    fn build(&self, app: &mut App) {
        app.add_resource(PairListRequested);
    }
}

/// Binning plus the pair list: [`BinningPlugin`] + [`PairListPlugin`]. The
/// right choice for pairwise methods (DEM, SPH, MD, peridynamics family
/// builds), and what every existing SOIL app already uses.
pub struct NeighborPlugin;

impl Plugin for NeighborPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(BinningPlugin).add_plugins(PairListPlugin);
    }
}

impl Plugin for BinningPlugin {
    fn provides(&self) -> Vec<&str> {
        vec!["particle_bins"]
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
# Sort atoms by spatial bin at the first full rebuild after every N steps (0 = every rebuild)
sort_every = 1000
# Newton's third law optimization (true = half list + reverse comm, false = full list)
newton = true
# Neighbor distance kernel: scalar or simd (falls back to scalar if unsupported)
distance_kernel = "scalar"
# Build the pair list when PairListPlugin/NeighborPlugin is installed (false = bins only)
pairs = true"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<NeighborConfig>(app, "neighbor");

        app.add_resource(Neighbor::new())
            .add_resource(crate::TagIndex::new())
            .add_resource(CurrentState(CommState::FullRebuild))
            .add_setup_system(neighbor_read_input, ScheduleSetupSet::Setup)
            .add_setup_system(
                neighbor_setup.label(NEIGHBOR_SETUP),
                ScheduleSetupSet::PostSetup,
            );

        app.add_update_system(
            decide_rebuild
                .label(DECIDE_REBUILD)
                .before(crate::remove_ghost_atoms)
                .run_if(in_state(CommState::CommunicateOnly)),
            ParticleSimScheduleSet::PostInitialIntegration,
        );
        app.add_update_system(
            sort_atoms_by_bin.label(SORT_ATOMS).before(crate::BORDERS),
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
    pairs_requested: Option<Res<PairListRequested>>,
) {
    neighbor.skin_fraction = config.skin_fraction;
    neighbor.bin_min_size = config.bin_size;
    neighbor.every = config.every;
    neighbor.check = config.check;
    neighbor.sort_every = config.sort_every;
    neighbor.newton = config.newton;
    neighbor.distance_kernel = config.distance_kernel;
    neighbor.build_pairs = pairs_requested.is_some() && config.pairs;
    if comm.rank() == 0 {
        let rebuild_str = if config.every == 0 {
            "displacement".to_string()
        } else if config.check {
            format!("every {} + check", config.every)
        } else {
            format!("every {}", config.every)
        };
        println!(
            "Neighbor: skin_fraction={} bin_size={} rebuild={} newton={} distance_kernel={:?} pairs={}",
            config.skin_fraction,
            config.bin_size,
            rebuild_str,
            if config.newton { "on" } else { "off" },
            config.distance_kernel,
            if neighbor.build_pairs { "on" } else { "off (bins only)" }
        );
    }
}

/// Setup system: computes bin grid, ghost cutoff, and stencil from atom cutoff radii.
///
/// Runs at [`ScheduleSetupSet::PostSetup`] after atoms are created. Determines:
/// - `max_cutoff = 2 * max_skin * skin_fraction` (largest pairwise neighbor distance)
/// - `ghost_cutoff = max(max_cutoff, domain.ghost_cutoff)` (communication distance for ghosts)
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
    // Ghost shell = neighbor-list cutoff (LAMMPS `cutghost = cutforce + skin`).
    // Ghost sets are frozen between rebuilds, and at a rebuild every local atom
    // lies inside its subdomain (exchange enforces it), so any pair the list
    // can hold is within `max_cutoff` of a subdomain face. A pair that starts
    // farther apart cannot reach contact before the displacement check forces
    // a rebuild, so no extra padding is needed. (A previous `2 * (skin - 1) *
    // r_max` pad made the shell ~13% thicker at skin 1.15 for no benefit.)
    // Plugins that need a deeper shell (bonds, clumps) raise
    // `domain.ghost_cutoff` before this system and are honored by the `max`.
    let ghost_cut = max_cutoff.max(domain.ghost_cutoff);
    neighbor.ghost_cutoff = ghost_cut;
    neighbor.cached_max_cutoff = max_cutoff;
    domain.ghost_cutoff = ghost_cut;
    if comm.rank() == 0 {
        println!(
            "Neighbor: ghost_cutoff={:.4} (pair_cutoff={:.4})",
            ghost_cut, max_cutoff
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
    super::grid::compute_bin_grid(&mut neighbor, &domain, comm.size());
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
