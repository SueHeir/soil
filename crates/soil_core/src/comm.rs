//! MPI and single-process communication: ghost atoms, exchange, and force reduction.
//!
//! This module provides:
//! - [`CommBackend`] trait abstracting MPI or single-process communication
//! - [`SingleProcessComm`]: no-op backend for serial runs
//! - [`MpiCommBackend`]: real MPI backend (behind the `mpi_backend` feature)
//! - [`CommunicationPlugin`]: sets up the processor grid, ghost border
//!   communication, and reverse force reduction
//!
//! # Communication phases (per timestep)
//!
//! 1. **Exchange** (MPI only): migrate atoms that left the sub-domain
//! 2. **Borders**: create ghost copies of near-boundary atoms on neighboring ranks
//! 3. **Forward comm**: lightweight position/velocity update for existing ghosts
//! 4. **Reverse comm**: accumulate forces from ghosts back to their owners

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

// Re-export comm abstractions from grass_mpi so downstream users see no change.
pub use grass_mpi::{finalize_mpi, CommBackend, CommResource, SendRecvOp, SingleProcessComm};
#[cfg(feature = "mpi_backend")]
pub use grass_mpi::{get_mpi_world, MpiCommBackend};

use crate::{
    debug_assert_particle_invariant, Accum, Atom, AtomDataRegistry, CommState, Config, Domain,
    Neighbor, ParticleSimScheduleSet, ParticleStore, Real, ScheduleSetupSet,
};

/// Type-level label for [`borders`].
pub struct BordersSystem;
impl SystemLabel for BordersSystem {
    const NAME: &'static str = "borders";
}
/// Stable scheduler key for [`borders`].
pub const BORDERS: SystemKey<BordersSystem> = SystemKey::new();

/// Type-level label for [`reverse_send_force`].
pub struct ReverseSendForceSystem;
impl SystemLabel for ReverseSendForceSystem {
    const NAME: &'static str = "reverse_send_force";
}
/// Stable scheduler key for [`reverse_send_force`].
pub const REVERSE_SEND_FORCE: SystemKey<ReverseSendForceSystem> = SystemKey::new();

/// Type-level label for [`exchange`].
pub struct ExchangeSystem;
impl SystemLabel for ExchangeSystem {
    const NAME: &'static str = "exchange";
}
/// Stable scheduler key for [`exchange`].
pub const EXCHANGE: SystemKey<ExchangeSystem> = SystemKey::new();

// ── CommConfig ──────────────────────────────────────────────────────────────

fn default_one_i32() -> i32 {
    1
}

/// Precision used for the base position/velocity payload in lightweight
/// forward ghost communication.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ForwardCommPrecision {
    /// Preserve the historical f64 forward communication path.
    F64,
    /// Round ghost positions/velocities to f32 before forward communication.
    F32,
}

impl Default for ForwardCommPrecision {
    fn default() -> Self {
        Self::F64
    }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[comm]` — MPI processor grid configuration.
pub struct CommConfig {
    /// Number of MPI ranks in x dimension. `0` = choose automatically
    /// (minimum-surface grid, like LAMMPS `processors * * *`).
    #[serde(default = "default_one_i32")]
    pub processors_x: i32,
    /// Number of MPI ranks in y dimension. `0` = choose automatically
    /// (minimum-surface grid, like LAMMPS `processors * * *`).
    #[serde(default = "default_one_i32")]
    pub processors_y: i32,
    /// Number of MPI ranks in z dimension. `0` = choose automatically
    /// (minimum-surface grid, like LAMMPS `processors * * *`).
    #[serde(default = "default_one_i32")]
    pub processors_z: i32,
    /// Precision for per-step forward ghost position/velocity updates.
    ///
    /// Defaults to `f64`. Setting `"f32"` is an explicit numerics experiment:
    /// reverse force accumulation and extension fields still use f64.
    #[serde(default)]
    pub forward_precision: ForwardCommPrecision,
}

impl Default for CommConfig {
    fn default() -> Self {
        CommConfig {
            processors_x: 1,
            processors_y: 1,
            processors_z: 1,
            forward_precision: ForwardCommPrecision::F64,
        }
    }
}

// ── CommBuffers ──────────────────────────────────────────────────────────────

/// Per-swap send/recv scratch for the overlapped, round-based ghost comm.
/// One pair of buffers per swap so the two swaps of a `(dim, need)` round can be
/// in flight simultaneously (their sends/recvs must not share a buffer). Reused
/// across timesteps so the hot path never reallocates.
#[derive(Default)]
pub struct SwapScratch {
    /// Send buffer for this swap's outgoing payload.
    pub send: Vec<f64>,
    /// Receive buffer for this swap's incoming payload.
    pub recv: Vec<f64>,
    /// f32 base pos/vel send buffer for precision-gated forward comm.
    pub send_f32: Vec<f32>,
    /// f32 base pos/vel receive buffer for precision-gated forward comm.
    pub recv_f32: Vec<f32>,
    /// Existing grass_mpi point-to-point transport is f64-only; this bridge
    /// carries already-rounded f32 values across that backend boundary.
    pub send_bridge: Vec<f64>,
    /// Receive-side bridge for already-rounded f32 base forward payloads.
    pub recv_bridge: Vec<f64>,
    /// f64 extension-field payload used alongside f32 base pos/vel forwarding.
    pub send_extra: Vec<f64>,
    /// f64 extension-field receive buffer used alongside f32 base forwarding.
    pub recv_extra: Vec<f64>,
}

/// Persistent communication buffers, reused across timesteps to avoid re-allocation.
pub struct CommBuffers {
    /// Send buffer for the ghost border build (forward path).
    pub border_send_buff: Vec<f64>,
    /// Per-rank exchange buffers, reused across exchange() calls.
    pub exchange_buffs: Vec<Vec<f64>>,
    /// Persistent receive buffer for the per-step ghost forward/reverse comm.
    /// Reused across timesteps so the hot path never heap-allocates a recv Vec.
    pub recv_buff: Vec<f64>,
    /// Persistent send buffer for reverse_send_force (mirrors border_send_buff,
    /// which is owned by the forward path). Avoids a fresh Vec::new() per step.
    pub reverse_send_buff: Vec<f64>,
    /// Per-swap scratch for the overlapped forward ghost-position comm.
    pub forward_scratch: Vec<SwapScratch>,
    /// Per-swap scratch for the overlapped reverse force comm.
    pub reverse_scratch: Vec<SwapScratch>,
    /// Scratch list of local rows leaving this rank in the current exchange
    /// dimension (strictly descending), reused across exchanges.
    pub exchange_removed: Vec<usize>,
    /// Persistent receive buffer for the count-first ghost rebuild in `borders`
    /// (C7): sized to the exact incoming payload and received into via the
    /// probe-free `_into` path, so a full rebuild never heap-allocates a recv Vec.
    pub border_recv_buff: Vec<f64>,
}

impl Default for CommBuffers {
    fn default() -> Self {
        CommBuffers {
            border_send_buff: Vec::new(),
            exchange_buffs: Vec::new(),
            recv_buff: Vec::new(),
            reverse_send_buff: Vec::new(),
            forward_scratch: Vec::new(),
            reverse_scratch: Vec::new(),
            exchange_removed: Vec::new(),
            border_recv_buff: Vec::new(),
        }
    }
}

// ── SwapData ─────────────────────────────────────────────────────────────────

/// Saved sendlist data for a single border swap, enabling lightweight forward_comm
/// position updates without full ghost rebuild.
pub struct SwapData {
    /// Local atom indices packed (sent) in this swap.
    pub send_indices: Vec<usize>,
    /// First ghost index at which received atoms are stored.
    pub recv_start: usize,
    /// Number of ghosts received in this swap.
    pub recv_count: usize,
    /// Rank this swap sends to.
    pub to_proc: i32,
    /// Rank this swap receives from.
    pub from_proc: i32,
    /// Precomputed periodic position shift (real lattice vector) applied to ghosts.
    pub periodic_offset: [f64; 3],
    /// Lees–Edwards streaming-velocity shift applied to ghosts crossing a sheared face.
    pub vel_offset: [f64; 3],
}

// ── CommTopology ─────────────────────────────────────────────────────────────

/// Swap directions and counts for ghost communication (replaces MpiCommInternal).
/// Works for both single-process and MPI.
pub struct CommTopology {
    /// Neighbor rank direction per swap slot (lower/upper) in each dimension.
    pub swap_directions: [[i32; 3]; 2],
    /// Periodic position shift applied on each swap slot (lower/upper) per dimension.
    pub periodic_swap: [[f64; 3]; 2],
    /// Saved per-swap sendlists, enabling forward_comm without a full ghost rebuild.
    pub swap_data: Vec<SwapData>,
    /// Number of swap layers needed per dimension (0 = not yet computed).
    pub maxneed: [i32; 3],
}

/// Run condition: returns true when Newton's third law optimization is enabled.
fn newton_on(neighbor: Res<Neighbor>) -> bool {
    neighbor.newton
}

// ── Unified CommunicationPlugin ──────────────────────────────────────────────

/// Plugin that sets up communication infrastructure: processor grid, ghost
/// borders, exchange (MPI), and reverse force accumulation.
pub struct CommunicationPlugin;

impl Plugin for CommunicationPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[comm]
# Ranks per axis; 0 = automatic (minimum-surface grid for the run's rank count).
processors_x = 1
processors_y = 1
processors_z = 1
# Forward ghost position/velocity precision: "f64" (default) or "f32"
forward_precision = "f64""#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<CommConfig>(app, "comm");

        // Select backend based on feature flag
        #[cfg(feature = "mpi_backend")]
        {
            let world = get_mpi_world();
            app.add_resource(CommResource(Box::new(MpiCommBackend::new(world))));
        }
        #[cfg(not(feature = "mpi_backend"))]
        {
            app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
        }

        app.add_resource(CommBuffers::default());
        app.add_resource(CommTopology {
            swap_directions: [[-1, -1, -1], [-1, -1, -1]],
            periodic_swap: [[0.0; 3], [0.0; 3]],
            swap_data: Vec::new(),
            maxneed: [0, 0, 0],
        });

        // Validate against the actual backend before legacy scheduler setup.
        // `App::try_start` returns this error and never advances a step.
        app.add_fallible_setup_system("validate communication processor grid", |app: &mut App| {
            let config = app.get_resource_ref::<CommConfig>()
                .ok_or_else(|| AppError::message("CommunicationPlugin did not register CommConfig"))?;
            let comm = app.get_resource_ref::<CommResource>()
                .ok_or_else(|| AppError::message("CommunicationPlugin did not register CommResource"))?;
            let box_len = app
                .get_resource_ref::<crate::DomainConfig>()
                .map(|d| d.box_lengths());
            // `comm_read_input` fills in automatic (0) counts the same way.
            resolve_processor_grid(&config, box_len, comm.size()).map_err(AppError::message)?;
            Ok(())
        }, ScheduleSetupSet::PreSetup);

        app.add_setup_system(
            comm_read_input.run_if(first_stage_only()),
            ScheduleSetupSet::PreSetup,
        )
        .add_setup_system(
            comm_setup.run_if(first_stage_only()),
            ScheduleSetupSet::PostSetup,
        )
        .add_update_system(
            borders
                .label(BORDERS)
                .requires(crate::SORT_ATOMS)
                .run_if(in_state(CommState::FullRebuild)),
            ParticleSimScheduleSet::PreNeighbor,
        )
        .add_update_system(
            forward_comm_borders.run_if(in_state(CommState::CommunicateOnly)),
            ParticleSimScheduleSet::PreNeighbor,
        )
        .add_update_system(
            reverse_send_force
                .label(REVERSE_SEND_FORCE)
                .run_if(newton_on),
            ParticleSimScheduleSet::PostForce,
        );

        #[cfg(feature = "mpi_backend")]
        app.add_update_system(
            exchange
                .label(EXCHANGE)
                .run_if(in_state(CommState::FullRebuild)),
            ParticleSimScheduleSet::Exchange,
        );

        app.add_cleanup(finalize_mpi);
    }
}

// ── Shared systems ───────────────────────────────────────────────────────────

/// Read `[comm]` config and set up the processor grid decomposition.
pub fn comm_read_input(
    mut config: ResMut<CommConfig>,
    domain_config: Option<Res<crate::DomainConfig>>,
    mut comm: ResMut<CommResource>,
) {
    let box_len = domain_config.map(|d| d.box_lengths());
    let decomp = match resolve_processor_grid(&config, box_len, comm.size()) {
        Ok(grid) => grid,
        Err(message) => {
            if comm.rank() == 0 {
                println!("{message}");
            }
            panic!("CommunicationPlugin startup validation was bypassed: {message}");
        }
    };
    config.processors_x = decomp[0];
    config.processors_y = decomp[1];
    config.processors_z = decomp[2];
    if comm.rank() == 0 {
        println!("Comm: processors {} {} {}", decomp[0], decomp[1], decomp[2]);
    }

    let rank = comm.rank();
    let pz = decomp[2];
    let py = decomp[1];
    let position = [rank / (py * pz), (rank / pz) % py, rank % pz];

    comm.set_processor_grid(decomp, position);
}

/// Turn the `[comm]` processor counts into a concrete grid for `nprocs`
/// ranks: zeros are filled in by [`auto_processor_grid`] using the box
/// lengths (a unit cube when no `[domain]` is configured), and the product
/// must equal `nprocs`.
pub fn resolve_processor_grid(
    config: &CommConfig,
    box_len: Option<[f64; 3]>,
    nprocs: i32,
) -> Result<[i32; 3], String> {
    let requested = [config.processors_x, config.processors_y, config.processors_z];
    if requested.iter().any(|&p| p < 0) {
        return Err(format!(
            "[comm] processors {} {} {} must be positive, or 0 for automatic",
            requested[0], requested[1], requested[2]
        ));
    }
    if requested.contains(&0) {
        return auto_processor_grid(nprocs, box_len.unwrap_or([1.0; 3]), requested).ok_or_else(
            || {
                format!(
                    "[comm] processors {} {} {}: no grid with those fixed counts multiplies to {nprocs} ranks",
                    requested[0], requested[1], requested[2]
                )
            },
        );
    }
    let configured = requested[0] * requested[1] * requested[2];
    if configured != nprocs {
        return Err(format!(
            "[comm] processors {} {} {} multiply to {configured}, but the active communicator has {nprocs} ranks",
            requested[0], requested[1], requested[2]
        ));
    }
    Ok(requested)
}

/// Which way `exchange` sends an atom along one axis: `0` keeps it, `-1`
/// sends it to the low neighbor, `1` to the high neighbor. On a periodic axis
/// the coordinate is first wrapped into the box, then an atom outside the slab
/// goes toward whichever face is nearer around the ring. This stays correct
/// for slabs wider than half the box, which load balancing can produce (a
/// min-image displacement from each face does not).
fn exchange_direction(
    coord: f64,
    sub_lo: f64,
    sub_hi: f64,
    box_lo: f64,
    box_size: f64,
    periodic: bool,
) -> i32 {
    if !periodic {
        return if coord < sub_lo {
            -1
        } else if coord >= sub_hi {
            1
        } else {
            0
        };
    }
    let c = box_lo + (coord - box_lo).rem_euclid(box_size);
    if c >= sub_lo && c < sub_hi {
        return 0;
    }
    let below = (sub_lo - c).rem_euclid(box_size);
    let above = (c - sub_hi).rem_euclid(box_size);
    if below <= above {
        -1
    } else {
        1
    }
}

/// Pick a processor grid for `nprocs` ranks that minimizes the total
/// sub-domain surface area (and so the ghost volume), like LAMMPS
/// `processors * * *`. Entries of `fixed` greater than zero are kept; zeros
/// are free. Ties go to the grid with more ranks along x, then y, which keeps
/// the choice deterministic on every rank. Returns `None` when the fixed
/// entries cannot be completed to a grid of exactly `nprocs` ranks.
pub fn auto_processor_grid(nprocs: i32, box_len: [f64; 3], fixed: [i32; 3]) -> Option<[i32; 3]> {
    if nprocs <= 0 {
        return None;
    }
    let mut best: Option<([i32; 3], f64)> = None;
    for px in 1..=nprocs {
        if nprocs % px != 0 || (fixed[0] > 0 && fixed[0] != px) {
            continue;
        }
        let rest = nprocs / px;
        for py in 1..=rest {
            if rest % py != 0 || (fixed[1] > 0 && fixed[1] != py) {
                continue;
            }
            let pz = rest / py;
            if fixed[2] > 0 && fixed[2] != pz {
                continue;
            }
            let (lx, ly, lz) = (
                box_len[0] / px as f64,
                box_len[1] / py as f64,
                box_len[2] / pz as f64,
            );
            let surface = lx * ly + ly * lz + lz * lx;
            // `<=` lets later candidates win ties: larger px, then larger py.
            let better = match best {
                None => true,
                Some((_, s)) => surface <= s * (1.0 + 1e-12),
            };
            if better {
                best = Some(([px, py, pz], surface));
            }
        }
    }
    best.map(|(grid, _)| grid)
}

/// Convert a 3D processor grid position to a linear MPI rank.
fn pos_to_rank(pos: [i32; 3], decomp: [i32; 3]) -> i32 {
    pos[0] * decomp[1] * decomp[2] + pos[1] * decomp[2] + pos[2]
}

/// Compute neighbor ranks and periodic swap offsets for each dimension.
pub fn comm_setup(comm: Res<CommResource>, mut topo: ResMut<CommTopology>, domain: Res<Domain>) {
    let decomp = comm.processor_decomposition();
    let pos = comm.processor_position();
    let periodic = domain.periodic_flags();

    for dim in 0..3 {
        // Forward neighbor (+1 in this dimension)
        if pos[dim] + 1 < decomp[dim] {
            let mut neighbor_pos = pos;
            neighbor_pos[dim] += 1;
            topo.swap_directions[1][dim] = pos_to_rank(neighbor_pos, decomp);
        } else if periodic[dim] {
            let mut neighbor_pos = pos;
            neighbor_pos[dim] = 0;
            topo.swap_directions[1][dim] = pos_to_rank(neighbor_pos, decomp);
            topo.periodic_swap[1][dim] = -1.0;
        }

        // Backward neighbor (-1 in this dimension)
        if pos[dim] >= 1 {
            let mut neighbor_pos = pos;
            neighbor_pos[dim] -= 1;
            topo.swap_directions[0][dim] = pos_to_rank(neighbor_pos, decomp);
        } else if periodic[dim] {
            let mut neighbor_pos = pos;
            neighbor_pos[dim] = decomp[dim] - 1;
            topo.swap_directions[0][dim] = pos_to_rank(neighbor_pos, decomp);
            topo.periodic_swap[0][dim] = 1.0;
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn pack_border_atoms(
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    neighbor: Option<&Neighbor>,
    dim: usize,
    swap: usize,
    change_pos: [f64; 3],
    vel_offset: [f64; 3],
    domain: &Domain,
    send_buff: &mut Vec<f64>,
    packed_indices: &mut Vec<usize>,
    scan_end: usize,
    ghost_cutoff: f64,
) -> i32 {
    let mut count = 0i32;
    packed_indices.clear();
    let use_fixed_cut = ghost_cutoff > 0.0;

    if domain.triclinic {
        // Triclinic face selection in fractional (lamda) coordinates: the box is a
        // unit cube, so the gradient/vorticity faces are axis-aligned planes. The
        // x face (λ0) is sheared, so its lamda thickness is inflated by the tilt to
        // stay conservative (the real-space force filter removes false positives).
        let lo_face = domain.sub_lamda_low[dim];
        let hi_face = domain.sub_lamda_high[dim];
        let inflate = if dim == 0 {
            1.0 + domain.tilt[0].abs() / domain.size[1]
        } else {
            1.0
        };
        let cut_l = if use_fixed_cut {
            (ghost_cutoff / domain.size[dim]) * inflate
        } else {
            0.0
        };
        for i in 0..scan_end {
            let pr = atoms.pos[i];
            let lam_dim = domain.x2lamda([pr[0] as f64, pr[1] as f64, pr[2] as f64])[dim];
            let in_skin = if use_fixed_cut {
                if swap == 0 {
                    lam_dim < lo_face + cut_l
                } else {
                    lam_dim >= hi_face - cut_l
                }
            } else {
                let cut = (atoms.cutoff_radius[i] as f64 * 4.0 / domain.size[dim]) * inflate;
                if swap == 0 {
                    lam_dim < lo_face + cut
                } else {
                    lam_dim >= hi_face - cut
                }
            };
            if in_skin {
                ParticleStore::new(atoms, registry)
                    .pack_border(i, change_pos, vel_offset, send_buff)
                    .expect("border packing found invalid particle columns");
                packed_indices.push(i);
                count += 1;
            }
        }
        return count;
    }

    // Orthogonal fast path. Hoist everything loop-invariant out of the O(N)
    // per-swap scan (C1): whether ghost_cutoff is in use and the face threshold
    // for the fixed-cutoff fast path (the DEM common case).
    let lo_face = domain.sub_domain_low[dim];
    let hi_face = domain.sub_domain_high[dim];
    let fixed_thresh = if swap == 0 {
        lo_face + ghost_cutoff
    } else {
        hi_face - ghost_cutoff
    };

    if use_fixed_cut {
        if let Some(neighbor) = neighbor {
            if neighbor.local_cells_valid
                && neighbor.sorted_local_cell.len() >= atoms.nlocal as usize
                && neighbor.bin_total_cells > 0
                && pack_orthogonal_border_bins(
                    atoms,
                    registry,
                    neighbor,
                    dim,
                    swap,
                    fixed_thresh,
                    change_pos,
                    vel_offset,
                    send_buff,
                    packed_indices,
                    scan_end,
                )
            {
                return packed_indices.len() as i32;
            }
        }
    }

    for i in 0..scan_end {
        let pos_dim = atoms.pos_component(i, dim);
        let in_skin = if use_fixed_cut {
            // Fixed ghost_cutoff: compare against the precomputed face threshold.
            if swap == 0 {
                pos_dim < fixed_thresh
            } else {
                pos_dim >= fixed_thresh
            }
        } else {
            // Polydisperse fallback: per-atom skin * 4.0 (DEM default).
            let cut = atoms.cutoff_radius[i] as f64 * 4.0;
            if swap == 0 {
                pos_dim < lo_face + cut
            } else {
                pos_dim >= hi_face - cut
            }
        };
        if in_skin {
            ParticleStore::new(atoms, registry)
                .pack_border(i, change_pos, vel_offset, send_buff)
                .expect("border packing found invalid particle columns");
            packed_indices.push(i);
            count += 1;
        }
    }
    count
}

#[allow(clippy::too_many_arguments)]
fn pack_orthogonal_border_bins(
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    neighbor: &Neighbor,
    dim: usize,
    swap: usize,
    fixed_thresh: f64,
    change_pos: [f64; 3],
    vel_offset: [f64; 3],
    send_buff: &mut Vec<f64>,
    packed_indices: &mut Vec<usize>,
    scan_end: usize,
) -> bool {
    let nlocal = atoms.nlocal as usize;
    if nlocal == 0 || scan_end < nlocal || neighbor.bin_size[dim] <= 0.0 {
        return false;
    }

    let cells = &neighbor.sorted_local_cell[..nlocal];
    let [nx, ny, nz] = neighbor.bin_count;
    if nx <= 0 || ny <= 0 || nz <= 0 {
        return false;
    }

    let mut pack_cell = |cell: i32| {
        let cell = cell as u32;
        let mut start = lower_bound_u32(cells, cell);
        while start < nlocal && cells[start] == cell {
            let i = start;
            let pos_dim = atoms.pos_component(i, dim);
            let in_skin = if swap == 0 {
                pos_dim < fixed_thresh
            } else {
                pos_dim >= fixed_thresh
            };
            if in_skin {
                ParticleStore::new(atoms, registry)
                    .pack_border(i, change_pos, vel_offset, send_buff)
                    .expect("border packing found invalid particle columns");
                packed_indices.push(i);
            }
            start += 1;
        }
    };

    let [ox, oy, oz] = neighbor.bin_origin;
    let [bsx, bsy, bsz] = neighbor.bin_size;
    let coord_range = |origin: f64, bin_size: f64, n: i32| -> (i32, i32) {
        if swap == 0 {
            (
                0,
                (((fixed_thresh - origin) / bin_size).ceil() as i32 - 1).clamp(0, n - 1),
            )
        } else {
            (
                (((fixed_thresh - origin) / bin_size).floor() as i32).clamp(0, n - 1),
                n - 1,
            )
        }
    };

    match dim {
        0 => {
            let (cx0, cx1) = coord_range(ox, bsx, nx);
            for cx in cx0..=cx1 {
                for cy in 0..ny {
                    for cz in 0..nz {
                        pack_cell(cx * ny * nz + cy * nz + cz);
                    }
                }
            }
        }
        1 => {
            let (cy0, cy1) = coord_range(oy, bsy, ny);
            for cx in 0..nx {
                for cy in cy0..=cy1 {
                    for cz in 0..nz {
                        pack_cell(cx * ny * nz + cy * nz + cz);
                    }
                }
            }
        }
        _ => {
            let (cz0, cz1) = coord_range(oz, bsz, nz);
            for cx in 0..nx {
                for cy in 0..ny {
                    for cz in cz0..=cz1 {
                        pack_cell(cx * ny * nz + cy * nz + cz);
                    }
                }
            }
        }
    }

    // Ghosts received in earlier swaps are not part of sorted_local_cell, but they
    // must still be eligible for multi-hop/corner forwarding.
    for i in nlocal..scan_end {
        let pos_dim = atoms.pos_component(i, dim);
        let in_skin = if swap == 0 {
            pos_dim < fixed_thresh
        } else {
            pos_dim >= fixed_thresh
        };
        if in_skin {
            ParticleStore::new(atoms, registry)
                .pack_border(i, change_pos, vel_offset, send_buff)
                .expect("border packing found invalid particle columns");
            packed_indices.push(i);
        }
    }

    true
}

fn lower_bound_u32(values: &[u32], target: u32) -> usize {
    let mut lo = 0;
    let mut hi = values.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if values[mid] < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn unpack_ghost_atoms(atoms: &mut Atom, registry: &AtomDataRegistry, buf: &[f64], count: usize) {
    let data = &buf[..buf.len() - 1];
    ParticleStore::new(atoms, registry)
        .append_ghost_records(data, count)
        .expect("ghost exchange payload is malformed");
}

/// Like [`unpack_ghost_atoms`] but for a payload that does NOT carry a trailing
/// count element (the count-first C7 path receives the exact payload, with the
/// atom count delivered separately by a tiny pre-message).
fn unpack_ghost_payload(atoms: &mut Atom, registry: &AtomDataRegistry, data: &[f64], count: usize) {
    ParticleStore::new(atoms, registry)
        .append_ghost_records(data, count)
        .expect("ghost exchange payload is malformed");
}

/// Unpack `count` migrated (non-ghost) atoms from a count-first exchange payload
/// (no trailing count element). Mirrors [`unpack_ghost_payload`] for [`exchange`].
// Only called from the `mpi_backend`-gated `exchange` fn; gate the definition to
// match so a no-MPI build doesn't flag it as dead code.
#[cfg_attr(not(feature = "mpi_backend"), allow(dead_code))]
fn unpack_exchanged(atoms: &mut Atom, registry: &AtomDataRegistry, data: &[f64], count: usize) {
    ParticleStore::new(atoms, registry)
        .append_migrant_records(data, count)
        .expect("exchange payload is malformed");
}

// ── Forward comm (lightweight ghost position update) ─────────────────────────

/// Per-atom stride in forward_comm: pos(3) + vel(3) = 6 f64s (base fields only).
const FORWARD_PACK_SIZE: usize = 6;

fn unpack_forward(
    msg: &[f64],
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    recv_start: usize,
    recv_count: usize,
    extra_per_atom: usize,
) {
    // Wire layout per swap: [pos|vel for every atom] then one block per forward
    // extension store (see `pack_forward_into`).
    let base_len = recv_count * FORWARD_PACK_SIZE;
    for (k, rec) in msg[..base_len].chunks_exact(FORWARD_PACK_SIZE).enumerate() {
        atoms.pos[recv_start + k] = [rec[0] as Real, rec[1] as Real, rec[2] as Real];
        atoms.vel[recv_start + k] = [rec[3] as Real, rec[4] as Real, rec[5] as Real];
    }
    if extra_per_atom > 0 {
        registry.unpack_forward_rows(recv_start, recv_count, &msg[base_len..]);
    }
}

fn unpack_forward_f32(
    base_msg: &[f32],
    extra_msg: &[f64],
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    recv_start: usize,
    recv_count: usize,
    extra_per_atom: usize,
) {
    for (k, rec) in base_msg[..recv_count * FORWARD_PACK_SIZE]
        .chunks_exact(FORWARD_PACK_SIZE)
        .enumerate()
    {
        atoms.pos[recv_start + k] = [rec[0] as Real, rec[1] as Real, rec[2] as Real];
        atoms.vel[recv_start + k] = [rec[3] as Real, rec[4] as Real, rec[5] as Real];
    }
    if extra_per_atom > 0 {
        registry.unpack_forward_rows(recv_start, recv_count, extra_msg);
    }
}

/// Append one swap's outgoing ghost pos/vel (+ registry forward fields) to `buf`.
///
/// The periodic-image offset is the FIXED value from the last neighbor build
/// (`swap.periodic_offset`); it must NOT be recomputed from the atom's current
/// position — between rebuilds an atom can drift across a periodic boundary, and
/// flipping the offset jerks the ghost a full box length, dropping a real contact
/// and reintroducing it with deep overlap at the next rebuild (a spurious force
/// that injects energy — a real Haff-cooling energy-conservation bug).
fn pack_forward_into(
    swap: &SwapData,
    buf: &mut Vec<f64>,
    atoms: &Atom,
    registry: &AtomDataRegistry,
) {
    let offset = swap.periodic_offset;
    let voff = swap.vel_offset;
    // Base block first (pos|vel per atom), then each forward extension store
    // packs all of this swap's atoms in one batched call — one RefCell borrow
    // and one dynamic dispatch per store instead of per atom.
    buf.reserve(swap.send_indices.len() * FORWARD_PACK_SIZE);
    for &idx in &swap.send_indices {
        let p = atoms.pos[idx];
        let v = atoms.vel[idx];
        buf.extend_from_slice(&[
            p[0] as f64 + offset[0],
            p[1] as f64 + offset[1],
            p[2] as f64 + offset[2],
            v[0] as f64 + voff[0],
            v[1] as f64 + voff[1],
            v[2] as f64 + voff[2],
        ]);
    }
    registry.pack_forward_rows(&swap.send_indices, buf);
}

fn pack_forward_base_f32_into(swap: &SwapData, buf: &mut Vec<f32>, atoms: &Atom) {
    let offset = swap.periodic_offset;
    let voff = swap.vel_offset;
    for &idx in &swap.send_indices {
        buf.push((atoms.pos[idx][0] as f64 + offset[0]) as f32);
        buf.push((atoms.pos[idx][1] as f64 + offset[1]) as f32);
        buf.push((atoms.pos[idx][2] as f64 + offset[2]) as f32);
        buf.push((atoms.vel[idx][0] as f64 + voff[0]) as f32);
        buf.push((atoms.vel[idx][1] as f64 + voff[1]) as f32);
        buf.push((atoms.vel[idx][2] as f64 + voff[2]) as f32);
    }
}

fn pack_forward_extra_into(swap: &SwapData, buf: &mut Vec<f64>, registry: &AtomDataRegistry) {
    registry.pack_forward_rows(&swap.send_indices, buf);
}

/// Pack one swap's outgoing ghost data into `scratch.send`. If the swap is a
/// self-send (single-process / periodic-to-self), unpack it locally and return
/// `None`. Otherwise size `scratch.recv` and return a [`SendRecvOp`] for the
/// batch comm to complete. The returned op borrows `scratch`, so the caller must
/// finish the batch (and any unpack) before reusing it.
fn forward_prepare_swap<'s>(
    swap: &SwapData,
    scratch: &'s mut SwapScratch,
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    rank: i32,
    stride: usize,
    extra: usize,
) -> Option<SendRecvOp<'s>> {
    // Pack positions and velocities (6 f64s per atom) + registry forward fields
    let buf = &mut scratch.send;
    buf.clear();
    buf.reserve(swap.send_indices.len() * stride);
    pack_forward_into(swap, buf, atoms, registry);

    if swap.to_proc == rank {
        // Self-send: copy directly into ghost data, no MPI.
        unpack_forward(
            &scratch.send,
            atoms,
            registry,
            swap.recv_start,
            swap.recv_count,
            extra,
        );
        return None;
    }

    // MPI op. `-1` on either side disables that half (non-periodic boundary).
    let source = swap.from_proc;
    let recv_len = if source != -1 {
        swap.recv_count * stride
    } else {
        0
    };
    scratch.recv.resize(recv_len, 0.0);
    Some(SendRecvOp {
        dest: swap.to_proc,
        send_buf: &scratch.send,
        source,
        recv_buf: &mut scratch.recv,
    })
}

fn forward_comm_f32(
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    topo: &CommTopology,
    comm: &dyn CommBackend,
    pool: &mut Vec<SwapScratch>,
) {
    let rank = comm.rank();
    let extra = registry.forward_comm_size();
    let swaps = &topo.swap_data;
    if pool.len() < swaps.len() {
        pool.resize_with(swaps.len(), SwapScratch::default);
    }

    let mut r = 0;
    while r < swaps.len() {
        let end = (r + 2).min(swaps.len());

        if end == r + 2 && round_aggregatable(&swaps[r], &swaps[r + 1], rank) {
            let lo = &swaps[r];
            let hi = &swaps[r + 1];
            let scratch = &mut pool[r];

            scratch.send_f32.clear();
            scratch
                .send_f32
                .reserve((lo.send_indices.len() + hi.send_indices.len()) * FORWARD_PACK_SIZE);
            pack_forward_base_f32_into(lo, &mut scratch.send_f32, atoms);
            pack_forward_base_f32_into(hi, &mut scratch.send_f32, atoms);

            scratch.send_bridge.clear();
            scratch
                .send_bridge
                .extend(scratch.send_f32.iter().map(|&x| x as f64));
            scratch
                .recv_bridge
                .resize((lo.recv_count + hi.recv_count) * FORWARD_PACK_SIZE, 0.0);
            comm.sendrecv_f64_into(
                lo.to_proc,
                &scratch.send_bridge,
                lo.from_proc,
                &mut scratch.recv_bridge,
            );

            scratch.recv_f32.clear();
            scratch
                .recv_f32
                .extend(scratch.recv_bridge.iter().map(|&x| x as f32));

            if extra > 0 {
                scratch.send_extra.clear();
                scratch
                    .send_extra
                    .reserve((lo.send_indices.len() + hi.send_indices.len()) * extra);
                pack_forward_extra_into(lo, &mut scratch.send_extra, registry);
                pack_forward_extra_into(hi, &mut scratch.send_extra, registry);
                scratch
                    .recv_extra
                    .resize((lo.recv_count + hi.recv_count) * extra, 0.0);
                comm.sendrecv_f64_into(
                    lo.to_proc,
                    &scratch.send_extra,
                    lo.from_proc,
                    &mut scratch.recv_extra,
                );
            } else {
                scratch.recv_extra.clear();
            }

            let lo_base = lo.recv_count * FORWARD_PACK_SIZE;
            let lo_extra = lo.recv_count * extra;
            unpack_forward_f32(
                &scratch.recv_f32[..lo_base],
                &scratch.recv_extra[..lo_extra],
                atoms,
                registry,
                lo.recv_start,
                lo.recv_count,
                extra,
            );
            unpack_forward_f32(
                &scratch.recv_f32[lo_base..],
                &scratch.recv_extra[lo_extra..],
                atoms,
                registry,
                hi.recv_start,
                hi.recv_count,
                extra,
            );

            r = end;
            continue;
        }

        for (idx, swap) in swaps[r..end].iter().enumerate() {
            let idx = r + idx;
            let scratch = &mut pool[idx];
            scratch.send_f32.clear();
            scratch
                .send_f32
                .reserve(swap.send_indices.len() * FORWARD_PACK_SIZE);
            pack_forward_base_f32_into(swap, &mut scratch.send_f32, atoms);

            scratch.send_extra.clear();
            if extra > 0 {
                scratch.send_extra.reserve(swap.send_indices.len() * extra);
                pack_forward_extra_into(swap, &mut scratch.send_extra, registry);
            }

            if swap.to_proc == rank {
                unpack_forward_f32(
                    &scratch.send_f32,
                    &scratch.send_extra,
                    atoms,
                    registry,
                    swap.recv_start,
                    swap.recv_count,
                    extra,
                );
                continue;
            }

            let recv_base_len = if swap.from_proc != -1 {
                swap.recv_count * FORWARD_PACK_SIZE
            } else {
                0
            };
            scratch.send_bridge.clear();
            scratch
                .send_bridge
                .extend(scratch.send_f32.iter().map(|&x| x as f64));
            scratch.recv_bridge.resize(recv_base_len, 0.0);
            comm.sendrecv_batch_f64_into(&mut [SendRecvOp {
                dest: swap.to_proc,
                send_buf: &scratch.send_bridge,
                source: swap.from_proc,
                recv_buf: &mut scratch.recv_bridge,
            }]);

            if extra > 0 {
                let recv_extra_len = if swap.from_proc != -1 {
                    swap.recv_count * extra
                } else {
                    0
                };
                scratch.recv_extra.resize(recv_extra_len, 0.0);
                comm.sendrecv_batch_f64_into(&mut [SendRecvOp {
                    dest: swap.to_proc,
                    send_buf: &scratch.send_extra,
                    source: swap.from_proc,
                    recv_buf: &mut scratch.recv_extra,
                }]);
            } else {
                scratch.recv_extra.clear();
            }

            if swap.from_proc != -1 {
                scratch.recv_f32.clear();
                scratch
                    .recv_f32
                    .extend(scratch.recv_bridge.iter().map(|&x| x as f32));
                unpack_forward_f32(
                    &scratch.recv_f32,
                    &scratch.recv_extra,
                    atoms,
                    registry,
                    swap.recv_start,
                    swap.recv_count,
                    extra,
                );
            }
        }
        r = end;
    }
}

/// C5: can a round's two swaps be aggregated into a single sendrecv? True when
/// both involve the SAME neighbour rank in both directions — the `decomp == 2`
/// periodic case, where the `+` and `−` swaps in a dimension both wrap to the one
/// neighbour. Requires all four endpoints to be a real, non-self rank, which rules
/// out self-sends (forward-self `to_proc==rank`, reverse-self `from_proc==rank`)
/// and non-periodic edges (`-1`). Used by both forward and reverse comm.
fn round_aggregatable(lo: &SwapData, hi: &SwapData, rank: i32) -> bool {
    lo.to_proc != rank
        && lo.from_proc != rank
        && lo.to_proc != -1
        && lo.from_proc != -1
        && lo.to_proc == hi.to_proc
        && lo.from_proc == hi.from_proc
}

/// Forward ghost position/velocity update, overlapping the two swaps of each
/// `(dim, need)` round via a single batched non-blocking sendrecv.
///
/// Swaps are processed in rounds of two (the lo/hi pair recorded consecutively
/// by `borders`). The two swaps in a round are independent — disjoint receive
/// regions, send-lists fixed at the last rebuild — so their messages are posted
/// together and their latencies overlap. Rounds stay ordered: a later round can
/// forward corner ghosts that an earlier round's receive just filled (multi-hop),
/// so each round's batch completes before the next begins.
fn forward_comm(
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    topo: &CommTopology,
    comm: &dyn CommBackend,
    pool: &mut Vec<SwapScratch>,
) {
    let rank = comm.rank();
    let extra = registry.forward_comm_size();
    let stride = FORWARD_PACK_SIZE + extra;
    let swaps = &topo.swap_data;
    if pool.len() < swaps.len() {
        pool.resize_with(swaps.len(), SwapScratch::default);
    }

    let mut r = 0;
    while r < swaps.len() {
        let end = (r + 2).min(swaps.len());

        // C5: aggregate the round's two same-neighbour swaps into one sendrecv.
        // Forward stride is fixed, so the combined receive — [lo | hi], in the
        // sender's swap order, which both ranks share — splits at lo.recv_count.
        if end == r + 2 && round_aggregatable(&swaps[r], &swaps[r + 1], rank) {
            let lo = &swaps[r];
            let hi = &swaps[r + 1];
            let lo_rc = lo.recv_count;
            let hi_rc = hi.recv_count;
            {
                let scratch = &mut pool[r];
                scratch.send.clear();
                scratch
                    .send
                    .reserve((lo.send_indices.len() + hi.send_indices.len()) * stride);
                pack_forward_into(lo, &mut scratch.send, atoms, registry);
                pack_forward_into(hi, &mut scratch.send, atoms, registry);
                scratch.recv.resize((lo_rc + hi_rc) * stride, 0.0);
                comm.sendrecv_f64_into(lo.to_proc, &scratch.send, lo.from_proc, &mut scratch.recv);
            }
            let recv = &pool[r].recv;
            unpack_forward(
                &recv[..lo_rc * stride],
                atoms,
                registry,
                lo.recv_start,
                lo_rc,
                extra,
            );
            unpack_forward(
                &recv[lo_rc * stride..],
                atoms,
                registry,
                hi.recv_start,
                hi_rc,
                extra,
            );
            r = end;
            continue;
        }

        // `jobs` records which swaps in this round did an MPI receive that still
        // needs unpacking after the batch completes (self-sends already unpacked).
        let mut jobs: [usize; 2] = [0, 0];
        let mut njobs = 0;
        {
            let round = &mut pool[r..end];
            let (first, rest) = round.split_first_mut().unwrap();
            let mut ops: Vec<SendRecvOp> = Vec::with_capacity(2);
            if let Some(op) =
                forward_prepare_swap(&swaps[r], first, atoms, registry, rank, stride, extra)
            {
                ops.push(op);
                jobs[njobs] = r;
                njobs += 1;
            }
            if end > r + 1 {
                if let Some(op) = forward_prepare_swap(
                    &swaps[r + 1],
                    &mut rest[0],
                    atoms,
                    registry,
                    rank,
                    stride,
                    extra,
                ) {
                    ops.push(op);
                    jobs[njobs] = r + 1;
                    njobs += 1;
                }
            }
            if !ops.is_empty() {
                comm.sendrecv_batch_f64_into(&mut ops);
            }
        } // `ops` dropped: pool borrows released, recv data now resident in pool.

        // Unpack the receives (send-only swaps have from_proc == -1, nothing to do).
        for &j in &jobs[..njobs] {
            let swap = &swaps[j];
            if swap.from_proc != -1 {
                unpack_forward(
                    &pool[j].recv,
                    atoms,
                    registry,
                    swap.recv_start,
                    swap.recv_count,
                    extra,
                );
            }
        }
        r = end;
    }
}

// ── Unified borders ──────────────────────────────────────────────────────────

/// Lightweight ghost position update: forward-comm only (no full rebuild).
///
/// Gated by `run_if(in_state(CommState::CommunicateOnly))`. Ghost periodic
/// offsets are the fixed values recorded at the last rebuild (see `forward_comm`),
/// so positions stay correct even when atoms cross PBC boundaries between rebuilds.
pub fn forward_comm_borders(
    comm: Res<CommResource>,
    config: Res<CommConfig>,
    topo: Res<CommTopology>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    mut buffers: ResMut<CommBuffers>,
) {
    debug_assert_particle_invariant(&atoms, &registry, "forward ghost communication (entry)");
    let mut pool = std::mem::take(&mut buffers.forward_scratch);
    match config.forward_precision {
        ForwardCommPrecision::F64 => forward_comm(&mut atoms, &registry, &topo, &**comm, &mut pool),
        ForwardCommPrecision::F32 => {
            forward_comm_f32(&mut atoms, &registry, &topo, &**comm, &mut pool)
        }
    }
    buffers.forward_scratch = pool;
    debug_assert_particle_invariant(&atoms, &registry, "forward ghost communication (exit)");
}

/// Forward ghost comm that runs `overlap(atoms)` *while the first round's swaps are
/// in flight* — roadmap step 4's interior/boundary overlap. The caller passes a
/// closure computing interior forces (pairs that need no ghosts); it executes
/// during the MPI latency of the first ghost exchange, after which the remaining
/// rounds complete and the caller runs the boundary pass on the freshly-landed
/// ghosts. The result is bit-identical to `forward_comm` + a single-pass force
/// (the interior/boundary split is exact — see `dirt_granular`'s
/// `interior_boundary_split_matches_single_pass`).
///
/// Unlike [`forward_comm`] this never takes the aggregated single-`sendrecv`
/// shortcut: every round uses the overlap-capable batch primitive so the closure
/// truly overlaps. If there are no MPI swaps (single process, or all self-sends)
/// the closure still runs — interior == all when there are no ghosts.
pub fn forward_comm_overlap(
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    topo: &CommTopology,
    comm: &dyn CommBackend,
    pool: &mut Vec<SwapScratch>,
    overlap: &mut dyn FnMut(&mut Atom),
) {
    let rank = comm.rank();
    let extra = registry.forward_comm_size();
    let stride = FORWARD_PACK_SIZE + extra;
    let swaps = &topo.swap_data;
    if pool.len() < swaps.len() {
        pool.resize_with(swaps.len(), SwapScratch::default);
    }

    let mut overlap_done = false;
    let mut r = 0;
    while r < swaps.len() {
        let end = (r + 2).min(swaps.len());
        let mut jobs: [usize; 2] = [0, 0];
        let mut njobs = 0;
        {
            let round = &mut pool[r..end];
            let (first, rest) = round.split_first_mut().unwrap();
            let mut ops: Vec<SendRecvOp> = Vec::with_capacity(2);
            if let Some(op) =
                forward_prepare_swap(&swaps[r], first, atoms, registry, rank, stride, extra)
            {
                ops.push(op);
                jobs[njobs] = r;
                njobs += 1;
            }
            if end > r + 1 {
                if let Some(op) = forward_prepare_swap(
                    &swaps[r + 1],
                    &mut rest[0],
                    atoms,
                    registry,
                    rank,
                    stride,
                    extra,
                ) {
                    ops.push(op);
                    jobs[njobs] = r + 1;
                    njobs += 1;
                }
            }
            if !ops.is_empty() {
                if overlap_done {
                    comm.sendrecv_batch_f64_into(&mut ops);
                } else {
                    // `ops` borrow `pool` scratch; `overlap` reborrows `atoms`
                    // (the pack already copied positions into the send buffers, so
                    // `atoms` is free here). Disjoint — the closure mutates forces
                    // while the swaps fly.
                    let mut wrapped = || overlap(atoms);
                    comm.sendrecv_batch_overlap_f64_into(&mut ops, &mut wrapped);
                    overlap_done = true;
                }
            }
        }
        for &j in &jobs[..njobs] {
            let swap = &swaps[j];
            if swap.from_proc != -1 {
                unpack_forward(
                    &pool[j].recv,
                    atoms,
                    registry,
                    swap.recv_start,
                    swap.recv_count,
                    extra,
                );
            }
        }
        r = end;
    }

    // Single-process or all-self-send: no batch ran, so the interior work — which
    // is the entire force when there are no ghosts — still has to execute.
    if !overlap_done {
        overlap(atoms);
    }
}

/// Build ghost atoms by scanning near-boundary locals and sending them to neighbors.
///
/// Gated by `run_if(in_state(CommState::FullRebuild))`.
pub fn borders(
    comm: Res<CommResource>,
    mut topo: ResMut<CommTopology>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    neighbor: Option<Res<Neighbor>>,
    registry: Res<AtomDataRegistry>,
    mut buffers: ResMut<CommBuffers>,
) {
    debug_assert_particle_invariant(&atoms, &registry, "ghost rebuild (entry)");
    let mut send_buff = std::mem::take(&mut buffers.border_send_buff);
    let mut recv_buff = std::mem::take(&mut buffers.border_recv_buff);

    // Full ghost rebuild: remove old ghosts and derive the ownership counters
    // through the facade before appending a new ghost suffix.
    ParticleStore::new(&mut atoms, &registry)
        .begin_ghost_rebuild()
        .expect("borders found misaligned particle columns");

    // `natoms` (global count) is reporting-only — nothing in the neighbor/force/
    // exchange/integrator path reads it. It's refreshed by `print_thermo` (which
    // already runs a collective on its cadence), so we no longer pay an all_reduce
    // collective here on every ghost rebuild.
    // Compute maxneed on first full rebuild (ghost_cutoff is set by neighbor_setup)
    if topo.maxneed == [0, 0, 0] {
        let decomp = comm.processor_decomposition();
        for (dim, (&dec, &sub_len)) in decomp.iter().zip(domain.sub_length.iter()).enumerate() {
            if dec == 1 {
                topo.maxneed[dim] = 1;
            } else {
                topo.maxneed[dim] = (domain.ghost_cutoff / sub_len).ceil().max(1.0) as i32;
            }
        }
        if comm.rank() == 0 {
            for dim in 0..3 {
                if topo.maxneed[dim] > 1 {
                    let dim_name = ["x", "y", "z"][dim];
                    println!(
                        "Comm: multi-hop ghost communication in dim {}: need={}",
                        dim_name, topo.maxneed[dim]
                    );
                }
            }
        }
    }

    // Refill sendlists in place across rebuilds: walk existing SwapData entries
    // (reusing their `send_indices` allocations) instead of clear() + push, which
    // would free and re-grow ~6 Vecs every rebuild. The number of swaps is stable
    // after the first rebuild (maxneed is constant), so this is allocation-free.
    let mut swap_idx = 0usize;

    let rank = comm.rank();
    let mut scan_end = atoms.nlocal as usize;

    for dim in 0..3 {
        for _need in 0..topo.maxneed[dim] {
            let dim_scan_end = scan_end;
            for swap in 0..2 {
                let to_proc = topo.swap_directions[swap][dim];
                let from_proc = topo.swap_directions[(swap + 1) % 2][dim];
                let periodic_offset = topo.periodic_swap[swap][dim];
                send_buff.clear();

                // Periodic image shift = ±(the box edge vector for this dimension).
                // For an orthogonal box this is axis-aligned (= ±L along `dim`); for a
                // triclinic box the y/z edges carry the tilt (e.g. b = (xy, Ly, 0)), so
                // a y-swap ghost is shifted in x by the tilt automatically.
                let po = periodic_offset;
                let pbc_shift = match dim {
                    0 => [po * domain.size[0], 0.0, 0.0],
                    1 => [po * domain.tilt[0], po * domain.size[1], 0.0],
                    _ => [
                        po * domain.tilt[1],
                        po * domain.tilt[2],
                        po * domain.size[2],
                    ],
                };
                // Lees–Edwards streaming-velocity shift on ghosts that wrap across the
                // moving (sheared) y face. Zero for an orthogonal box (boundary_vel = 0).
                let vel_shift = if dim == 1 {
                    [
                        po * domain.boundary_vel[0],
                        po * domain.boundary_vel[1],
                        po * domain.boundary_vel[2],
                    ]
                } else {
                    [0.0; 3]
                };

                // Get-or-create this swap's persistent entry, and borrow its
                // send_indices buffer (capacity retained across rebuilds).
                if swap_idx == topo.swap_data.len() {
                    topo.swap_data.push(SwapData {
                        send_indices: Vec::new(),
                        recv_start: 0,
                        recv_count: 0,
                        to_proc: -1,
                        from_proc: -1,
                        periodic_offset: [0.0; 3],
                        vel_offset: [0.0; 3],
                    });
                }
                let mut packed_indices = std::mem::take(&mut topo.swap_data[swap_idx].send_indices);

                let count = pack_border_atoms(
                    &mut atoms,
                    &registry,
                    neighbor.as_deref(),
                    dim,
                    swap,
                    pbc_shift,
                    vel_shift,
                    &domain,
                    &mut send_buff,
                    &mut packed_indices,
                    dim_scan_end,
                    domain.ghost_cutoff,
                );
                let swap_recv_start;
                let swap_recv_count;

                if to_proc == rank {
                    // Self-send (single-process periodic): unpack locally
                    send_buff.push(count as f64);
                    swap_recv_start = atoms.len();
                    swap_recv_count = count as usize;
                    unpack_ghost_atoms(&mut atoms, &registry, &send_buff, count as usize);
                } else if to_proc != -1 && from_proc != -1 {
                    // C7 count-first: exchange (atom_count, payload_len) in one tiny
                    // fixed-size message, then receive the exact payload into a
                    // persistent buffer — no MPI_Probe, no per-call heap alloc. The
                    // payload length is sent explicitly (not derived from a per-atom
                    // stride) because registry fields (e.g. tangential contact history)
                    // are variable-length per atom. Both ranks always take this path,
                    // so the paired sendrecv is symmetric.
                    let send_meta = [count as f64, send_buff.len() as f64];
                    let mut recv_meta = [0.0f64; 2];
                    comm.sendrecv_f64_into(to_proc, &send_meta, from_proc, &mut recv_meta);
                    let recv_count = recv_meta[0] as usize;
                    let recv_len = recv_meta[1] as usize;
                    recv_buff.resize(recv_len, 0.0);
                    comm.sendrecv_f64_into(to_proc, &send_buff, from_proc, &mut recv_buff);
                    swap_recv_start = atoms.len();
                    swap_recv_count = recv_count;
                    unpack_ghost_payload(&mut atoms, &registry, &recv_buff, recv_count);
                } else if to_proc != -1 {
                    // Send only (non-periodic boundary, no neighbor to receive from).
                    // Same 2-message (meta, payload) protocol as the symmetric path so
                    // a count-first peer on the other side matches; otherwise a
                    // symmetric rank would deadlock against this edge rank.
                    let send_meta = [count as f64, send_buff.len() as f64];
                    comm.send_f64(to_proc, &send_meta);
                    comm.send_f64(to_proc, &send_buff);
                    swap_recv_start = atoms.len();
                    swap_recv_count = 0;
                } else if from_proc != -1 {
                    // Receive only — matching 2-message protocol (meta then payload),
                    // probe-free into persistent buffers. A recv-only batch op
                    // (dest = -1) posts a single Irecv into a caller-sized buffer,
                    // so there is no MPI_Probe and no per-call heap allocation: the
                    // recv-side analog of the symmetric sendrecv_f64_into path above.
                    let mut recv_meta = [0.0f64; 2];
                    comm.sendrecv_batch_f64_into(&mut [SendRecvOp {
                        dest: -1,
                        send_buf: &[],
                        source: from_proc,
                        recv_buf: &mut recv_meta,
                    }]);
                    let recv_count = recv_meta[0] as usize;
                    recv_buff.resize(recv_meta[1] as usize, 0.0);
                    comm.sendrecv_batch_f64_into(&mut [SendRecvOp {
                        dest: -1,
                        send_buf: &[],
                        source: from_proc,
                        recv_buf: &mut recv_buff,
                    }]);
                    swap_recv_start = atoms.len();
                    swap_recv_count = recv_count;
                    unpack_ghost_payload(&mut atoms, &registry, &recv_buff, recv_count);
                } else {
                    swap_recv_start = atoms.len();
                    swap_recv_count = 0;
                }

                let entry = &mut topo.swap_data[swap_idx];
                entry.send_indices = packed_indices; // give the buffer back (capacity kept)
                entry.recv_start = swap_recv_start;
                entry.recv_count = swap_recv_count;
                entry.to_proc = to_proc;
                entry.from_proc = from_proc;
                entry.periodic_offset = pbc_shift;
                entry.vel_offset = vel_shift;
                swap_idx += 1;
            }
            scan_end = atoms.nlocal as usize + atoms.nghost as usize;
        }
    }
    // Drop any stale entries from a prior rebuild that produced more swaps
    // (only possible if maxneed shrank, which it cannot after the first rebuild).
    topo.swap_data.truncate(swap_idx);

    buffers.border_send_buff = send_buff;
    buffers.border_recv_buff = recv_buff;
    debug_assert_particle_invariant(&atoms, &registry, "ghost rebuild (exit)");
}

// ── Unified reverse send force ────────────────────────────────────────────────

/// Accumulate ghost forces from owner atoms received back into local force, for
/// one swap's MPI receive. The k-th returned force belongs to `swap.send_indices[k]`
/// (the same stable ghost→origin mapping `forward_comm` relies on).
fn reverse_accumulate(
    swap: &SwapData,
    recv: &[f64],
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    per_atom: usize,
) {
    // Wire layout per swap: [force for every atom] then one block per reverse
    // extension store (see `pack_reverse_into`).
    let n = swap.send_indices.len();
    for (rec, &origin) in recv[..3 * n].chunks_exact(3).zip(&swap.send_indices) {
        let f = &mut atoms.force[origin];
        f[0] += rec[0] as Accum;
        f[1] += rec[1] as Accum;
        f[2] += rec[2] as Accum;
    }
    if per_atom > 3 {
        registry.unpack_reverse_rows(&swap.send_indices, &recv[3 * n..]);
    }
}

/// Pack one swap's ghost forces into `scratch.send`. If the swap is a self-send,
/// accumulate locally and return `None`. Otherwise size `scratch.recv` and return
/// a [`SendRecvOp`] (note: send/recv directions are *reversed* vs forward comm —
/// forces flow back from `from_proc` toward `to_proc`).
/// Append one swap's ghost forces (+ registry reverse fields) to `buf`, in
/// ghost-index order over the swap's received-ghost region.
fn pack_reverse_into(
    swap: &SwapData,
    buf: &mut Vec<f64>,
    atoms: &Atom,
    registry: &AtomDataRegistry,
) {
    let range = swap.recv_start..(swap.recv_start + swap.recv_count);
    buf.reserve(3 * swap.recv_count);
    for i in range {
        debug_assert!(
            atoms.is_ghost[i],
            "reverse_send_force: atom {} is not ghost",
            i
        );
        let f = atoms.force[i];
        buf.extend_from_slice(&[f[0] as f64, f[1] as f64, f[2] as f64]);
    }
    registry.pack_reverse_rows(swap.recv_start, swap.recv_count, buf);
}

fn reverse_prepare_swap<'s>(
    swap: &SwapData,
    scratch: &'s mut SwapScratch,
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    rank: i32,
    per_atom: usize,
) -> Option<SendRecvOp<'s>> {
    // Pack force (+ registry reverse fields) per ghost atom, in ghost-index order.
    let buf = &mut scratch.send;
    buf.clear();
    buf.reserve(swap.recv_count * per_atom);
    pack_reverse_into(swap, buf, atoms, registry);

    if swap.from_proc == rank {
        // Self-send: apply forces locally. recv_count == send_indices.len() here.
        let send = std::mem::take(&mut scratch.send);
        reverse_accumulate(swap, &send, atoms, registry, per_atom);
        scratch.send = send;
        return None;
    }

    // MPI: we receive forces for exactly the atoms we sent to `to_proc`
    // (swap.send_indices), in order. `-1` disables the corresponding half.
    let source = swap.to_proc;
    let recv_len = if source != -1 {
        swap.send_indices.len() * per_atom
    } else {
        0
    };
    scratch.recv.resize(recv_len, 0.0);
    Some(SendRecvOp {
        dest: swap.from_proc,
        send_buf: &scratch.send,
        source,
        recv_buf: &mut scratch.recv,
    })
}

/// Accumulate ghost forces back onto their owner atoms.
///
/// Mirror of `forward_comm`: swaps are processed in `(dim, need)` rounds of two,
/// overlapping the two swaps' messages via one batched non-blocking sendrecv.
/// Rounds are visited in **reverse** order (last dimension first) so an
/// intermediate rank accumulates a multi-hop ghost's force before forwarding it
/// further back. The two swaps within a round are independent.
pub fn reverse_send_force(
    comm: Res<CommResource>,
    topo: Res<CommTopology>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    mut buffers: ResMut<CommBuffers>,
) {
    let mut pool = std::mem::take(&mut buffers.reverse_scratch);
    reverse_send_force_impl(&**comm, &topo, &mut atoms, &registry, &mut pool);
    buffers.reverse_scratch = pool;
}

fn reverse_send_force_impl(
    comm: &dyn CommBackend,
    topo: &CommTopology,
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    pool: &mut Vec<SwapScratch>,
) {
    let rank = comm.rank();

    // Per-atom stride: force×3 + registry reverse fields. The owner of each
    // returned force is *not* transmitted: ghosts were built from `swap.send_indices`
    // in order (see `borders`), and that mapping is stable across the
    // CommunicateOnly steps between rebuilds (the same invariant `forward_comm`
    // relies on every step). So the k-th returned force belongs to
    // `swap.send_indices[k]` — verified empirically to always equal the formerly
    // transmitted `origin_index`. Dropping `tag`+`origin_index` saves 2 f64/atom.
    let per_atom = 3 + registry.reverse_comm_size();

    let swaps = &topo.swap_data;
    if pool.len() < swaps.len() {
        pool.resize_with(swaps.len(), SwapScratch::default);
    }

    // Iterate rounds from the last back to the first (mirrors borders() in reverse).
    let mut end = swaps.len();
    while end > 0 {
        let start = end.saturating_sub(2);

        // C5: aggregate the round's two same-neighbour swaps into one sendrecv
        // (mirror of forward_comm). Send [lo | hi] ghost forces to from_proc,
        // receive [lo | hi] owner forces from to_proc, split at lo's owned count.
        if start + 2 == end && round_aggregatable(&swaps[start], &swaps[start + 1], rank) {
            let lo = &swaps[start];
            let hi = &swaps[start + 1];
            {
                let scratch = &mut pool[start];
                scratch.send.clear();
                scratch
                    .send
                    .reserve((lo.recv_count + hi.recv_count) * per_atom);
                pack_reverse_into(lo, &mut scratch.send, atoms, registry);
                pack_reverse_into(hi, &mut scratch.send, atoms, registry);
                let recv_len = (lo.send_indices.len() + hi.send_indices.len()) * per_atom;
                scratch.recv.resize(recv_len, 0.0);
                comm.sendrecv_f64_into(lo.from_proc, &scratch.send, lo.to_proc, &mut scratch.recv);
            }
            let lo_split = lo.send_indices.len() * per_atom;
            reverse_accumulate(lo, &pool[start].recv[..lo_split], atoms, registry, per_atom);
            reverse_accumulate(hi, &pool[start].recv[lo_split..], atoms, registry, per_atom);
            end = start;
            continue;
        }

        let mut jobs: [usize; 2] = [0, 0];
        let mut njobs = 0;
        {
            let round = &mut pool[start..end];
            let (first, rest) = round.split_first_mut().unwrap();
            let mut ops: Vec<SendRecvOp> = Vec::with_capacity(2);
            if let Some(op) =
                reverse_prepare_swap(&swaps[start], first, atoms, registry, rank, per_atom)
            {
                ops.push(op);
                jobs[njobs] = start;
                njobs += 1;
            }
            if end > start + 1 {
                if let Some(op) = reverse_prepare_swap(
                    &swaps[start + 1],
                    &mut rest[0],
                    atoms,
                    registry,
                    rank,
                    per_atom,
                ) {
                    ops.push(op);
                    jobs[njobs] = start + 1;
                    njobs += 1;
                }
            }
            if !ops.is_empty() {
                comm.sendrecv_batch_f64_into(&mut ops);
            }
        } // `ops` dropped: pool borrows released, recv data now resident in pool.

        // Accumulate received forces (send-only swaps have to_proc == -1, nothing to do).
        for &j in &jobs[..njobs] {
            let swap = &swaps[j];
            if swap.to_proc != -1 {
                reverse_accumulate(swap, &pool[j].recv, atoms, registry, per_atom);
            }
        }
        end = start;
    }
}

// ── Exchange (MPI only) ──────────────────────────────────────────────────────

fn exchange_same_neighbor(
    comm: &dyn CommBackend,
    peer: i32,
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    lo_buf: &[f64],
    lo_count: f64,
    hi_buf: &[f64],
    hi_count: f64,
    aggregate: &mut Vec<f64>,
    recv_buff: &mut Vec<f64>,
) {
    let send_meta = [lo_count, lo_buf.len() as f64, hi_count, hi_buf.len() as f64];
    let mut recv_meta = [0.0f64; 4];
    comm.sendrecv_f64_into(peer, &send_meta, peer, &mut recv_meta);

    aggregate.clear();
    aggregate.reserve(lo_buf.len() + hi_buf.len());
    aggregate.extend_from_slice(lo_buf);
    aggregate.extend_from_slice(hi_buf);

    let recv_lo_len = recv_meta[1] as usize;
    let recv_hi_len = recv_meta[3] as usize;
    recv_buff.resize(recv_lo_len + recv_hi_len, 0.0);
    comm.sendrecv_f64_into(peer, aggregate, peer, recv_buff);
    unpack_exchanged(
        atoms,
        registry,
        &recv_buff[..recv_lo_len],
        recv_meta[0] as usize,
    );
    unpack_exchanged(
        atoms,
        registry,
        &recv_buff[recv_lo_len..],
        recv_meta[2] as usize,
    );
}

#[cfg(feature = "mpi_backend")]
/// Exchange (migrate) atoms that have crossed a sub-domain boundary to their new
/// owning rank. Runs at `Exchange` on full-rebuild steps, before the ghost
/// halo and neighbor list are rebuilt. One send/recv hop per dimension moves each
/// atom to the correct neighbor; each migrated atom carries its full base fields
/// plus every [`AtomData`](crate::AtomData) extension column.
pub fn exchange(
    comm: Res<CommResource>,
    topo: Res<CommTopology>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
    mut buffers: ResMut<CommBuffers>,
) {
    debug_assert_particle_invariant(&atoms, &registry, "particle exchange (entry)");
    let decomp = comm.processor_decomposition();

    // Reuse persistent exchange buffers: lo, hi, and a same-neighbor aggregate
    // payload scratch for the periodic two-rank case.
    let mut atoms_buff = std::mem::take(&mut buffers.exchange_buffs);
    atoms_buff.resize_with(3, Vec::new);
    // Reuse the border recv scratch (exchange runs before borders on a rebuild, so
    // they never overlap) for the probe-free count-first receive.
    let mut recv_buff = std::mem::take(&mut buffers.border_recv_buff);

    // Per-dimension exchange: atoms migrating in dim 0 are sent first,
    // received atoms may continue migrating in dims 1, 2.
    for dim in 0..3usize {
        if decomp[dim] == 1 {
            continue; // No exchange needed in this dimension (PBC handled elsewhere)
        }

        let lo_proc = topo.swap_directions[0][dim]; // neighbor in -dim direction
        let hi_proc = topo.swap_directions[1][dim]; // neighbor in +dim direction

        atoms_buff[0].clear(); // lo send buffer
        atoms_buff[1].clear(); // hi send buffer
        let mut lo_count = 0.0f64;
        let mut hi_count = 0.0f64;

        // Periodic-wrap-aware classification (keeps exchange single-hop):
        //
        // `pbc` runs before exchange and wraps every atom into the GLOBAL box.
        // An atom that physically left the low-edge rank's subdomain (pos < 0)
        // is wrapped to pos ≈ box_high, which is numerically `>= sub_domain_high`
        // and would naively be sent to the +dim (hi) neighbor — the WRONG
        // direction. Its true owner is the periodic lo neighbor (the rank at the
        // far high edge). With ≥3 ranks in a periodic dim those are different
        // ranks, so the raw comparison mis-routes the atom and a single hop
        // never reaches the owner (the original ≥3-proc SIGSEGV).
        //
        // Fix: wrap the coordinate into the box, keep it if it is inside the
        // slab, and otherwise send it toward the nearer face around the ring
        // (see `exchange_direction`). A single lo/hi swap per dim still
        // suffices because each atom moves at most one subdomain per step.
        let periodic = domain.periodic_flags()[dim];
        // For a triclinic box, classify ownership in fractional (lamda) coordinates
        // where the box is a unit cube (period 1) and subdomain faces are the
        // uniform lamda bounds — so the orthogonal single-hop-per-dim logic holds
        // even under shear, including simultaneous x+y decomposition.
        let triclinic = domain.triclinic;
        let box_size = if triclinic { 1.0 } else { domain.size[dim] };
        let box_lo = if triclinic { 0.0 } else { domain.boundaries_low[dim] };
        let (sub_lo, sub_hi) = if triclinic {
            (domain.sub_lamda_low[dim], domain.sub_lamda_high[dim])
        } else {
            (domain.sub_domain_low[dim], domain.sub_domain_high[dim])
        };
        let mut removed = std::mem::take(&mut buffers.exchange_removed);
        removed.clear();
        {
            let store = ParticleStore::new(&mut atoms, &registry);
            for i in (0..store.atoms().len()).rev() {
                let pr = store.atoms().pos[i];
                let pf = [pr[0] as f64, pr[1] as f64, pr[2] as f64];
                let coord = if triclinic {
                    domain.x2lamda(pf)[dim]
                } else {
                    pf[dim]
                };
                let direction =
                    exchange_direction(coord, sub_lo, sub_hi, box_lo, box_size, periodic);
                let target = match direction {
                    -1 => {
                        lo_count += 1.0;
                        0
                    }
                    1 => {
                        hi_count += 1.0;
                        1
                    }
                    _ => continue,
                };
                store
                    .pack_migrant(i, &mut atoms_buff[target])
                    .expect("exchange found invalid particle columns");
                removed.push(i);
            }
        }
        ParticleStore::new(&mut atoms, &registry)
            .swap_remove_descending(&removed)
            .expect("exchange found an invalid local particle index");
        buffers.exchange_removed = removed;

        // C7 count-first exchange. Both directions use the same 2-message
        // (meta=[atom_count, payload_len], then payload) protocol on every branch —
        // symmetric, send-only and recv-only — so neighbours always agree and the
        // probe-free `_into` receive can size its buffer exactly. The explicit
        // payload length (not a per-atom stride) handles variable-length registry
        // data (e.g. tangential contact history).

        // Periodic two-rank decompositions have the same neighbor on both
        // faces. Send [lo | hi] in one meta/payload pair instead of two pairs.
        if lo_proc != -1 && lo_proc == hi_proc {
            let (lo_hi, aggregate) = atoms_buff.split_at_mut(2);
            exchange_same_neighbor(
                &**comm,
                lo_proc,
                &mut atoms,
                &registry,
                &lo_hi[0],
                lo_count,
                &lo_hi[1],
                hi_count,
                &mut aggregate[0],
                &mut recv_buff,
            );
            continue;
        }

        // Send lo, receive from hi
        if lo_proc != -1 && hi_proc != -1 {
            let send_meta = [lo_count, atoms_buff[0].len() as f64];
            let mut recv_meta = [0.0f64; 2];
            comm.sendrecv_f64_into(lo_proc, &send_meta, hi_proc, &mut recv_meta);
            recv_buff.resize(recv_meta[1] as usize, 0.0);
            comm.sendrecv_f64_into(lo_proc, &atoms_buff[0], hi_proc, &mut recv_buff);
            unpack_exchanged(&mut atoms, &registry, &recv_buff, recv_meta[0] as usize);
        } else if lo_proc != -1 {
            let send_meta = [lo_count, atoms_buff[0].len() as f64];
            comm.send_f64(lo_proc, &send_meta);
            comm.send_f64(lo_proc, &atoms_buff[0]);
        } else if hi_proc != -1 {
            // Receive only — probe-free into persistent buffers via a recv-only
            // batch op (dest = -1): one Irecv, no MPI_Probe, no per-call heap Vec.
            let mut recv_meta = [0.0f64; 2];
            comm.sendrecv_batch_f64_into(&mut [SendRecvOp {
                dest: -1,
                send_buf: &[],
                source: hi_proc,
                recv_buf: &mut recv_meta,
            }]);
            recv_buff.resize(recv_meta[1] as usize, 0.0);
            comm.sendrecv_batch_f64_into(&mut [SendRecvOp {
                dest: -1,
                send_buf: &[],
                source: hi_proc,
                recv_buf: &mut recv_buff,
            }]);
            unpack_exchanged(&mut atoms, &registry, &recv_buff, recv_meta[0] as usize);
        }

        // Send hi, receive from lo
        if hi_proc != -1 && lo_proc != -1 {
            let send_meta = [hi_count, atoms_buff[1].len() as f64];
            let mut recv_meta = [0.0f64; 2];
            comm.sendrecv_f64_into(hi_proc, &send_meta, lo_proc, &mut recv_meta);
            recv_buff.resize(recv_meta[1] as usize, 0.0);
            comm.sendrecv_f64_into(hi_proc, &atoms_buff[1], lo_proc, &mut recv_buff);
            unpack_exchanged(&mut atoms, &registry, &recv_buff, recv_meta[0] as usize);
        } else if hi_proc != -1 {
            let send_meta = [hi_count, atoms_buff[1].len() as f64];
            comm.send_f64(hi_proc, &send_meta);
            comm.send_f64(hi_proc, &atoms_buff[1]);
        } else if lo_proc != -1 {
            // Receive only — probe-free into persistent buffers via a recv-only
            // batch op (dest = -1): one Irecv, no MPI_Probe, no per-call heap Vec.
            let mut recv_meta = [0.0f64; 2];
            comm.sendrecv_batch_f64_into(&mut [SendRecvOp {
                dest: -1,
                send_buf: &[],
                source: lo_proc,
                recv_buf: &mut recv_meta,
            }]);
            recv_buff.resize(recv_meta[1] as usize, 0.0);
            comm.sendrecv_batch_f64_into(&mut [SendRecvOp {
                dest: -1,
                send_buf: &[],
                source: lo_proc,
                recv_buf: &mut recv_buff,
            }]);
            unpack_exchanged(&mut atoms, &registry, &recv_buff, recv_meta[0] as usize);
        }
    }

    buffers.exchange_buffs = atoms_buff;
    buffers.border_recv_buff = recv_buff;

    // ── Single-hop safety check ──────────────────────────────────────────────
    // Exchange is intentionally single-hop (one lo/hi swap per dimension): an
    // atom can migrate at most one subdomain per step. With parallel insertion
    // every atom is born inside its owner's subdomain, so a single hop always
    // suffices — provided no atom moves more than one subdomain in a step
    // (which would mean dt/skin is too large) and no insertion placed an atom
    // far from its owner. If, after the pass, any local atom is still outside
    // this subdomain in a decomposed dimension, the single hop did NOT suffice;
    // fail hard so release builds cannot proceed with atoms owned by the wrong
    // rank and later mis-bin them.
    let mut lost = 0usize;
    for i in 0..atoms.len() {
        let lam = if domain.triclinic {
            let pr = atoms.pos[i];
            Some(domain.x2lamda([pr[0] as f64, pr[1] as f64, pr[2] as f64]))
        } else {
            None
        };
        for dim in 0..3usize {
            if decomp[dim] == 1 {
                continue;
            }
            let (coord, lo, hi) = match lam {
                Some(l) => (
                    l[dim],
                    domain.sub_lamda_low[dim],
                    domain.sub_lamda_high[dim],
                ),
                None => (
                    atoms.pos[i][dim] as f64,
                    domain.sub_domain_low[dim],
                    domain.sub_domain_high[dim],
                ),
            };
            if coord < lo || coord >= hi {
                lost += 1;
                break;
            }
        }
    }
    if lost > 0 {
        panic!(
            "SOIL exchange left {lost} local atom(s) outside rank {}'s subdomain after one migration pass. \
             At least one atom moved more than one subdomain in a step, a rebalance/deformation changed ownership by more than one rank, \
             or insertion placed an atom far from its owner. Reduce timestep/displacement, rebalance more gradually, or reinsert atoms on their owning rank; \
             multi-hop owner migration is not yet implemented.",
            comm.rank()
        );
    }
    debug_assert_particle_invariant(&atoms, &registry, "particle exchange (exit)");
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn auto_processor_grid_minimizes_surface() {
        // Cube: as close to cubic as the factorization allows.
        assert_eq!(auto_processor_grid(8, [1.0; 3], [0; 3]), Some([2, 2, 2]));
        assert_eq!(auto_processor_grid(1, [1.0; 3], [0; 3]), Some([1, 1, 1]));
        assert_eq!(auto_processor_grid(4, [1.0; 3], [0; 3]), Some([2, 2, 1]));
        // Prime count: a slab along an axis, ties going to x.
        assert_eq!(auto_processor_grid(7, [1.0; 3], [0; 3]), Some([7, 1, 1]));
        // Long box: cut the long axis.
        assert_eq!(auto_processor_grid(4, [8.0, 1.0, 1.0], [0; 3]), Some([4, 1, 1]));
        assert_eq!(auto_processor_grid(4, [1.0, 1.0, 8.0], [0; 3]), Some([1, 1, 4]));
        // Flat box: 2D grid in the plane.
        assert_eq!(auto_processor_grid(16, [4.0, 4.0, 0.25], [0; 3]), Some([4, 4, 1]));
    }

    #[test]
    fn auto_processor_grid_honors_fixed_axes() {
        assert_eq!(auto_processor_grid(8, [1.0; 3], [0, 0, 1]), Some([4, 2, 1]));
        assert_eq!(auto_processor_grid(12, [1.0; 3], [3, 0, 0]), Some([3, 2, 2]));
        assert_eq!(auto_processor_grid(6, [1.0; 3], [4, 0, 0]), None);
        assert_eq!(auto_processor_grid(6, [1.0; 3], [2, 3, 1]), Some([2, 3, 1]));
        assert_eq!(auto_processor_grid(6, [1.0; 3], [2, 2, 2]), None);
    }

    #[derive(Default)]
    struct CountingComm {
        sendrecv_into_calls: AtomicUsize,
        batch_calls: AtomicUsize,
    }

    impl CountingComm {
        fn sendrecv_into_calls(&self) -> usize {
            self.sendrecv_into_calls.load(Ordering::SeqCst)
        }
    }

    impl CommBackend for CountingComm {
        fn rank(&self) -> i32 {
            0
        }
        fn size(&self) -> i32 {
            2
        }
        fn processor_decomposition(&self) -> [i32; 3] {
            [2, 1, 1]
        }
        fn processor_position(&self) -> [i32; 3] {
            [0, 0, 0]
        }
        fn set_processor_grid(&mut self, _decomp: [i32; 3], _position: [i32; 3]) {}
        fn all_reduce_sum_f64(&self, local: f64) -> f64 {
            local
        }
        fn all_reduce_min_f64(&self, local: f64) -> f64 {
            local
        }
        fn barrier(&self) {}
        fn send_f64(&self, _dest: i32, _buf: &[f64]) {
            unreachable!("CountingComm only supports sendrecv in comm aggregation tests");
        }
        fn recv_f64(&self, _source: i32) -> Vec<f64> {
            unreachable!("CountingComm only supports sendrecv in comm aggregation tests");
        }
        fn recv_f64_any(&self) -> Vec<f64> {
            unreachable!("CountingComm only supports sendrecv in comm aggregation tests");
        }
        fn sendrecv_f64(&self, _dest: i32, _send_buf: &[f64], _source: i32) -> Vec<f64> {
            unreachable!("CountingComm only supports sendrecv_f64_into in comm aggregation tests");
        }
        fn sendrecv_f64_into(
            &self,
            _dest: i32,
            send_buf: &[f64],
            _source: i32,
            recv_buf: &mut [f64],
        ) {
            self.sendrecv_into_calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(send_buf.len(), recv_buf.len());
            recv_buf.copy_from_slice(send_buf);
        }
        fn sendrecv_batch_f64_into(&self, ops: &mut [SendRecvOp<'_>]) {
            self.batch_calls.fetch_add(1, Ordering::SeqCst);
            for op in ops {
                assert_eq!(op.send_buf.len(), op.recv_buf.len());
                op.recv_buf.copy_from_slice(op.send_buf);
            }
        }
    }

    #[test]
    fn pos_to_rank_basic() {
        let decomp = [2, 2, 1];
        assert_eq!(pos_to_rank([0, 0, 0], decomp), 0);
        assert_eq!(pos_to_rank([0, 1, 0], decomp), 1);
        assert_eq!(pos_to_rank([1, 0, 0], decomp), 2);
        assert_eq!(pos_to_rank([1, 1, 0], decomp), 3);
    }

    #[test]
    fn pos_to_rank_3d() {
        let decomp = [2, 2, 2];
        assert_eq!(pos_to_rank([0, 0, 0], decomp), 0);
        assert_eq!(pos_to_rank([0, 0, 1], decomp), 1);
        assert_eq!(pos_to_rank([0, 1, 0], decomp), 2);
        assert_eq!(pos_to_rank([1, 0, 0], decomp), 4);
        assert_eq!(pos_to_rank([1, 1, 1], decomp), 7);
    }

    #[test]
    fn lower_bound_u32_finds_first_matching_cell() {
        let cells = [0, 0, 2, 2, 2, 5];
        assert_eq!(lower_bound_u32(&cells, 0), 0);
        assert_eq!(lower_bound_u32(&cells, 2), 2);
        assert_eq!(lower_bound_u32(&cells, 4), 5);
        assert_eq!(lower_bound_u32(&cells, 6), 6);
    }

    fn border_pack_fixture() -> (Atom, Domain, Neighbor, AtomDataRegistry) {
        let mut atoms = Atom::new();
        for (tag, x) in [0.2, 0.8, 1.3, 2.9, 3.8].into_iter().enumerate() {
            atoms.push_test_atom(tag as u32, [x, 0.5, 0.5], 0.25, 1.0);
        }
        atoms.push_test_atom(99, [0.4, 0.5, 0.5], 0.25, 1.0);
        atoms.nlocal = 5;
        atoms.nghost = 1;
        atoms.is_ghost[5] = true;

        let mut domain = Domain::new();
        domain.sub_domain_low = [0.0, 0.0, 0.0];
        domain.sub_domain_high = [4.0, 4.0, 4.0];
        domain.sub_length = [4.0, 4.0, 4.0];
        domain.ghost_cutoff = 1.0;

        let mut neighbor = Neighbor::new();
        neighbor.local_cells_valid = true;
        neighbor.sorted_local_cell = vec![0, 0, 1, 2, 3];
        neighbor.bin_count = [4, 1, 1];
        neighbor.bin_size = [1.0, 4.0, 4.0];
        neighbor.bin_origin = [0.0, 0.0, 0.0];
        neighbor.bin_total_cells = 4;

        (atoms, domain, neighbor, AtomDataRegistry::new())
    }

    #[test]
    fn binned_border_pack_matches_linear_scan_order() {
        let (mut linear_atoms, domain, _neighbor, registry) = border_pack_fixture();
        let mut linear_buff = Vec::new();
        let mut linear_indices = Vec::new();
        pack_border_atoms(
            &mut linear_atoms,
            &registry,
            None,
            0,
            0,
            [0.0; 3],
            [0.0; 3],
            &domain,
            &mut linear_buff,
            &mut linear_indices,
            6,
            domain.ghost_cutoff,
        );

        let (mut binned_atoms, domain, neighbor, registry) = border_pack_fixture();
        let mut binned_buff = Vec::new();
        let mut binned_indices = Vec::new();
        pack_border_atoms(
            &mut binned_atoms,
            &registry,
            Some(&neighbor),
            0,
            0,
            [0.0; 3],
            [0.0; 3],
            &domain,
            &mut binned_buff,
            &mut binned_indices,
            6,
            domain.ghost_cutoff,
        );

        assert_eq!(binned_indices, linear_indices);
        assert_eq!(binned_buff, linear_buff);
    }

    fn classify(pos: f64, sub_low: f64, sub_high: f64, box_size: f64, periodic: bool) -> i32 {
        exchange_direction(pos, sub_low, sub_high, 0.0, box_size, periodic)
    }

    #[test]
    fn exchange_classification_handles_slabs_wider_than_half_the_box() {
        // Two ranks after a rebalance: [0, 11.8) and [11.8, 32.64). Min-image
        // displacement from the faces would send x=30 "lo" from the wide slab;
        // it is inside and must stay.
        let l = 32.64;
        assert_eq!(classify(30.0, 11.8, l, l, true), 0);
        assert_eq!(classify(12.0, 11.8, l, l, true), 0);
        // Owned by the narrow slab now: the nearer face decides.
        assert_eq!(classify(11.0, 11.8, l, l, true), -1);
        // From the narrow slab, 14.0 is past the high face.
        assert_eq!(classify(14.0, 0.0, 11.8, l, true), 1);
        // Wrapped across the periodic seam (just below 0 -> near l).
        assert_eq!(classify(32.5, 0.0, 11.8, l, true), -1);
        // Unwrapped coordinates resolve the same way.
        assert_eq!(classify(-0.1, 0.0, 11.8, l, true), -1);
        assert_eq!(classify(l + 1.0, 0.0, 11.8, l, true), 0);
    }

    #[test]
    fn exchange_classification_periodic_wrap_4procs() {
        // Box [0,16), 4 ranks in this periodic dim → subdomain width 4.
        let box_size = 16.0;
        // Rank 0: [0,4). An atom that left at x<0 is PBC-wrapped to ≈15.9.
        // It must be routed LO (to the far-high-edge rank), not HI.
        assert_eq!(
            classify(15.9, 0.0, 4.0, box_size, true),
            -1,
            "wrapped low→ lo"
        );
        // A genuine forward migrant just past sub_high stays HI.
        assert_eq!(classify(4.1, 0.0, 4.0, box_size, true), 1, "forward → hi");
        // Interior atom stays local.
        assert_eq!(
            classify(2.0, 0.0, 4.0, box_size, true),
            0,
            "interior → local"
        );

        // Rank 3 (high edge): [12,16). An atom that left at x≥16 is wrapped to ≈0.1
        // and must be routed HI (to rank 0), not LO.
        assert_eq!(
            classify(0.1, 12.0, 16.0, box_size, true),
            1,
            "wrapped high → hi"
        );
        // Backward migrant just below sub_low stays LO.
        assert_eq!(
            classify(11.9, 12.0, 16.0, box_size, true),
            -1,
            "backward → lo"
        );
    }

    #[test]
    fn exchange_classification_nonperiodic() {
        // Non-periodic dim: plain comparison, no wrap.
        let box_size = 16.0;
        assert_eq!(classify(4.1, 0.0, 4.0, box_size, false), 1);
        assert_eq!(classify(-0.1, 0.0, 4.0, box_size, false), -1);
        assert_eq!(classify(2.0, 0.0, 4.0, box_size, false), 0);
    }

    #[test]
    fn swap_data_records_correct_info() {
        let swap = SwapData {
            send_indices: vec![0, 3, 7],
            recv_start: 100,
            recv_count: 5,
            to_proc: 1,
            from_proc: 2,
            periodic_offset: [10.0, 0.0, 0.0],
            vel_offset: [0.0, 0.0, 0.0],
        };
        assert_eq!(swap.send_indices.len(), 3);
        assert_eq!(swap.recv_start, 100);
        assert_eq!(swap.recv_count, 5);
        assert_eq!(swap.to_proc, 1);
        assert_eq!(swap.from_proc, 2);
    }

    fn same_neighbor_round_fixture() -> (Atom, CommTopology, AtomDataRegistry) {
        let mut atoms = Atom::new();
        atoms.push_test_atom(0, [1.0, 0.0, 0.0], 0.5, 1.0);
        atoms.push_test_atom(1, [2.0, 0.0, 0.0], 0.5, 1.0);
        atoms.push_test_atom(2, [0.0, 0.0, 0.0], 0.5, 1.0);
        atoms.push_test_atom(3, [0.0, 0.0, 0.0], 0.5, 1.0);
        atoms.vel[0] = [0.1, 0.2, 0.3];
        atoms.vel[1] = [0.4, 0.5, 0.6];
        atoms.nlocal = 2;
        atoms.nghost = 2;
        atoms.is_ghost[2] = true;
        atoms.is_ghost[3] = true;

        let topo = CommTopology {
            swap_directions: [[1, -1, -1], [1, -1, -1]],
            periodic_swap: [[0.0; 3], [0.0; 3]],
            swap_data: vec![
                SwapData {
                    send_indices: vec![0],
                    recv_start: 2,
                    recv_count: 1,
                    to_proc: 1,
                    from_proc: 1,
                    periodic_offset: [10.0, 0.0, 0.0],
                    vel_offset: [0.0; 3],
                },
                SwapData {
                    send_indices: vec![1],
                    recv_start: 3,
                    recv_count: 1,
                    to_proc: 1,
                    from_proc: 1,
                    periodic_offset: [20.0, 0.0, 0.0],
                    vel_offset: [0.0; 3],
                },
            ],
            maxneed: [1, 0, 0],
        };

        (atoms, topo, AtomDataRegistry::new())
    }

    #[test]
    fn forward_comm_aggregates_same_neighbor_round() {
        let (mut atoms, topo, registry) = same_neighbor_round_fixture();
        let comm = CountingComm::default();
        let mut pool = Vec::new();

        forward_comm(&mut atoms, &registry, &topo, &comm, &mut pool);

        assert_eq!(
            comm.sendrecv_into_calls(),
            1,
            "two same-neighbor swaps use one message"
        );
        assert_eq!(atoms.pos[2], [11.0 as Real, 0.0, 0.0]);
        assert_eq!(atoms.pos[3], [22.0 as Real, 0.0, 0.0]);
        assert_eq!(atoms.vel[2], [0.1 as Real, 0.2 as Real, 0.3 as Real]);
        assert_eq!(atoms.vel[3], [0.4 as Real, 0.5 as Real, 0.6 as Real]);
    }

    #[test]
    fn reverse_comm_aggregates_same_neighbor_round() {
        let (mut atoms, topo, registry) = same_neighbor_round_fixture();
        atoms.force[2] = [1.0 as Accum, 2.0 as Accum, 3.0 as Accum];
        atoms.force[3] = [4.0 as Accum, 5.0 as Accum, 6.0 as Accum];
        let comm = CountingComm::default();
        let mut pool = Vec::new();

        reverse_send_force_impl(&comm, &topo, &mut atoms, &registry, &mut pool);

        assert_eq!(
            comm.sendrecv_into_calls(),
            1,
            "two same-neighbor reverse swaps use one message"
        );
        assert_eq!(atoms.force[0], [1.0 as Accum, 2.0 as Accum, 3.0 as Accum]);
        assert_eq!(atoms.force[1], [4.0 as Accum, 5.0 as Accum, 6.0 as Accum]);
    }

    #[test]
    fn exchange_aggregates_same_neighbor_meta_and_payload() {
        let registry = AtomDataRegistry::new();
        let mut source = Atom::new();
        source.push_test_atom(10, [0.1, 0.0, 0.0], 0.5, 1.0);
        source.push_test_atom(20, [0.9, 0.0, 0.0], 0.5, 1.0);
        source.nlocal = 2;

        let mut lo = Vec::new();
        let mut hi = Vec::new();
        ParticleStore::new(&mut source, &registry)
            .pack_migrant(0, &mut lo)
            .unwrap();
        ParticleStore::new(&mut source, &registry)
            .pack_migrant(1, &mut hi)
            .unwrap();

        let mut atoms = Atom::new();
        let comm = CountingComm::default();
        let mut aggregate = Vec::new();
        let mut recv = Vec::new();
        exchange_same_neighbor(
            &comm,
            1,
            &mut atoms,
            &registry,
            &lo,
            1.0,
            &hi,
            1.0,
            &mut aggregate,
            &mut recv,
        );

        assert_eq!(
            comm.sendrecv_into_calls(),
            2,
            "same-neighbor exchange uses one meta message and one payload message"
        );
        assert_eq!(atoms.len(), 2);
        assert_eq!(
            atoms.tag,
            vec![10, 20],
            "lo payload is unpacked before hi payload, matching the old path"
        );
    }

    /// Build a 2-local + 1-ghost atom set and a single self-send periodic swap
    /// (atom 0 → ghost slot 2, offset +10 in x). Used by the two overlap tests.
    fn overlap_fixture() -> (Atom, CommTopology, AtomDataRegistry) {
        let mut atoms = Atom::new();
        atoms.push_test_atom(0, [1.0, 2.0, 3.0], 0.5, 1.0);
        atoms.push_test_atom(1, [4.0, 5.0, 6.0], 0.5, 1.0);
        atoms.push_test_atom(2, [0.0, 0.0, 0.0], 0.5, 1.0); // ghost slot
        atoms.vel[0] = [0.1, 0.2, 0.3];
        atoms.nlocal = 2;
        atoms.nghost = 1;

        let topo = CommTopology {
            swap_directions: [[-1, -1, -1], [-1, -1, -1]],
            periodic_swap: [[0.0; 3], [0.0; 3]],
            swap_data: vec![SwapData {
                send_indices: vec![0],
                recv_start: 2,
                recv_count: 1,
                to_proc: 0, // self-send (rank 0 → rank 0): periodic image onto self
                from_proc: 0,
                periodic_offset: [10.0, 0.0, 0.0],
                vel_offset: [0.0, 0.0, 0.0],
            }],
            maxneed: [1, 0, 0],
        };
        (atoms, topo, AtomDataRegistry::new())
    }

    /// `forward_comm_overlap` must (a) run the caller's overlap closure and
    /// (b) produce exactly the ghost data `forward_comm` produces. Uses the
    /// serial [`SingleProcessComm`] backend (no MPI runtime needed): the swap is
    /// a periodic self-send, so `forward_comm_overlap` takes its "no batch ran"
    /// path and still executes the closure — the substrate-level guarantee that
    /// interior work is never dropped even when there are no real MPI messages.
    #[test]
    fn forward_comm_overlap_matches_forward_comm_and_runs_closure() {
        let comm = SingleProcessComm::new();

        // Reference: standard forward_comm fills the ghost from local atom 0 + offset.
        let (mut ref_atoms, topo, registry) = overlap_fixture();
        let mut pool = Vec::new();
        forward_comm(&mut ref_atoms, &registry, &topo, &comm, &mut pool);

        // Under test: forward_comm_overlap with a closure that records it ran.
        let (mut atoms, topo2, registry2) = overlap_fixture();
        let mut pool2 = Vec::new();
        let mut ran = false;
        {
            let mut overlap = |_a: &mut Atom| {
                ran = true;
            };
            forward_comm_overlap(
                &mut atoms,
                &registry2,
                &topo2,
                &comm,
                &mut pool2,
                &mut overlap,
            );
        }

        assert!(
            ran,
            "overlap closure must execute during forward_comm_overlap"
        );
        // Ghost position = local atom 0 shifted by the periodic offset; velocity copied.
        assert_eq!(
            atoms.pos[2], ref_atoms.pos[2],
            "ghost pos must match forward_comm"
        );
        assert_eq!(
            atoms.vel[2], ref_atoms.vel[2],
            "ghost vel must match forward_comm"
        );
        assert_eq!(
            atoms.pos[2][0] as f64, 11.0,
            "pos[0]+offset applied to ghost"
        );
        assert_eq!(
            atoms.vel[2][0] as f64, 0.1,
            "ghost velocity copied from owner"
        );
    }
}
