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

use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

// Re-export comm abstractions from grass_mpi so downstream users see no change.
pub use grass_mpi::{CommBackend, CommResource, SingleProcessComm, finalize_mpi};
#[cfg(feature = "mpi_backend")]
pub use grass_mpi::{MpiCommBackend, get_mpi_world};

use crate::{Atom, AtomDataRegistry, CommState, Config, Domain, Neighbor, ParticleSimScheduleSet, ScheduleSetupSet};

// ── CommConfig ──────────────────────────────────────────────────────────────

fn default_one_i32() -> i32 {
    1
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[comm]` — MPI processor grid configuration.
pub struct CommConfig {
    /// Number of MPI ranks in x dimension.
    #[serde(default = "default_one_i32")]
    pub processors_x: i32,
    /// Number of MPI ranks in y dimension.
    #[serde(default = "default_one_i32")]
    pub processors_y: i32,
    /// Number of MPI ranks in z dimension.
    #[serde(default = "default_one_i32")]
    pub processors_z: i32,
}

impl Default for CommConfig {
    fn default() -> Self {
        CommConfig {
            processors_x: 1,
            processors_y: 1,
            processors_z: 1,
        }
    }
}

// ── CommBuffers ──────────────────────────────────────────────────────────────

/// Persistent communication buffers, reused across timesteps to avoid re-allocation.
pub struct CommBuffers {
    pub border_send_buff: Vec<f64>,
    /// Per-rank exchange buffers, reused across exchange() calls.
    pub exchange_buffs: Vec<Vec<f64>>,
    /// Persistent receive buffer for the per-step ghost forward/reverse comm.
    /// Reused across timesteps so the hot path never heap-allocates a recv Vec.
    pub recv_buff: Vec<f64>,
    /// Persistent send buffer for reverse_send_force (mirrors border_send_buff,
    /// which is owned by the forward path). Avoids a fresh Vec::new() per step.
    pub reverse_send_buff: Vec<f64>,
}

impl Default for CommBuffers {
    fn default() -> Self {
        CommBuffers {
            border_send_buff: Vec::new(),
            exchange_buffs: Vec::new(),
            recv_buff: Vec::new(),
            reverse_send_buff: Vec::new(),
        }
    }
}

// ── SwapData ─────────────────────────────────────────────────────────────────

/// Saved sendlist data for a single border swap, enabling lightweight forward_comm
/// position updates without full ghost rebuild.
pub struct SwapData {
    pub send_indices: Vec<usize>,   // atom indices packed in this swap
    pub recv_start: usize,          // first ghost index for received atoms
    pub recv_count: usize,          // number of ghosts received
    pub to_proc: i32,
    pub from_proc: i32,
    pub periodic_offset: [f64; 3],  // precomputed position shift
}

// ── CommTopology ─────────────────────────────────────────────────────────────

/// Swap directions and counts for ghost communication (replaces MpiCommInternal).
/// Works for both single-process and MPI.
pub struct CommTopology {
    pub swap_directions: [[i32; 3]; 2],
    pub periodic_swap: [[f64; 3]; 2],
    pub swap_data: Vec<SwapData>,   // sendlists for forward_comm
    pub maxneed: [i32; 3],          // number of swap layers per dimension (0 = not yet computed)
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
# Number of MPI processors in each dimension
processors_x = 1
processors_y = 1
processors_z = 1"#,
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

        app.add_setup_system(comm_read_input.run_if(first_stage_only()), ScheduleSetupSet::PreSetup)
            .add_setup_system(comm_setup.run_if(first_stage_only()), ScheduleSetupSet::PostSetup)
            .add_update_system(
                borders.run_if(in_state(CommState::FullRebuild)),
                ParticleSimScheduleSet::PreNeighbor,
            )
            .add_update_system(
                forward_comm_borders.run_if(in_state(CommState::CommunicateOnly)),
                ParticleSimScheduleSet::PreNeighbor,
            )
            .add_update_system(
                reverse_send_force.label("reverse_send_force").run_if(newton_on),
                ParticleSimScheduleSet::PostForce,
            );

        #[cfg(feature = "mpi_backend")]
        app.add_update_system(
            exchange.label("exchange").run_if(in_state(CommState::FullRebuild)),
            ParticleSimScheduleSet::Exchange,
        );

        app.add_cleanup(finalize_mpi);
    }
}

// ── Shared systems ───────────────────────────────────────────────────────────

/// Read `[comm]` config and set up the processor grid decomposition.
pub fn comm_read_input(config: Res<CommConfig>, mut comm: ResMut<CommResource>) {
    if comm.rank() == 0 {
        println!(
            "Comm: processors {} {} {}",
            config.processors_x, config.processors_y, config.processors_z
        );
    }

    let decomp = [
        config.processors_x,
        config.processors_y,
        config.processors_z,
    ];
    let mul = config.processors_x * config.processors_y * config.processors_z;
    if mul != comm.size() {
        if comm.rank() == 0 {
            println!(
                "processors {} {} {} with {} processors does not match {}",
                config.processors_x,
                config.processors_y,
                config.processors_z,
                mul,
                comm.size()
            );
        }
        exit(1);
    }

    let rank = comm.rank();
    let pz = config.processors_z;
    let py = config.processors_y;
    let position = [
        rank / (py * pz),
        (rank / pz) % py,
        rank % pz,
    ];

    comm.set_processor_grid(decomp, position);
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
    dim: usize,
    swap: usize,
    periodic_offset: f64,
    domain: &Domain,
    send_buff: &mut Vec<f64>,
    scan_end: usize,
    ghost_cutoff: f64,
) -> (i32, Vec<usize>) {
    let mut count = 0i32;
    let mut packed_indices = Vec::new();
    // Use ghost_cutoff if set (> 0), otherwise fall back to per-atom skin * 4.0 (DEM default)
    for i in 0..scan_end {
        let pos_dim = atoms.pos_component(i, dim);
        let cut = if ghost_cutoff > 0.0 { ghost_cutoff } else { atoms.cutoff_radius[i] * 4.0 };
        let in_skin = if swap == 0 {
            pos_dim < domain.sub_domain_low[dim] + cut
        } else {
            pos_dim >= domain.sub_domain_high[dim] - cut
        };
        if in_skin {
            let mut change_pos = [0.0; 3];
            change_pos[dim] = periodic_offset * domain.size[dim];
            atoms.pack_border(i, change_pos, send_buff);
            registry.pack_all(i, send_buff);
            packed_indices.push(i);
            count += 1;
        }
    }
    (count, packed_indices)
}

fn unpack_ghost_atoms(atoms: &mut Atom, registry: &AtomDataRegistry, buf: &[f64], count: usize) {
    atoms.reserve(count);
    let data = &buf[..buf.len() - 1];
    let mut pos = 0;
    for _ in 0..count {
        pos += atoms.unpack_atom(&data[pos..], true);
        pos += registry.unpack_all(&data[pos..]);
        atoms.nghost += 1;
    }
}

// ── Forward comm (lightweight ghost position update) ─────────────────────────

/// Per-atom stride in forward_comm: pos(3) + vel(3) = 6 f64s (base fields only).
const FORWARD_PACK_SIZE: usize = 6;

/// For self-sends (single-process periodic), compute the periodic offset for
/// an atom's ghost. Uses the stored swap direction to determine which periodic
/// image to create.
///
/// The stored_offset sign indicates the swap direction:
///   +size → ghost should be above (atom was near low boundary, swap=0)
///   -size → ghost should be below (atom was near high boundary, swap=1)
///
/// After PBC wrapping, the atom may have moved to the other side of the box.
/// We use the stored direction sign and the atom's current position to ensure
/// the ghost always ends up on the opposite side from the atom.
#[inline]
fn compute_per_atom_offset(
    _pos: &[f64; 3],
    stored_offset: &[f64; 3],
    _boundaries_low: &[f64; 3],
    _boundaries_high: &[f64; 3],
    _size: &[f64; 3],
) -> [f64; 3] {
    // The periodic-image offset of a ghost is FIXED at the value established when
    // the neighbor list was built (`stored_offset`). It must NOT be recomputed or
    // flipped from the atom's *current* position: between rebuilds an atom can drift
    // across a periodic boundary, and flipping the offset then jerks the ghost by a
    // full box length — dropping a real contact and reintroducing it with deep
    // overlap at the next rebuild, a spurious force that injects energy (this was a
    // real Haff-cooling energy-conservation bug, previously masked by a forced
    // rebuild on every boundary crossing). The ghost simply tracks its owner via the
    // constant shift; this stays correct as long as atoms move less than the
    // neighbor skin between rebuilds (guaranteed by the rebuild schedule) and pbc
    // wraps + ghost refresh happen together on a FullRebuild.
    *stored_offset
}

fn unpack_forward(msg: &[f64], atoms: &mut Atom, registry: &AtomDataRegistry, recv_start: usize, recv_count: usize, extra_per_atom: usize) {
    let stride = FORWARD_PACK_SIZE + extra_per_atom;
    for k in 0..recv_count {
        let base = k * stride;
        atoms.pos[recv_start + k] = [msg[base], msg[base + 1], msg[base + 2]];
        atoms.vel[recv_start + k] = [msg[base + 3], msg[base + 4], msg[base + 5]];
        if extra_per_atom > 0 {
            registry.unpack_forward_all(recv_start + k, &msg[base + FORWARD_PACK_SIZE..]);
        }
    }
}

fn forward_comm(
    atoms: &mut Atom,
    registry: &AtomDataRegistry,
    topo: &CommTopology,
    comm: &dyn CommBackend,
    domain: &Domain,
    buf: &mut Vec<f64>,
    recv: &mut Vec<f64>,
) {
    let rank = comm.rank();
    let extra = registry.forward_comm_size();
    let stride = FORWARD_PACK_SIZE + extra;
    for swap in &topo.swap_data {
        buf.clear();
        buf.reserve(swap.send_indices.len() * stride);

        // For self-sends (single-process periodic), recompute the periodic offset
        // per atom based on current position. After PBC wrapping, an atom originally
        // near one boundary may now be near the other, so the stored offset is stale.
        // Recomputing ensures the ghost ends up at the correct periodic image.
        let is_self_send = swap.to_proc == rank;

        // Pack positions and velocities (6 f64s per atom) + registry forward fields
        for &idx in &swap.send_indices {
            let offset = if is_self_send {
                compute_per_atom_offset(
                    &atoms.pos[idx],
                    &swap.periodic_offset,
                    &domain.boundaries_low,
                    &domain.boundaries_high,
                    &domain.size,
                )
            } else {
                swap.periodic_offset
            };
            buf.push(atoms.pos[idx][0] + offset[0]);
            buf.push(atoms.pos[idx][1] + offset[1]);
            buf.push(atoms.pos[idx][2] + offset[2]);
            buf.push(atoms.vel[idx][0]);
            buf.push(atoms.vel[idx][1]);
            buf.push(atoms.vel[idx][2]);
            registry.pack_forward_all(idx, buf);
        }

        if swap.to_proc == rank {
            // Self-send: copy directly into ghost data
            unpack_forward(buf, atoms, registry, swap.recv_start, swap.recv_count, extra);
        } else {
            // MPI: sendrecv is deadlock-free regardless of symmetric/asymmetric
            if swap.to_proc != -1 && swap.from_proc != -1 {
                // Common periodic path: recv size is known (recv_count * stride), so
                // use the probe-free, allocation-free into-variant with a persistent buffer.
                recv.resize(swap.recv_count * stride, 0.0);
                comm.sendrecv_f64_into(swap.to_proc, buf, swap.from_proc, recv);
                unpack_forward(recv, atoms, registry, swap.recv_start, swap.recv_count, extra);
            } else if swap.to_proc != -1 {
                comm.send_f64(swap.to_proc, buf);
            } else if swap.from_proc != -1 {
                let msg = comm.recv_f64(swap.from_proc);
                unpack_forward(&msg, atoms, registry, swap.recv_start, swap.recv_count, extra);
            }
        }
    }
}

// ── Unified borders ──────────────────────────────────────────────────────────

/// Lightweight ghost position update: forward-comm only (no full rebuild).
///
/// Gated by `run_if(in_state(CommState::CommunicateOnly))`.
/// Recomputes per-atom periodic offsets for self-sends, so ghost
/// positions stay correct even when atoms cross PBC boundaries between rebuilds.
pub fn forward_comm_borders(
    comm: Res<CommResource>,
    topo: Res<CommTopology>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
    mut buffers: ResMut<CommBuffers>,
) {
    let mut send_buff = std::mem::take(&mut buffers.border_send_buff);
    let mut recv_buff = std::mem::take(&mut buffers.recv_buff);
    forward_comm(&mut atoms, &registry, &topo, &**comm, &domain, &mut send_buff, &mut recv_buff);
    buffers.border_send_buff = send_buff;
    buffers.recv_buff = recv_buff;
}

/// Build ghost atoms by scanning near-boundary locals and sending them to neighbors.
///
/// Gated by `run_if(in_state(CommState::FullRebuild))`.
pub fn borders(
    comm: Res<CommResource>,
    mut topo: ResMut<CommTopology>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
    mut buffers: ResMut<CommBuffers>,
) {
    let mut send_buff = std::mem::take(&mut buffers.border_send_buff);

    // Full ghost rebuild: remove old ghosts first
    if atoms.nghost > 0 {
        atoms.truncate_to_nlocal();
        registry.truncate_all(atoms.nlocal as usize);
        atoms.nghost = 0;
    }

    let local_count = atoms.len() as f64;
    let global_count = comm.all_reduce_sum_f64(local_count);
    atoms.natoms = global_count as u64;
    atoms.nlocal = atoms.len() as u32;
    atoms.nghost = 0;

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
                    println!("Comm: multi-hop ghost communication in dim {}: need={}", dim_name, topo.maxneed[dim]);
                }
            }
        }
    }

    // Clear sendlists for fresh recording
    topo.swap_data.clear();

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

                // Precompute periodic position shift for sendlist
                let mut pbc_shift = [0.0; 3];
                pbc_shift[dim] = periodic_offset * domain.size[dim];

                let (count, packed_indices) = pack_border_atoms(
                    &mut atoms,
                    &registry,
                    dim,
                    swap,
                    periodic_offset,
                    &domain,
                    &mut send_buff,
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
                    // MPI: sendrecv is deadlock-free
                    send_buff.push(count as f64);
                    let msg = comm.sendrecv_f64(to_proc, &send_buff, from_proc);
                    let recv_count = msg[msg.len() - 1] as usize;
                    swap_recv_start = atoms.len();
                    swap_recv_count = recv_count;
                    unpack_ghost_atoms(&mut atoms, &registry, &msg, recv_count);
                } else if to_proc != -1 {
                    // Send only (non-periodic boundary, no neighbor to receive from)
                    send_buff.push(count as f64);
                    comm.send_f64(to_proc, &send_buff);
                    swap_recv_start = atoms.len();
                    swap_recv_count = 0;
                } else if from_proc != -1 {
                    // Receive only
                    let msg = comm.recv_f64(from_proc);
                    let recv_count = msg[msg.len() - 1] as usize;
                    swap_recv_start = atoms.len();
                    swap_recv_count = recv_count;
                    unpack_ghost_atoms(&mut atoms, &registry, &msg, recv_count);
                } else {
                    swap_recv_start = atoms.len();
                    swap_recv_count = 0;
                }

                topo.swap_data.push(SwapData {
                    send_indices: packed_indices,
                    recv_start: swap_recv_start,
                    recv_count: swap_recv_count,
                    to_proc,
                    from_proc,
                    periodic_offset: pbc_shift,
                });
            }
            scan_end = atoms.nlocal as usize + atoms.nghost as usize;
        }
    }
    buffers.border_send_buff = send_buff;
}

// ── Unified reverse send force ────────────────────────────────────────────────

/// Accumulate ghost forces back onto their owner atoms.
///
/// Iterates swap data in reverse order (mirroring `borders`), packing
/// force + registry reverse fields from each ghost and sending them to the
/// rank that owns the original atom.
pub fn reverse_send_force(
    comm: Res<CommResource>,
    topo: Res<CommTopology>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    mut buffers: ResMut<CommBuffers>,
) {
    let rank = comm.rank();
    // Reuse persistent send/recv buffers instead of allocating fresh Vecs each step.
    let mut send_buff = std::mem::take(&mut buffers.reverse_send_buff);
    let mut recv_buff = std::mem::take(&mut buffers.recv_buff);

    // Per-atom stride: force×3 + registry reverse fields. The owner of each
    // returned force is *not* transmitted: ghosts were built from `swap.send_indices`
    // in order (see `borders`), and that mapping is stable across the
    // CommunicateOnly steps between rebuilds (the same invariant `forward_comm`
    // relies on every step). So the k-th returned force belongs to
    // `swap.send_indices[k]` — verified empirically to always equal the formerly
    // transmitted `origin_index`. Dropping `tag`+`origin_index` saves 2 f64/atom.
    let per_atom = 3 + registry.reverse_comm_size();

    // Iterate swaps in reverse order (mirrors borders() forward order)
    for swap in topo.swap_data.iter().rev() {
        send_buff.clear();
        send_buff.reserve(swap.recv_count * per_atom);

        // Pack force (+ registry reverse fields) per ghost atom, in ghost-index order.
        for i in swap.recv_start..(swap.recv_start + swap.recv_count) {
            debug_assert!(atoms.is_ghost[i], "reverse_send_force: atom {} is not ghost", i);
            send_buff.push(atoms.force[i][0]);
            send_buff.push(atoms.force[i][1]);
            send_buff.push(atoms.force[i][2]);
            registry.pack_reverse_all(i, &mut send_buff);
        }

        if swap.from_proc == rank {
            // Self-send: apply forces locally. recv_count == send_indices.len() here.
            for k in 0..swap.recv_count {
                let base = k * per_atom;
                let origin = swap.send_indices[k];
                atoms.force[origin][0] += send_buff[base];
                atoms.force[origin][1] += send_buff[base + 1];
                atoms.force[origin][2] += send_buff[base + 2];
                if per_atom > 3 {
                    registry.unpack_reverse_all(origin, &send_buff[base + 3..]);
                }
            }
        } else if swap.from_proc != -1 && swap.to_proc != -1 {
            // Common periodic path: we receive forces for exactly the atoms we
            // originally sent to to_proc (swap.send_indices), in order. Size is
            // known, so use the probe-free into-variant with the persistent buffer.
            let recv_count = swap.send_indices.len();
            recv_buff.resize(recv_count * per_atom, 0.0);
            comm.sendrecv_f64_into(swap.from_proc, &send_buff, swap.to_proc, &mut recv_buff);
            for k in 0..recv_count {
                let base = k * per_atom;
                let origin = swap.send_indices[k];
                atoms.force[origin][0] += recv_buff[base];
                atoms.force[origin][1] += recv_buff[base + 1];
                atoms.force[origin][2] += recv_buff[base + 2];
                if per_atom > 3 {
                    registry.unpack_reverse_all(origin, &recv_buff[base + 3..]);
                }
            }
        } else if swap.from_proc != -1 {
            comm.send_f64(swap.from_proc, &send_buff);
        } else if swap.to_proc != -1 {
            let msg = comm.recv_f64(swap.to_proc);
            let recv_count = msg.len() / per_atom;
            for k in 0..recv_count {
                let base = k * per_atom;
                let origin = swap.send_indices[k];
                atoms.force[origin][0] += msg[base];
                atoms.force[origin][1] += msg[base + 1];
                atoms.force[origin][2] += msg[base + 2];
                if per_atom > 3 {
                    registry.unpack_reverse_all(origin, &msg[base + 3..]);
                }
            }
        }
    }

    // Return persistent buffers for reuse next step.
    buffers.reverse_send_buff = send_buff;
    buffers.recv_buff = recv_buff;
}

// ── Exchange (MPI only) ──────────────────────────────────────────────────────

#[cfg(feature = "mpi_backend")]
pub fn exchange(
    comm: Res<CommResource>,
    topo: Res<CommTopology>,
    mut atoms: ResMut<Atom>,
    domain: Res<Domain>,
    registry: Res<AtomDataRegistry>,
    mut buffers: ResMut<CommBuffers>,
) {
    let decomp = comm.processor_decomposition();

    // Reuse persistent exchange buffers (only need 2: lo and hi per dimension)
    let mut atoms_buff = std::mem::take(&mut buffers.exchange_buffs);
    atoms_buff.resize_with(2, Vec::new);

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
        // Fix: classify by the periodic minimum-image displacement from the
        // subdomain faces. A wrapped atom near box_high then has a small NEGATIVE
        // min-image displacement from the low face → correctly routed lo; a
        // genuine forward migrant just past sub_high keeps a small positive
        // displacement → routed hi. A single lo/hi swap per dim still suffices
        // because each atom moves at most one subdomain per step.
        let periodic = domain.periodic_flags()[dim];
        let box_size = domain.size[dim];
        let half = 0.5 * box_size;
        let min_image = |delta: f64| -> f64 {
            if periodic {
                if delta > half {
                    delta - box_size
                } else if delta < -half {
                    delta + box_size
                } else {
                    delta
                }
            } else {
                delta
            }
        };

        // Scan local atoms: pack those outside subdomain in this dimension
        for i in (0..atoms.len()).rev() {
            let disp_lo = min_image(atoms.pos[i][dim] - domain.sub_domain_low[dim]);
            let disp_hi = min_image(atoms.pos[i][dim] - domain.sub_domain_high[dim]);
            if disp_lo < 0.0 {
                lo_count += 1.0;
                atoms.pack_exchange(i, &mut atoms_buff[0]);
                registry.pack_all(i, &mut atoms_buff[0]);
                atoms.swap_remove(i);
                registry.swap_remove_all(i);
            } else if disp_hi >= 0.0 {
                hi_count += 1.0;
                atoms.pack_exchange(i, &mut atoms_buff[1]);
                registry.pack_all(i, &mut atoms_buff[1]);
                atoms.swap_remove(i);
                registry.swap_remove_all(i);
            }
        }

        atoms_buff[0].push(lo_count);
        atoms_buff[1].push(hi_count);

        // Send lo, receive from hi
        if lo_proc != -1 && hi_proc != -1 {
            let msg = comm.sendrecv_f64(lo_proc, &atoms_buff[0], hi_proc);
            let msg_count = msg[msg.len() - 1] as usize;
            let data = &msg[..msg.len() - 1];
            let mut pos = 0;
            for _ in 0..msg_count {
                pos += atoms.unpack_atom(&data[pos..], false);
                pos += registry.unpack_all(&data[pos..]);
            }
        } else if lo_proc != -1 {
            comm.send_f64(lo_proc, &atoms_buff[0]);
        } else if hi_proc != -1 {
            let msg = comm.recv_f64(hi_proc);
            let msg_count = msg[msg.len() - 1] as usize;
            let data = &msg[..msg.len() - 1];
            let mut pos = 0;
            for _ in 0..msg_count {
                pos += atoms.unpack_atom(&data[pos..], false);
                pos += registry.unpack_all(&data[pos..]);
            }
        }

        // Send hi, receive from lo
        if hi_proc != -1 && lo_proc != -1 {
            let msg = comm.sendrecv_f64(hi_proc, &atoms_buff[1], lo_proc);
            let msg_count = msg[msg.len() - 1] as usize;
            let data = &msg[..msg.len() - 1];
            let mut pos = 0;
            for _ in 0..msg_count {
                pos += atoms.unpack_atom(&data[pos..], false);
                pos += registry.unpack_all(&data[pos..]);
            }
        } else if hi_proc != -1 {
            comm.send_f64(hi_proc, &atoms_buff[1]);
        } else if lo_proc != -1 {
            let msg = comm.recv_f64(lo_proc);
            let msg_count = msg[msg.len() - 1] as usize;
            let data = &msg[..msg.len() - 1];
            let mut pos = 0;
            for _ in 0..msg_count {
                pos += atoms.unpack_atom(&data[pos..], false);
                pos += registry.unpack_all(&data[pos..]);
            }
        }
    }

    buffers.exchange_buffs = atoms_buff;

    // ── Single-hop safety check ──────────────────────────────────────────────
    // Exchange is intentionally single-hop (one lo/hi swap per dimension): an
    // atom can migrate at most one subdomain per step. With parallel insertion
    // every atom is born inside its owner's subdomain, so a single hop always
    // suffices — provided no atom moves more than one subdomain in a step
    // (which would mean dt/skin is too large) and no insertion placed an atom
    // far from its owner. If, after the pass, any local atom is still outside
    // this subdomain in a decomposed dimension, the single hop did NOT suffice;
    // warn once so the bug/instability is visible rather than silently
    // corrupting the bin index on the next neighbor build.
    let mut lost = 0usize;
    for i in 0..atoms.len() {
        for dim in 0..3usize {
            if decomp[dim] == 1 {
                continue;
            }
            if atoms.pos[i][dim] < domain.sub_domain_low[dim]
                || atoms.pos[i][dim] >= domain.sub_domain_high[dim]
            {
                lost += 1;
                break;
            }
        }
    }
    if lost > 0 {
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "WARNING: single-hop exchange left {} atom(s) outside rank {}'s subdomain. \
                 An atom moved more than one subdomain in a step (timestep/skin too large) \
                 or was inserted far from its owner. These atoms will be mis-binned. \
                 (This warning is emitted once.)",
                lost,
                comm.rank()
            );
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
    fn compute_per_atom_offset_follows_stored_direction() {
        let boundaries_low = [0.0, 0.0, 0.0];
        let boundaries_high = [10.0, 10.0, 10.0];
        let size = [10.0, 10.0, 10.0];

        // stored > 0 means swap=0 (low boundary ghost, shift +size)
        // stored < 0 means swap=1 (high boundary ghost, shift -size)
        let stored = [10.0, 0.0, -10.0];

        // Atom in lower half: stored direction produces ghost outside box
        let pos = [2.0, 5.0, 3.0];
        let offset = compute_per_atom_offset(&pos, &stored, &boundaries_low, &boundaries_high, &size);
        assert_eq!(offset[0], 10.0);   // stored +, ghost at 12 (outside) → keep +size
        assert_eq!(offset[1], 0.0);    // no periodic
        assert_eq!(offset[2], -10.0);  // stored -, ghost at -7 (outside) → keep -size

        // Atom in upper half
        let pos = [8.0, 5.0, 8.0];
        let offset = compute_per_atom_offset(&pos, &stored, &boundaries_low, &boundaries_high, &size);
        assert_eq!(offset[0], 10.0);   // stored +, ghost at 18 (outside) → keep +size
        assert_eq!(offset[1], 0.0);
        assert_eq!(offset[2], -10.0);  // stored -, ghost at -2 (outside) → keep -size
    }

    #[test]
    fn compute_per_atom_offset_pbc_wrapped_atom() {
        let boundaries_low = [0.0, 0.0, 0.0];
        let boundaries_high = [10.0, 10.0, 10.0];
        let size = [10.0, 10.0, 10.0];

        // Atom was originally near high boundary (z=9), stored=-10 (swap=1).
        // After PBC wrapping, atom is now at z=1. The ghost should flip to +size.
        let stored = [0.0, 0.0, -10.0];
        let pos = [5.0, 5.0, 1.0];
        let offset = compute_per_atom_offset(&pos, &stored, &boundaries_low, &boundaries_high, &size);
        // ghost at 1-10=-9 → outside → keep -size
        assert_eq!(offset[2], -10.0);

        // Atom was originally near low boundary (x=1), stored=+10 (swap=0).
        // After PBC wrapping, atom is now at x=9. Ghost at 9+10=19 → outside → keep +size
        let stored = [10.0, 0.0, 0.0];
        let pos = [9.0, 5.0, 5.0];
        let offset = compute_per_atom_offset(&pos, &stored, &boundaries_low, &boundaries_high, &size);
        assert_eq!(offset[0], 10.0);
    }

    #[test]
    fn compute_per_atom_offset_overlap_zone() {
        // Critical test: atom in the overlap zone (near midpoint) with
        // different stored directions must produce DIFFERENT ghost positions.
        let boundaries_low = [0.0, 0.0, 0.0];
        let boundaries_high = [0.006, 0.006, 0.1];
        let size = [0.006, 0.006, 0.1];

        let pos = [0.003, 0.003, 0.05]; // at midpoint of y

        // swap=0 direction: stored_offset positive
        let stored_swap0 = [0.0, 0.006, 0.0];
        let offset0 = compute_per_atom_offset(&pos, &stored_swap0, &boundaries_low, &boundaries_high, &size);

        // swap=1 direction: stored_offset negative
        let stored_swap1 = [0.0, -0.006, 0.0];
        let offset1 = compute_per_atom_offset(&pos, &stored_swap1, &boundaries_low, &boundaries_high, &size);

        // The two ghosts MUST be at different positions
        assert_ne!(offset0[1], offset1[1], "Ghosts from different swaps must have different offsets");
        // swap=0: ghost above → +0.006
        assert_eq!(offset0[1], 0.006);
        // swap=1: ghost below → -0.006
        assert_eq!(offset1[1], -0.006);
    }

    /// Mirror of the periodic min-image lo/hi classification used in `exchange`.
    /// Returns -1 for "send lo", +1 for "send hi", 0 for "keep local".
    fn classify(pos: f64, sub_low: f64, sub_high: f64, box_size: f64, periodic: bool) -> i32 {
        let half = 0.5 * box_size;
        let min_image = |delta: f64| -> f64 {
            if periodic {
                if delta > half {
                    delta - box_size
                } else if delta < -half {
                    delta + box_size
                } else {
                    delta
                }
            } else {
                delta
            }
        };
        let disp_lo = min_image(pos - sub_low);
        let disp_hi = min_image(pos - sub_high);
        if disp_lo < 0.0 {
            -1
        } else if disp_hi >= 0.0 {
            1
        } else {
            0
        }
    }

    #[test]
    fn exchange_classification_periodic_wrap_4procs() {
        // Box [0,16), 4 ranks in this periodic dim → subdomain width 4.
        let box_size = 16.0;
        // Rank 0: [0,4). An atom that left at x<0 is PBC-wrapped to ≈15.9.
        // It must be routed LO (to the far-high-edge rank), not HI.
        assert_eq!(classify(15.9, 0.0, 4.0, box_size, true), -1, "wrapped low→ lo");
        // A genuine forward migrant just past sub_high stays HI.
        assert_eq!(classify(4.1, 0.0, 4.0, box_size, true), 1, "forward → hi");
        // Interior atom stays local.
        assert_eq!(classify(2.0, 0.0, 4.0, box_size, true), 0, "interior → local");

        // Rank 3 (high edge): [12,16). An atom that left at x≥16 is wrapped to ≈0.1
        // and must be routed HI (to rank 0), not LO.
        assert_eq!(classify(0.1, 12.0, 16.0, box_size, true), 1, "wrapped high → hi");
        // Backward migrant just below sub_low stays LO.
        assert_eq!(classify(11.9, 12.0, 16.0, box_size, true), -1, "backward → lo");
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
        };
        assert_eq!(swap.send_indices.len(), 3);
        assert_eq!(swap.recv_start, 100);
        assert_eq!(swap.recv_count, 5);
        assert_eq!(swap.to_proc, 1);
        assert_eq!(swap.from_proc, 2);
    }
}
