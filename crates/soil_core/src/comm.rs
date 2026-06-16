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
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicBool, Ordering};

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::{Deserialize, Serialize};

// Re-export comm abstractions from grass_mpi so downstream users see no change.
pub use grass_mpi::{CommBackend, CommResource, SendRecvOp, SingleProcessComm, finalize_mpi};
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

/// Per-swap send/recv scratch for the overlapped, round-based ghost comm.
/// One pair of buffers per swap so the two swaps of a `(dim, need)` round can be
/// in flight simultaneously (their sends/recvs must not share a buffer). Reused
/// across timesteps so the hot path never reallocates.
#[derive(Default)]
pub struct SwapScratch {
    pub send: Vec<f64>,
    pub recv: Vec<f64>,
}

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
    /// Per-swap scratch for the overlapped forward ghost-position comm.
    pub forward_scratch: Vec<SwapScratch>,
    /// Per-swap scratch for the overlapped reverse force comm.
    pub reverse_scratch: Vec<SwapScratch>,
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
            border_recv_buff: Vec::new(),
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
    packed_indices: &mut Vec<usize>,
    scan_end: usize,
    ghost_cutoff: f64,
) -> i32 {
    let mut count = 0i32;
    packed_indices.clear();
    // Hoist everything loop-invariant out of the O(N) per-swap scan (C1):
    //  - whether ghost_cutoff is in use (set once at setup, never per atom),
    //  - the face threshold for the fixed-cutoff fast path (the DEM common case),
    //  - the periodic position shift applied to every packed ghost.
    let lo_face = domain.sub_domain_low[dim];
    let hi_face = domain.sub_domain_high[dim];
    let use_fixed_cut = ghost_cutoff > 0.0;
    let fixed_thresh = if swap == 0 { lo_face + ghost_cutoff } else { hi_face - ghost_cutoff };
    let mut change_pos = [0.0; 3];
    change_pos[dim] = periodic_offset * domain.size[dim];

    for i in 0..scan_end {
        let pos_dim = atoms.pos_component(i, dim);
        let in_skin = if use_fixed_cut {
            // Fixed ghost_cutoff: compare against the precomputed face threshold.
            if swap == 0 { pos_dim < fixed_thresh } else { pos_dim >= fixed_thresh }
        } else {
            // Polydisperse fallback: per-atom skin * 4.0 (DEM default).
            let cut = atoms.cutoff_radius[i] * 4.0;
            if swap == 0 { pos_dim < lo_face + cut } else { pos_dim >= hi_face - cut }
        };
        if in_skin {
            atoms.pack_border(i, change_pos, send_buff);
            registry.pack_all(i, send_buff);
            packed_indices.push(i);
            count += 1;
        }
    }
    count
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

/// Like [`unpack_ghost_atoms`] but for a payload that does NOT carry a trailing
/// count element (the count-first C7 path receives the exact payload, with the
/// atom count delivered separately by a tiny pre-message).
fn unpack_ghost_payload(atoms: &mut Atom, registry: &AtomDataRegistry, data: &[f64], count: usize) {
    atoms.reserve(count);
    let mut pos = 0;
    for _ in 0..count {
        pos += atoms.unpack_atom(&data[pos..], true);
        pos += registry.unpack_all(&data[pos..]);
        atoms.nghost += 1;
    }
}

/// Unpack `count` migrated (non-ghost) atoms from a count-first exchange payload
/// (no trailing count element). Mirrors [`unpack_ghost_payload`] for [`exchange`].
fn unpack_exchanged(atoms: &mut Atom, registry: &AtomDataRegistry, data: &[f64], count: usize) {
    atoms.reserve(count);
    let mut pos = 0;
    for _ in 0..count {
        pos += atoms.unpack_atom(&data[pos..], false);
        pos += registry.unpack_all(&data[pos..]);
    }
}

// ── Forward comm (lightweight ghost position update) ─────────────────────────

/// Per-atom stride in forward_comm: pos(3) + vel(3) = 6 f64s (base fields only).
const FORWARD_PACK_SIZE: usize = 6;

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

/// Append one swap's outgoing ghost pos/vel (+ registry forward fields) to `buf`.
///
/// The periodic-image offset is the FIXED value from the last neighbor build
/// (`swap.periodic_offset`); it must NOT be recomputed from the atom's current
/// position — between rebuilds an atom can drift across a periodic boundary, and
/// flipping the offset jerks the ghost a full box length, dropping a real contact
/// and reintroducing it with deep overlap at the next rebuild (a spurious force
/// that injects energy — a real Haff-cooling energy-conservation bug).
fn pack_forward_into(swap: &SwapData, buf: &mut Vec<f64>, atoms: &Atom, registry: &AtomDataRegistry) {
    let offset = swap.periodic_offset;
    for &idx in &swap.send_indices {
        buf.push(atoms.pos[idx][0] + offset[0]);
        buf.push(atoms.pos[idx][1] + offset[1]);
        buf.push(atoms.pos[idx][2] + offset[2]);
        buf.push(atoms.vel[idx][0]);
        buf.push(atoms.vel[idx][1]);
        buf.push(atoms.vel[idx][2]);
        registry.pack_forward_all(idx, buf);
    }
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
        unpack_forward(&scratch.send, atoms, registry, swap.recv_start, swap.recv_count, extra);
        return None;
    }

    // MPI op. `-1` on either side disables that half (non-periodic boundary).
    let source = swap.from_proc;
    let recv_len = if source != -1 { swap.recv_count * stride } else { 0 };
    scratch.recv.resize(recv_len, 0.0);
    Some(SendRecvOp {
        dest: swap.to_proc,
        send_buf: &scratch.send,
        source,
        recv_buf: &mut scratch.recv,
    })
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
                scratch.send.reserve((lo.send_indices.len() + hi.send_indices.len()) * stride);
                pack_forward_into(lo, &mut scratch.send, atoms, registry);
                pack_forward_into(hi, &mut scratch.send, atoms, registry);
                scratch.recv.resize((lo_rc + hi_rc) * stride, 0.0);
                comm.sendrecv_f64_into(lo.to_proc, &scratch.send, lo.from_proc, &mut scratch.recv);
            }
            let recv = &pool[r].recv;
            unpack_forward(&recv[..lo_rc * stride], atoms, registry, lo.recv_start, lo_rc, extra);
            unpack_forward(&recv[lo_rc * stride..], atoms, registry, hi.recv_start, hi_rc, extra);
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
            if let Some(op) = forward_prepare_swap(&swaps[r], first, atoms, registry, rank, stride, extra) {
                ops.push(op);
                jobs[njobs] = r;
                njobs += 1;
            }
            if end > r + 1 {
                if let Some(op) = forward_prepare_swap(&swaps[r + 1], &mut rest[0], atoms, registry, rank, stride, extra) {
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
                unpack_forward(&pool[j].recv, atoms, registry, swap.recv_start, swap.recv_count, extra);
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
    topo: Res<CommTopology>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    mut buffers: ResMut<CommBuffers>,
) {
    let mut pool = std::mem::take(&mut buffers.forward_scratch);
    forward_comm(&mut atoms, &registry, &topo, &**comm, &mut pool);
    buffers.forward_scratch = pool;
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
    let mut recv_buff = std::mem::take(&mut buffers.border_recv_buff);

    // Full ghost rebuild: remove old ghosts first
    if atoms.nghost > 0 {
        atoms.truncate_to_nlocal();
        registry.truncate_all(atoms.nlocal as usize);
        atoms.nghost = 0;
    }

    // `natoms` (global count) is reporting-only — nothing in the neighbor/force/
    // exchange/integrator path reads it. It's refreshed by `print_thermo` (which
    // already runs a collective on its cadence), so we no longer pay an all_reduce
    // collective here on every ghost rebuild.
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

                // Precompute periodic position shift for sendlist
                let mut pbc_shift = [0.0; 3];
                pbc_shift[dim] = periodic_offset * domain.size[dim];

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
                    });
                }
                let mut packed_indices = std::mem::take(&mut topo.swap_data[swap_idx].send_indices);

                let count = pack_border_atoms(
                    &mut atoms,
                    &registry,
                    dim,
                    swap,
                    periodic_offset,
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
                    // Receive only — matching 2-message protocol (meta then payload).
                    let meta = comm.recv_f64(from_proc);
                    let recv_count = meta[0] as usize;
                    let payload = comm.recv_f64(from_proc);
                    swap_recv_start = atoms.len();
                    swap_recv_count = recv_count;
                    unpack_ghost_payload(&mut atoms, &registry, &payload, recv_count);
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
}

// ── Unified reverse send force ────────────────────────────────────────────────

/// Accumulate ghost forces from owner atoms received back into local force, for
/// one swap's MPI receive. The k-th returned force belongs to `swap.send_indices[k]`
/// (the same stable ghost→origin mapping `forward_comm` relies on).
fn reverse_accumulate(swap: &SwapData, recv: &[f64], atoms: &mut Atom, registry: &AtomDataRegistry, per_atom: usize) {
    for k in 0..swap.send_indices.len() {
        let base = k * per_atom;
        let origin = swap.send_indices[k];
        atoms.force[origin][0] += recv[base];
        atoms.force[origin][1] += recv[base + 1];
        atoms.force[origin][2] += recv[base + 2];
        if per_atom > 3 {
            registry.unpack_reverse_all(origin, &recv[base + 3..]);
        }
    }
}

/// Pack one swap's ghost forces into `scratch.send`. If the swap is a self-send,
/// accumulate locally and return `None`. Otherwise size `scratch.recv` and return
/// a [`SendRecvOp`] (note: send/recv directions are *reversed* vs forward comm —
/// forces flow back from `from_proc` toward `to_proc`).
/// Append one swap's ghost forces (+ registry reverse fields) to `buf`, in
/// ghost-index order over the swap's received-ghost region.
fn pack_reverse_into(swap: &SwapData, buf: &mut Vec<f64>, atoms: &Atom, registry: &AtomDataRegistry) {
    for i in swap.recv_start..(swap.recv_start + swap.recv_count) {
        debug_assert!(atoms.is_ghost[i], "reverse_send_force: atom {} is not ghost", i);
        buf.push(atoms.force[i][0]);
        buf.push(atoms.force[i][1]);
        buf.push(atoms.force[i][2]);
        registry.pack_reverse_all(i, buf);
    }
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
    let recv_len = if source != -1 { swap.send_indices.len() * per_atom } else { 0 };
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
    let rank = comm.rank();
    let mut pool = std::mem::take(&mut buffers.reverse_scratch);

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
                scratch.send.reserve((lo.recv_count + hi.recv_count) * per_atom);
                pack_reverse_into(lo, &mut scratch.send, &atoms, &registry);
                pack_reverse_into(hi, &mut scratch.send, &atoms, &registry);
                let recv_len = (lo.send_indices.len() + hi.send_indices.len()) * per_atom;
                scratch.recv.resize(recv_len, 0.0);
                comm.sendrecv_f64_into(lo.from_proc, &scratch.send, lo.to_proc, &mut scratch.recv);
            }
            let lo_split = lo.send_indices.len() * per_atom;
            reverse_accumulate(lo, &pool[start].recv[..lo_split], &mut atoms, &registry, per_atom);
            reverse_accumulate(hi, &pool[start].recv[lo_split..], &mut atoms, &registry, per_atom);
            end = start;
            continue;
        }

        let mut jobs: [usize; 2] = [0, 0];
        let mut njobs = 0;
        {
            let round = &mut pool[start..end];
            let (first, rest) = round.split_first_mut().unwrap();
            let mut ops: Vec<SendRecvOp> = Vec::with_capacity(2);
            if let Some(op) = reverse_prepare_swap(&swaps[start], first, &mut atoms, &registry, rank, per_atom) {
                ops.push(op);
                jobs[njobs] = start;
                njobs += 1;
            }
            if end > start + 1 {
                if let Some(op) = reverse_prepare_swap(&swaps[start + 1], &mut rest[0], &mut atoms, &registry, rank, per_atom) {
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
                reverse_accumulate(swap, &pool[j].recv, &mut atoms, &registry, per_atom);
            }
        }
        end = start;
    }

    buffers.reverse_scratch = pool;
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

        // C7 count-first exchange. Both directions use the same 2-message
        // (meta=[atom_count, payload_len], then payload) protocol on every branch —
        // symmetric, send-only and recv-only — so neighbours always agree and the
        // probe-free `_into` receive can size its buffer exactly. The explicit
        // payload length (not a per-atom stride) handles variable-length registry
        // data (e.g. tangential contact history).

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
            let meta = comm.recv_f64(hi_proc);
            let payload = comm.recv_f64(hi_proc);
            unpack_exchanged(&mut atoms, &registry, &payload, meta[0] as usize);
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
            let meta = comm.recv_f64(lo_proc);
            let payload = comm.recv_f64(lo_proc);
            unpack_exchanged(&mut atoms, &registry, &payload, meta[0] as usize);
        }
    }

    buffers.exchange_buffs = atoms_buff;
    buffers.border_recv_buff = recv_buff;

    // ── Single-hop safety check (debug builds only) ──────────────────────────
    // Exchange is intentionally single-hop (one lo/hi swap per dimension): an
    // atom can migrate at most one subdomain per step. With parallel insertion
    // every atom is born inside its owner's subdomain, so a single hop always
    // suffices — provided no atom moves more than one subdomain in a step
    // (which would mean dt/skin is too large) and no insertion placed an atom
    // far from its owner. If, after the pass, any local atom is still outside
    // this subdomain in a decomposed dimension, the single hop did NOT suffice;
    // warn once so the bug/instability is visible. This is an O(N) post-condition
    // scan, so it's gated out of release builds (it's a developer assertion, not
    // production work).
    #[cfg(debug_assertions)]
    {
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
