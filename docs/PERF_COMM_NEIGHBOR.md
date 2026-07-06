# Performance review — communication & neighboring

Working status list for `soil_core` ghost/halo comm (`comm.rs`), atom migration
(`exchange`), and neighbor-list construction (`neighbor.rs`). Some items below
are implemented; open items keep their original impact/effort estimates and
describe the remaining work.

The reproducible profiling harness for the next tuning pass lives in
`examples/neighbor_comm_profile/`. It records timing, point-to-point message,
collective, and allocation counters for `borders`, `forward_comm`,
`reverse_comm`, `sort_atoms_by_bin`, and neighbor-list build on checked-in
serial and MPI layouts.

Legend: impact **H/M/L**, effort **S(mall)/M(edium)/L(arge)**.

---

## A. Communication (`comm.rs`)

### C1 — Brute-force border scan is O(N) per swap · impact H · effort M
### ✅ DONE (orthogonal fixed-cutoff path; small validation case is no-win)
`pack_border_atoms` no longer scans every local atom for each orthogonal
fixed-cutoff face swap. `borders` borrows the existing neighbor bin geometry and
the `sorted_local_cell` array produced by `sort_atoms_by_bin` on full rebuilds,
binary-searches only cells whose slab can overlap the face skin, and then applies
the exact face predicate before packing. Ghosts received by earlier swaps are
still scanned linearly for multi-hop/corner forwarding because they are not in
the local-cell index. Triclinic and per-atom-cutoff fallback paths keep the old
conservative scan.

Validation/perf (2026-07-05, `overlap_forward_validate`, 4 MPI ranks,
`SOIL_OVERLAP=0`, 5 warm-cache runs, release binary): baseline
`0.50, 0.43, 0.48, 0.47, 0.48 s` (mean `0.472 s`); bin-restricted path
`0.50, 0.48, 0.51, 0.52, 0.47 s` (mean `0.496 s`). This tiny 4096-atom validation
case is an honest no-win: the extra boundary-cell binary searches dominate the
saved local scans. Correctness was unchanged: the n1/n2/n4 validation sweep
passed 9/9, and every final per-rank CSV from the branch matched the pre-change
`origin/main` run exactly (`max|Δ|=0`) for both baseline and overlap modes.

### C2 — Fresh `Vec<usize>` allocated per swap in the rebuild · impact M-H · effort S
### ✅ DONE
`borders` now refills each swap's `send_indices` in place across full rebuilds:
it walks/extends the persistent `topo.swap_data`, temporarily takes the existing
`send_indices` buffer, lets `pack_border_atoms` clear and refill it, then assigns
the buffer back to the same `SwapData` entry. Stale swap entries are only
truncated after the rebuild. This keeps the per-swap `Vec` capacities instead of
freeing and regrowing the usual face-swap sendlists every rebuild
(`comm.rs:1071-1074`, `comm.rs:1107-1121`, `comm.rs:1202-1217`).

### C3 — Dead per-atom branch in the forward-comm hot path · impact M · effort S
### ✅ DONE
`compute_per_atom_offset` unconditionally returned `*stored_offset` (its body
documented that it must **not** recompute), yet `forward_comm` still branched
`if is_self_send { compute_per_atom_offset(…) } else { swap.periodic_offset }`
**per ghost**, both arms yielding the same value — a dead branch + dead call on
every `CommunicateOnly` step over every ghost. The branch and the function were
removed; `pack_forward_into` now applies `swap.periodic_offset` directly. Pure
win, no behavior change: the deleted function was already a no-op returning the
stored offset, so both branch arms yielded the identical value.

### C4 — Element-by-element packing in forward comm · impact L-M · effort S
### ✅ DONE
`forward_comm` pushed pos+vel one f64 at a time in the per-step hot loop.
`pack_forward_into` now builds the six base fields as a `[f64; 6]` and appends
them with one `extend_from_slice`, then immediately appends registry forward
fields as before, preserving the base-pos/base-vel/registry packing order. Pure
packing cleanup, no physics/tolerance change.

Validated: `cargo test -p soil_core comm --features precision-double` (6/6
comm tests pass); `examples/overlap_forward_validate/sweep.py` (1-, 2-, and
4-rank layouts, 9/9 checks) reports overlap vs baseline `max|Δ|=0.000e+00`
with tolerance `1e-12` on every layout.

### C5 — Many small messages: one sendrecv per swap · impact M-H · effort L
### ✅ DONE (same-neighbor round aggregation)
Forward and reverse ghost comm now detect the periodic two-rank case where a
round's lower/upper swaps both talk to the same neighbor and send `[lo | hi]` in
one `sendrecv_f64_into`, then split the receive at the recorded per-swap counts.
The opt-in f32 forward path applies the same aggregation to its rounded base
payload (and, when present, the f64 extension-field payload). Exchange migration
uses the same idea for two-rank periodic dimensions: the lo/hi count-first meta
messages become one `[lo_count, lo_len, hi_count, hi_len]` exchange followed by
one concatenated `[lo | hi]` payload, preserving the old unpack order.

Instrumentation/correctness: `cargo test -p soil_core comm --features
precision-double` includes `forward_comm_aggregates_same_neighbor_round` and
`reverse_comm_aggregates_same_neighbor_round`, which use a counting backend to
prove each same-neighbor ghost round takes one sendrecv instead of two while
producing the same ghost positions/velocities and reverse force accumulation;
`exchange_aggregates_same_neighbor_meta_and_payload` proves same-neighbor
exchange uses one meta message plus one payload message while preserving lo-then-hi
unpack order.
The SOIL validation sweeps below preserve the existing decomposition-invariance
and exchange gate results.

### C6 — Fully blocking comm; no compute/comm overlap · impact M-H · effort L
### ✅ DONE (per-swap overlap + interior/boundary compute overlap)
Every exchange *was* a blocking `sendrecv`. Classic MPI win: post ghost-position
forward comm with `Isend`/`Irecv` and overlap with local work, and/or overlap
the reverse-force return.

**Implemented:** added `CommBackend::sendrecv_batch_f64_into` + `SendRecvOp` in
`grass_mpi` (posts all `Isend`/`Irecv` for a batch, then `wait_all`). `forward_comm`
and `reverse_send_force` now process swaps in `(dim, need)` rounds and overlap the
two independent swaps of each round in one batch (6 serialized receives → 3
overlapped rounds for the common `maxneed==1` case). Rounds stay ordered because a
later round can forward corner ghosts an earlier round's receive just filled
(multi-hop). Per-swap send/recv scratch lives in `CommBuffers.{forward,reverse}_scratch`.
Validated: 2-rank + 2×2×2 8-rank Haff cooling decays correctly with no energy
injection; `bond_mpi_drift` behaviorally identical to baseline.

**Deeper win — now wired and verified:** overlapping comm with *local
force/integrator compute*. `forward_comm_overlap` (`comm.rs`) mirrors
`forward_comm` but always uses the overlap-capable batch primitive
(`grass_mpi::sendrecv_batch_overlap_f64_into`) and takes a caller closure that
runs *while the first round's ghost swaps are in flight*; `classify_boundary_atoms`
(`neighbor.rs`) supplies the interior/boundary split (interior atoms have no ghost
neighbours, so their force is halo-independent and computes during the exchange;
boundary atoms are finished once the halo lands, after the wait).

Validated at the **substrate** tier (this pass): the MPI example
`examples/overlap_forward_validate` runs a periodic
Lennard-Jones gas (generic MD — no DEM logic) with a full neighbour list so each
atom's force is a self-contained, order-preserving sum, then compares the
overlapped force against the standard non-overlapped `forward_comm` path.
`sweep.py` reports **max|Δ| = 0 (bit-identical, well inside the FP-associativity
floor)** for final positions *and* velocities on 1-, 2-, and 4-rank layouts, with
a non-empty interior population computed during the in-flight halo on every
multi-rank layout (n2: 1512 interior / 2584 boundary; n4: 927 / 3169) and the
boundary force issued only after the wait. `forward_comm_overlap` also gains a
serial unit test (`forward_comm_overlap_matches_forward_comm_and_runs_closure`).
Previously validated at the **DEM** tier by dirt's `mpi_loadbalance_validate`
(`DIRT_OVERLAP_FORCE=1`) — the same substrate primitive now proven one tier down,
so the overlap serves DEM/MD/SPH/Peridynamics alike.

### C7 — `borders`/`exchange` use the allocating, probing `sendrecv_f64` · impact M · effort M
### ✅ DONE
The symmetric paired-swap branches of `borders`/`exchange` already used the
probe-free `sendrecv_f64_into` (count-first meta, then payload into a persistent
buffer). The three **receive-only** branches (non-periodic-boundary edge ranks:
`borders` at the swap with a `from_proc` but no `to_proc`, and `exchange`'s
hi-/lo-only sides) still called `recv_f64`, which internally does
`receive_vec` → MPI_Probe + a fresh `Vec` alloc every step. They now receive the
already-count-first protocol probe-free: a single recv-only batch op (`dest = -1`
→ one `Irecv`) via `sendrecv_batch_f64_into` — the recv-side analog of
`sendrecv_f64_into` — first into a fixed `[f64; 2]` meta, then into the persistent
`recv_buff` sized from the meta. No `MPI_Probe`, no per-call heap `Vec`, on any
border/exchange branch. Kept fully in soil (reused the existing probe-free
`_into` batch primitive), so no `grass` change was needed. Validated: `cargo test
-p soil_core` (92 pass); the periodic `overlap_forward_validate` sweep (n1/n2/n4,
9/9); and a **non-periodic** 2-rank run (`boundary_x = fixed`) that exercises all
recv-only branches (600 firings) is **byte-identical** to the pre-change baseline
binary — no behavior change, transport only.

### C8 — `all_reduce_sum_f64` collective on every full rebuild · impact L-M · effort S
### ✅ DONE
`borders` no longer refreshes `atoms.natoms` with an `all_reduce_sum_f64` during
every full ghost rebuild. The code documents `natoms` as reporting-only and
leaves the global-count refresh to reporting paths that already run collectives
on their own cadence, while `borders` only resets `nlocal`/`nghost`
(`comm.rs:1044-1049`).

### C9 — Single-hop safety scan over all atoms every exchange · closed 2026-07-06
`exchange` now keeps the post-condition check in release builds and turns a
remaining outside-subdomain atom into a hard, actionable error. This preserves
correctness for skewed load balancing or large deformation steps while the
migration implementation remains single-hop.

### C10 — Ghost comm in f64; could be f32 (ties to GPU port) · impact H(bandwidth) · effort L
Prototype verdict (2026-07-05): keep f64 forward communication as the default.
SOIL now has an explicit `[comm] forward_precision = "f32"` gate that rounds and
buffers the base ghost **position/velocity** forward payload as f32 while keeping
reverse **force** accumulation and forward extension fields in f64. The current
`grass_mpi` point-to-point API is still f64-only, so this validates numerics and
SOIL pack/unpack structure but is not yet a wire-bandwidth win.

Validation evidence:
- SOIL MD-style `examples/overlap_forward_validate/sweep.py`: f64 overlap remains
  bit-identical to baseline (`max|Δ|=0` on 1/2/4 ranks, tolerance `1e-12`).
  Opt-in f32-forward drift against f64-forward after 300 LJ steps was `4.733e-05`
  on 2 ranks and `5.778e-05` on 4 ranks, about 24-29x the `2e-06` f32 ulp-scale
  reporting floor.
- DIRT/DEM consumer check: `mpi_loadbalance_validate` on 2 ranks with the local
  SOIL branch patched in and a temporary declarative config adding
  `forward_precision = "f32"` preserved all 2000 particle tags and measured
  `max|Δxyz|=4.453e-09` versus the f64-forward run, below the same f32 floor.

Next step for C10 is a grass/soil co-design PR that adds true f32 point-to-point
transport in `grass_mpi`; only then can this become a bandwidth optimization.

---

## B. Neighboring (`neighbor.rs`)

> Already in place (good): cell/bin list with precomputed stencil (O(N)), CSR
> output, counting-sort binning, uniform-cutoff fast path, `mul_add` FMA in the
> distance test, optional half list (`newton`), spatial atom sort, and
> displacement-based rebuild triggering. The items below separate closed
> follow-up work from the still-open edges.

### N1 — `sort_atoms_by_bin` allocates + O(N log N) sorts every rebuild · impact H · effort M
### ✅ DONE
`sort_atoms_by_bin` now uses an O(N) stable counting sort by bin cell, with
persistent scratch buffers on `Neighbor` (`sort_cell`, `sort_counts`,
`sort_perm`, `sort_pos_scratch`, and `sort_radius_scratch`). It takes those
buffers with `mem::take`, clears/resizes them, histograms cells, prefix-sums
counts, scatters into `perm`, applies the permutation, and returns the scratch
for reuse. There is no comparison-sort `Vec<(cell, index)>` rebuild path left
(`neighbor.rs:204-216`; `neighbor/binning.rs:578-676`).

### N2 — Sort and build recompute the identical bin assignment · impact H · effort M
### ✅ DONE
`sort_atoms_by_bin` publishes post-permutation local cell ids in
`Neighbor::sorted_local_cell` and sets `local_cells_valid` for the matching full
rebuild. `bin_neighbor_list` consumes that cache when it matches `nlocal`,
histograms those local cells directly, and starts the explicit position-to-cell
binning loop at `nlocal`, so only ghosts are recomputed on that path. The cache is
then returned and invalidated after use (`neighbor.rs:220-229`;
`neighbor/binning.rs:625-634`, `neighbor/binning.rs:743-784`).

### N3 — `displacement_exceeded` has loop-invariant PBC branches · impact M · effort M
### ✅ DONE
`displacement_exceeded` now splits the fully periodic case from the mixed/general
case. The common all-periodic path uses straight-line nearest-image math
(`d -= box * round(d / box)`) and an FMA distance reduction with no per-axis PBC
branches inside the loop; the general path keeps the explicit per-axis boundary
checks for mixed/non-periodic domains (`neighbor/rebuild.rs:38-117`).

### N4 — Rebuild threshold uses global-min skin (polydisperse) · impact M(poly)/L(mono) · effort M
### ✅ DONE
Rebuild displacement checks now save each local atom's cutoff radius at build
time and compare each atom against its own `(skin_fraction - 1) * radius` budget.
Changing a cutoff radius still forces a rebuild, and the regression test covers
the polydisperse case where a large atom may move farther than a small atom
without tripping the small atom's budget (`neighbor/rebuild.rs:10-19`,
`neighbor/rebuild.rs:66-72`, `neighbor/rebuild.rs:163-180`).

### N5 — `push_index!` capacity check branches per accepted neighbor · impact M · effort S
### ✅ DONE
Neighbor index emission now goes through `NeighborIndexWriter`, a raw-pointer
writer with a debug capacity assertion instead of a per-accepted-neighbor
`Vec::push` growth branch. Before distance filtering, the builder computes a
conservative candidate count over the selected stencil/self-cell ranges and
reserves once if needed, then writes accepted neighbors through the cached
writer until setting the final Vec length (`neighbor/binning.rs:9-25`,
`neighbor/binning.rs:837-887`, `neighbor/binning.rs:1065-1068`).

### N6 — Full-list self-cell `j == i` branch in inner loop · impact M(full)/0(half) · effort S
### ✅ DONE
The bin stencil now carries a separate `bin_stencil_nonself` plus a
`bin_stencil_self` flag, so the full-list path iterates non-self neighbor cells
without checking `offset == 0` in the inner stencil loop. For the self cell,
`newton=false` scans `[cell_start, self_pos)` and `(self_pos, end)` separately,
dropping the per-candidate `j == i` branch in both uniform and polydisperse paths
(`neighbor.rs:168-173`; `neighbor/binning.rs:337-362`,
`neighbor/binning.rs:931-957`, `neighbor/binning.rs:1013-1041`).

### N7 — Two position streams in the distance test · impact M-L · effort S
`pi` is still read from `atoms.pos[i]` in both the uniform and polydisperse
distance loops, while neighbors come from the cache-friendly `sorted_pos` stream
inside the range appenders. Since locals are bin-sorted after
`sort_atoms_by_bin`, read `pi` from `sorted_pos` too so the kernel walks a single
sequential stream (`neighbor/binning.rs:909-910`, `neighbor/binning.rs:990-991`;
range appenders consume `sorted_pos` at `neighbor/binning.rs:29-47` and
`neighbor/binning.rs:117-140`).

### N8 — Per-field scratch alloc in `apply_permutation` · impact L-M · effort M
### ✅ DONE
`Atom::apply_permutation`, `AtomDataRegistry::apply_permutation_all`, the
`#[derive(AtomData)]` output, and in-tree hand-written `AtomData` stores now
apply spatial-sort permutations by cycle-following in place. `sort_atoms_by_bin`
passes one persistent `Neighbor::sort_permutation_visited` scratch buffer through
the base atom fields and every registered extension, so rebuild-time spatial
sorts no longer allocate one full scratch vector per field/column. The legacy
entry points still work by creating a single visited buffer per call, but the
hot spatial-sort path reuses the persistent scratch.

### N9 — Redundant min/max fold of `cutoff_radius` each rebuild · impact L · effort S
`save_build_positions` still re-folds min/max over `cutoff_radius` every rebuild
to update `cached_min_skin` and `cached_uniform_cutoff_sq`. Cache a
`cutoff_is_uniform` flag at setup and skip the fold unless a plugin signals a
radius change (`neighbor/rebuild.rs:10-34`).

### N10 — SIMD distance kernel (strategic) · impact H · effort L
### ✅ DONE (runtime-gated AVX2 path)
The neighbor builder has a runtime-gated SIMD distance kernel. `NeighborConfig`
accepts `distance_kernel = "simd"`, `NeighborDistanceKernel::simd_available`
enables it only when supported, and `neighbor/binning.rs` dispatches uniform and
polydisperse range checks through AVX2 helpers on x86_64 with scalar fallback.
The GPU-port-adjacent work that remains is broader kernel/backend design, not
the original scalar-only distance-loop item (`neighbor.rs:59-73`,
`neighbor/binning.rs:50-114`, `neighbor/binning.rs:887-888`).

---

## Suggested order (cheap-and-safe first)

1. **N7** (read the owner position from the sorted-position stream) — S.
2. **N9** (avoid the min/max cutoff fold when radii are known unchanged) — S.
3. **Strategic:** C10 (true f32 point-to-point transport) — L, co-design with
   `grass_mpi` and the GPU port.

Closed items above cite the current source locations. Open items remain planning
notes until they have implementation and validation evidence.
