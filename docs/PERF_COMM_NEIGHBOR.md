# Performance review — communication & neighboring

Working list of candidate optimizations for `soil_core` ghost/halo comm
(`comm.rs`), atom migration (`exchange`), and neighbor-list construction
(`neighbor.rs`). Each item: location, current behavior, problem, proposed
change, expected impact, and rough effort. Nothing here is applied yet — this
is the evaluation pass.

Legend: impact **H/M/L**, effort **S(mall)/M(edium)/L(arge)**.

---

## A. Communication (`comm.rs`)

### C1 — Brute-force border scan is O(N) per swap · impact H · effort M
`pack_border_atoms` (`comm.rs:279`) scans **every** local atom (`0..scan_end`)
on each of the up-to-6 dim×swap iterations of a full ghost rebuild, recomputing
`cut` and a low/high branch per atom. This O(N · swaps) scan is the dominant
cost of `borders`. The neighbor build already produces a spatial bin list — the
border scan could restrict itself to atoms in the boundary bins instead of the
whole subdomain. Cheapest partial win: hoist the `ghost_cutoff > 0` test out of
the loop (it's loop-invariant) and precompute the comparison threshold once.

### C2 — Fresh `Vec<usize>` allocated per swap in the rebuild · impact M-H · effort S
`pack_border_atoms` allocates `packed_indices = Vec::new()` (`comm.rs:277`) every
swap and moves it into `SwapData.send_indices`. That's 6+ heap allocations per
full rebuild, growing from empty. Since `topo.swap_data` already persists across
rebuilds, reuse the existing `send_indices` Vecs (clear + refill, keeping
capacity) instead of `swap_data.clear()` + fresh Vecs at `comm.rs:494`.

### C3 — Dead per-atom branch in the forward-comm hot path · impact M · effort S
`compute_per_atom_offset` (`comm.rs:327`) now unconditionally returns
`*stored_offset` (the body documents that it must **not** recompute). But
`forward_comm` still does `if is_self_send { compute_per_atom_offset(…5 args…) }
else { swap.periodic_offset }` **per ghost** (`comm.rs:384`), and both arms yield
the same value. This branch + dead call run every `CommunicateOnly` step (the
common step) over every ghost. Delete the branch and the function; just use
`swap.periodic_offset`. Pure win, no behavior change.

### C4 — Element-by-element packing in forward comm · impact L-M · effort S
`forward_comm` pushes pos+vel one f64 at a time (`comm.rs:395-401`) in the
per-step hot loop. Pack the 6 base fields via a `[f64;6]` + `extend_from_slice`
(or write through a reserved slice with indices) to cut bounds-check overhead.

### C5 — Many small messages: one sendrecv per swap · impact M-H · effort L
Forward and reverse comm issue a separate `sendrecv` per swap
(`comm.rs:413`, `comm.rs:633`) — up to `maxneed × 2 × 3` messages/step. At small
subdomains or with multi-hop ghosts (`maxneed > 1`) MPI latency dominates over
bandwidth. Aggregate swaps that target the same neighbor rank into one message,
or pack all three dimensions' forward update into a single exchange round.
**Partially addressed by C6's round batching:** the two swaps of a round now fly
concurrently (latency overlap), but they're still distinct messages — true
same-rank aggregation into fewer messages is still open.

### C6 — Fully blocking comm; no compute/comm overlap · impact M-H · effort L
### ✅ DONE (per-swap overlap) — Aug pass
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

**Not yet done:** overlapping comm with *local force/integrator compute* (the
deeper win) — needs the ECS systems restructured so work exists to overlap with
while messages are in flight. Left for a later pass.

### C7 — `borders`/`exchange` use the allocating, probing `sendrecv_f64` · impact M · effort M
The hot per-step path already uses the probe-free `sendrecv_f64_into` with a
persistent buffer (`comm.rs:413`, `comm.rs:633`). But `borders` (`comm.rs:535`)
and `exchange` (`comm.rs:757`, `comm.rs:780`) still call `sendrecv_f64` /
`recv_f64`, which internally `receive_vec` → MPI_Probe + fresh `Vec` alloc
(`grass_mpi:320`, `grass_mpi:334`). The probe adds a round-trip. Exchange counts
can be communicated first (small fixed message), then payload received with the
`_into` variant into a persistent buffer.

### C8 — `all_reduce_sum_f64` collective on every full rebuild · impact L-M · effort S
`borders` calls `all_reduce_sum_f64` every rebuild (`comm.rs:468`) only to set
`atoms.natoms` for reporting. A global collective is a synchronization point that
scales poorly with rank count. `natoms` changes only on insertion/removal —
recompute it lazily (when thermo/dump needs it) or only on exchange steps that
actually moved atoms across ranks.

### C9 — Single-hop safety scan over all atoms every exchange · impact L · effort S
`exchange` ends with an O(N·3) double loop (`comm.rs:815`) purely as a
post-condition check that warns once. Gate it behind `cfg!(debug_assertions)` or
a config flag so production runs don't pay it.

### C10 — Ghost comm in f64; could be f32 (ties to GPU port) · impact H(bandwidth) · effort L
All pack buffers are f64 (`atom.rs:455-468`). With the Real/Accum f32 work in the
GPU roadmap ([[gpu-port]]), ghost **position/velocity** forward comm could move
as f32, halving MPI bandwidth and pack/unpack cost. Reverse **force**
accumulation needs precision care (accumulate in f64). Natural co-design with the
GPU force kernel.

---

## B. Neighboring (`neighbor.rs`)

> Already in place (good): cell/bin list with precomputed stencil (O(N)), CSR
> output, counting-sort binning, uniform-cutoff fast path, `mul_add` FMA in the
> distance test, optional half list (`newton`), spatial atom sort, and
> displacement-based rebuild triggering. The items below are the remaining edges.

### N1 — `sort_atoms_by_bin` allocates + O(N log N) sorts every rebuild · impact H · effort M
`neighbor.rs:733-747` builds a fresh `Vec<(u32,usize)>` and a fresh `perm` Vec,
then `sort_unstable_by_key`, plus `last_build_pos.clone()` at `:759` — every
full-rebuild step. The build right below already does an O(N) counting sort that
produces the same bin order. Make the scratch Vecs persistent (`mem::take` like
the other buffers) and replace the comparison sort with a counting sort.

### N2 — Sort and build recompute the identical bin assignment · impact H · effort M
`sort_atoms_by_bin` (`:733`) computes each local atom's cell, then
`bin_neighbor_list` (`:837`) recomputes the cell for all atoms from scratch.
Have the sort write local cells into the persistent `bin_atom_cell` buffer so the
build only bins ghosts (`nlocal..total`). Roughly halves the per-atom binning
arithmetic on rebuild.

### N3 — `displacement_exceeded` has loop-invariant PBC branches · impact M · effort M
`neighbor.rs:583-600` runs every step (when `every==0`/`check`) with three
per-atom `if px/py/pz` branches and a data-dependent wrap that blocks
vectorization. Monomorphize over the PBC flags (or split all-periodic vs general,
mirroring `pbc()` at `domain.rs:424`) and use branchless min-image in the
all-periodic path.

### N4 — Rebuild threshold uses global-min skin (polydisperse) · impact M(poly)/L(mono) · effort M
`cached_min_skin` (`:551`, used `:572`) makes the smallest particle set the
displacement budget for **every** atom, so large particles trigger rebuilds
sooner than needed. Use a per-atom / per-type budget (e.g. `(sf-1)·r_i`). Matters
when the size ratio is large.

### N5 — `push_index!` capacity check branches per accepted neighbor · impact M · effort S
`neighbor.rs:910-922` tests `nidx >= indices.capacity()` on every accepted
neighbor in the innermost loop. Reserve a provably-sufficient bound once
(stencil-cells × max atoms-per-cell, or last build × growth) and drop the in-loop
check, or hoist `capacity()` into a local updated only on growth.

### N6 — Full-list self-cell `j == i` branch in inner loop · impact M(full)/0(half) · effort S
For `newton=false`, the self-cell scan carries an `if j == i { continue; }`
(`:964`, `:1027`) per candidate. Split the scan into `[start..i]` and `[i+1..end]`
to drop the branch. Also build a `stencil_full_no_self` to drop the
`offset == 0` test (`:980`, `:1042`).

### N7 — Two position streams in the distance test · impact M-L · effort S
`pi` is read from `atoms.pos[i]` (`:940`) while neighbors come from the
cache-friendly `sorted_pos`. Since locals are bin-sorted after `sort_atoms_by_bin`,
read `pi` from `sorted_pos` too so the kernel walks a single sequential stream.

### N8 — Per-field scratch alloc in `apply_permutation` · impact L-M · effort M
`atom.rs:411-419` allocates one scratch Vec **per atom field** (11 base + every
registered extension) on each spatial sort. Reuse a persistent scratch buffer or
apply the permutation in place via cycle-following (no allocation).

### N9 — Redundant min/max fold of `cutoff_radius` each rebuild · impact L · effort S
`save_build_positions` (`:545-561`) re-folds min/max over `cutoff_radius` every
rebuild even for monodisperse systems. Cache a `cutoff_is_uniform` flag at setup
and skip the fold unless a plugin signals a radius change.

### N10 — SIMD distance kernel (strategic) · impact H · effort L
The inner distance loops are scalar `mul_add` (`:949-959`); auto-vectorization is
blocked by the conditional gather (`if r2 < cutoff_sq { push }`). A manual
`std::simd` path computing 4–8 `r²` lanes then compressing passing indices is the
largest remaining win on the build itself — and the same restructuring feeds the
GPU neighbor kernel ([[gpu-port]]).

---

## Suggested order (cheap-and-safe first)

1. **C3** (delete dead forward-comm branch) — S, pure win, hottest path.
2. **C9 / C8** (gate debug scan, defer the collective) — S.
3. **N5 / N6 / N1** (drop inner-loop branches, kill rebuild allocations) — S/M.
4. **C2 / N8** (pool the per-swap / per-field allocations) — S/M.
5. **N2 / N3 / C1** (share bin assignment, monomorphize disp check, bin-restrict border scan) — M.
6. **Strategic:** C5/C6 (message aggregation + non-blocking overlap), C10/N10
   (f32 comm + SIMD kernel) — L, co-design with the GPU port.

All findings are static-analysis only; none have been benchmarked yet. Next pass
should add a microbenchmark / profiling harness to confirm the hot spots before
implementing.
