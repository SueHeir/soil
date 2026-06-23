# Substrate Internals

The machinery the substrate runs so physics tiers don't have to. This chapter is
the architecture overview: how per-atom state is stored, how a timestep is
structured, and how neighbor detection and the halo width stay consistent.

## Struct-of-arrays and the parallel-array invariant

Per-atom state is struct-of-arrays: `Atom` holds one `Vec` per field (`pos`,
`vel`, `force`, `mass`, …), and each physics extension registered in the
`AtomDataRegistry` holds its own per-atom `Vec`s. **Every one of these vectors
is indexed by the same atom index `i`.**

The invariant the whole crate rests on: any *structural* operation must be
applied identically across the base `Atom` **and** every registered extension,
so the arrays never desync. The structural ops are:

- `swap_remove` — remove one atom,
- `apply_permutation` — spatial re-sort,
- `truncate` — drop ghosts.

Each is mirrored on the registry by `swap_remove_all`, `apply_permutation_all`,
and `truncate_all`. A new extension gets all of this for free from
`#[derive(AtomData)]` (see [The AtomData Contract](../substrate/atomdata-contract.md)).

## Local atoms and ghosts

The arrays are partitioned: indices `0..nlocal` are **local** atoms this rank
owns; `nlocal..nlocal+nghost` are **ghosts** — read-only halo copies of atoms
owned by neighboring ranks (or periodic images). Force systems iterate local
atoms and may read ghost neighbors; ghosts are discarded and rebuilt on each
full-rebuild step.

> **Three rules a physics tier must not break.** When you write into a ghost's
> force (Newton's-third-law optimization on), guard it with
> `if newton || j < nlocal { … }`: in a full list (`newton = false`) writing a
> ghost force double-counts, because the pair appears from both atoms. Never
> resize an `AtomData` column out of band — grow columns only through the
> substrate's `unpack` (migration) and `zero` (per-step reset) paths; a stray
> push or pop on one column desyncs every parallel array silently. And a field's
> attribute (`#[forward]` / `#[reverse]` / `#[zero]` / none) *is* its parallel
> contract: misclassify it and the bug shows up only under MPI, as a silent
> divergence between ranks, never as a crash.

## The two-path timestep (`CommState`)

Each step takes one of two paths, selected by `CommState`:

- **`CommState::FullRebuild`** — the heavy path: apply PBC, migrate atoms that
  crossed a subdomain boundary (exchange), rebuild the ghost halo (borders), and
  rebuild the neighbor list.
- **`CommState::CommunicateOnly`** — the light path: skip migration and
  rebuilds, just forward-communicate fresh ghost positions.

The path is chosen by `decide_rebuild`, which runs at `PostInitialIntegration`
on `CommunicateOnly` steps. It checks whether any atom has drifted more than the
skin allows since the last build, then — and this is the load-bearing part —
agrees on the answer **globally** with an `all_reduce` across all ranks. If *any*
rank needs a rebuild, *every* rank switches to `FullRebuild` together. This is
mandatory: the heavy path issues a matched pattern of MPI send/recv calls, so if
ranks took different paths the collective calls would deadlock. The same global
agreement governs the periodic spatial sort — a sort permutes atom indices, which
would invalidate the saved ghost sendlists, so it is tied to a globally
consistent `FullRebuild` and never happens on a lone `CommunicateOnly` step.

`soil_deform` forces this path too: every box change sets `domain.bounds_changed`
and clears `neighbor.last_build_pos`, so the next step is always a `FullRebuild`.
That is why a deforming or sheared run never gets the cheap light-path step.

Within a step the systems run in the canonical order defined by
`ParticleSimScheduleSet`:

```text
Setup
  → Pre / Initial / PostInitialIntegration   (first Verlet half-kick + drift;
                                               force zeroing happens here at
                                               PostInitialIntegration)
  → Pre / Exchange
  → Pre / Neighbor
  → Pre / Force / PostForce
  → Pre / Final / PostFinalIntegration        (second half-kick + output)
```

Forward comm replicates `#[forward]` state onto ghosts before forces; reverse
comm sums `#[reverse]` ghost contributions back to owners after. (The attribute
classification is covered in [Choosing Field Attributes](../tutorial/field-attributes.md).)

## The skin / cutoff / ghost-cutoff chain

Neighbor detection and halo width are linked:

- The pairwise interaction **cutoff** comes from the per-atom `cutoff_radius`.
- The neighbor **skin** (`Neighbor::skin_fraction`) pads that cutoff so the pair
  list stays valid for several steps without a rebuild.
- The **ghost cutoff** (`Neighbor::ghost_cutoff`, mirrored on
  `Domain::ghost_cutoff`) is the padded reach — it sets how far across a
  subdomain boundary atoms must be replicated as ghosts so every local atom sees
  all its real neighbors.

Concretely, `neighbor_setup` (run once at `PostSetup`, after atoms exist)
computes the chain as:

```text
max_cutoff   = 2 · max(cutoff_radius) · skin_fraction
ghost_cutoff = max_cutoff + 2 · displacement_buffer
displacement_buffer = (skin_fraction − 1) · min(cutoff_radius)
```

The factor of two on the displacement buffer is because two approaching atoms can
each drift by the per-atom displacement budget between rebuilds. If the
`cutoff_radius` column is still empty at setup (rate-based insertion fills it
later), the ghost cutoff falls back to `bin_size` and is corrected on the first
neighbor rebuild. If the ghost cutoff ever came out smaller than the skinned
interaction cutoff, real pairs would be silently missed — deriving it from the
skinned cutoff this way is what keeps that from happening.

## Domain decomposition

The global box is split across MPI ranks as a uniform Cartesian grid: the
`[comm]` processor counts (`processors_x/y/z`, product = total rank count) cut
each axis into equal slabs, and every rank owns one rectangular subdomain. The
decomposition is computed **once**, on the first `[[run]]` stage. Later stages
can deform or shrink-wrap the box, but the `[domain]` and `[comm]` config are not
re-read — re-reading them would clobber accumulated box deformation, which is the
deform plugin's job to own. An atom belongs to the rank whose subdomain contains
its position; for a triclinic (tilted) box, ownership is decided in fractional
(lamda) coordinates, where the tilted box maps to the unit cube.

Two boundary modes have single-process restrictions worth knowing: **shrink-wrap**
boundaries (the box auto-sizes to the atoms) panic at startup under MPI, and
`soil_deform` likewise assumes a single rank. Both would need global reductions
and per-rank subdomain updates to go multi-rank.

## Atom migration (Exchange)

On a `FullRebuild` step, after PBC wrapping, the **Exchange** phase migrates any
atom that has drifted out of its owner's subdomain into the neighboring rank's.
Migration moves the *whole* atom: the base `Atom` columns and **every** registered
`AtomData` extension, packed in field-declaration order into one flat `f64`
buffer, sent to the destination rank, and unpacked there as a new local atom. The
source rank then drops the atom with a `swap_remove` applied identically across
all columns. This is why field order is a wire format and why columns must never
be resized out of band: migration is one of only two paths (the other is `zero`)
that legitimately grows or shrinks a column, and it touches all of them at once.

## Ghost / halo exchange (borders)

After migration, **borders** builds the ghost halo: every local atom within
`ghost_cutoff` of a subdomain face is replicated as a read-only ghost on the
neighboring rank (or as a periodic image across a periodic face). Borders records
a *sendlist* per face swap — which local indices were sent, how many came back,
and any periodic position offset — and reuses it for the cheap per-step passes:

- **Forward comm** (`PreForce`, every step): the owner pushes its `#[forward]`
  columns onto its ghosts, **overwriting** them, so neighbors compute against
  current values. Ghost *positions* are refreshed this way on `CommunicateOnly`
  steps without rebuilding the halo.
- **Reverse comm** (`PostForce`, every step, Newton list only): each ghost pushes
  its accumulated `#[reverse]` columns (force, torque, …) back to the owner, which
  **adds** them in. This is what makes writing a force into a ghost's slot correct
  in the tutorial — the contribution is carried home by reverse comm.

The saved sendlist is why the spatial sort can only run on a `FullRebuild`:
permuting atom indices on a light-path step would leave the sendlist pointing at
the wrong atoms, and reverse comm would accumulate forces onto the wrong owners.

## Neighbor lists

The neighbor list is a half (Newton) CSR structure built by binning. Atoms are
sorted into a uniform bin grid that covers the subdomain plus its ghost layer;
each local atom's neighbors are found by sweeping a forward-only stencil of
neighboring bins, so each pair is recorded once. The list is built only on
`FullRebuild` steps; between rebuilds the displacement check decides whether the
saved list is still valid.

Physics iterates it through one API:

```rust,ignore
for (i, j) in neighbor.pairs(nlocal) {
    // i is always local (i < nlocal); j may be local or a ghost.
}
```

With `newton = true` (the default) the list is a half list: each pair appears
once, and the force written into a ghost `j` is carried back by reverse comm.
With `newton = false` the list is full, each pair appears from both ends, and
there is no reverse comm — so a force system **must not** write `force[j]` for a
ghost `j`, or it double-counts. The `if newton || j < nlocal` guard in the
tutorial handles both cases.

The rebuild policy follows LAMMPS `neigh_modify`: displacement-based when
`every = 0` (rebuild when any atom drifts past the skin), periodic when
`every = N`, or hybrid with `check = true`. A periodic spatial re-sort
(`sort_every`) reorders atoms for cache locality and, as above, only fires on a
globally consistent full-rebuild step.

## Regions and groups

A **region** is a TOML-declared spatial primitive — `block`, `sphere`,
`cylinder`, `cone`, `plane`, or boolean `union` / `intersect` — used to place,
select, or filter atoms. A **group** is a named subset of atoms defined by an
atom-type filter, a region, or both; the built-in `"all"` group always exists. A
group that is type-only locks its membership at first build and skips the per-step
rebuild; a group with a region (or `dynamic = true`) is rebuilt every step at
`PreForce`, after ghosts are fresh, so its mask reflects positions at the start of
the force phase. Systems filter loops with the group's boolean mask. Both are
documented key-by-key in the [Config Reference](./config.md).

## Virial stress

`soil_core` accumulates the full symmetric virial stress tensor
(`xx, yy, zz, xy, xz, yz`) when a physics tier opts in. It is zeroed each step at
`PreForce` and summed over pairs during the force phase with the convention
`dx = pos[j] − pos[i]`, `f` = force on `i` from `j`; the scalar pressure is
`P = NkT/V − tr/(3V)`. `soil_print` publishes the six components as thermo columns
automatically when the resource is present.

## Assembling the substrate directly

`CorePlugins` (a convenience plugin group that bundles input / comm / domain /
neighbor / run / output) is **not** defined in `soil_core` — it lives downstream
in `dirt_core`. To assemble a sim directly on the substrate, add the plugins
yourself: `AtomPlugin`, `DomainPlugin`, `NeighborPlugin`, `CommunicationPlugin`
(plus `InputPlugin` / `RunPlugin` for the config + run loop), as
`soil_core/examples/minimal_sim.rs` shows.

Source: `soil_core` (`neighbor.rs`, `comm.rs`, `domain.rs`, `region.rs`,
`group.rs`, `virial.rs`). Every TOML knob mentioned above is collected in the
[Config Reference](./config.md). See also
[`docs/PERF_COMM_NEIGHBOR.md`](https://github.com/SueHeir/soil/blob/master/docs/PERF_COMM_NEIGHBOR.md).
