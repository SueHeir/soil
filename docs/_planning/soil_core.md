# `soil_core` Documentation Planning

> Status: planning draft — do not publish directly.

---

## Purpose

`soil_core` is the method-agnostic particle substrate in the GRASS → SOIL → physics stack. It owns everything every particle method needs regardless of physics: the base `Atom` struct-of-arrays, the `AtomDataRegistry` extension mechanism, domain decomposition, ghost/halo communication, atom migration, neighbor-list construction, regions, groups, bonds, and the virial stress accumulator. It knows nothing about contact forces, DEM material models, or any specific physics.

It is the most important crate because **all physics tiers depend on it** and every parallel correctness invariant (forward vs. reverse comm, the parallel-array invariant, the two-path timestep, the skin/ghost-cutoff chain) is defined here. Everything downstream — `dirt_atom`, `dirt_granular`, `soil_verlet`, `soil_print`, etc. — is writing against the API `soil_core` exposes.

Crate description: `"Core simulation infrastructure: TOML config, domain decomposition, MPI communication, atom data structures"` (`Cargo.toml:8`). Dependencies: `grass_scheduler`, `grass_app`, `grass_mpi`, `grass_derive`, `grass_io`, `serde`, `toml`, `rand`.

---

## Public Surface to Document

### 1. The `Atom` resource (`atom.rs:406`)

Struct-of-arrays. **Every field is indexed identically — this is the core invariant** (`lib.rs:17`).

| field | type | meaning |
|---|---|---|
| `natoms` | `u64` | total atoms across all ranks (updated at borders, via `all_reduce`) |
| `nlocal` | `u32` | atoms owned by this rank; indices `0..nlocal` |
| `nghost` | `u32` | ghost (halo) copies; indices `nlocal..nlocal+nghost` |
| `ntypes` | `usize` | number of distinct atom types |
| `dt` | `f64` | current timestep size (set by integrator) |
| `tag` | `Vec<u32>` | global unique atom id |
| `atom_type` | `Vec<u32>` | material/type index (used by groups, force coefficients) |
| `origin_index` | `Vec<i32>` | for ghosts: local index on the owning rank; for locals: 0 |
| `is_ghost` | `Vec<bool>` | true for ghost atoms |
| `pos` | `Vec<[f64; 3]>` | position |
| `vel` | `Vec<[f64; 3]>` | velocity |
| `force` | `Vec<[f64; 3]>` | accumulated force (zeroed each step at `PostInitialIntegration`) |
| `cutoff_radius` | `Vec<f64>` | per-atom interaction radius; drives neighbor cutoff and ghost cutoff |
| `mass` | `Vec<f64>` | mass |
| `inv_mass` | `Vec<f64>` | `1/mass`, cached to avoid division in integrator |
| `image` | `Vec<[i32; 3]>` | PBC image flags: how many times atom crossed each periodic boundary |

Key methods: `len()`, `is_empty()`, `swap_remove(i)`, `truncate_to_nlocal()`, `apply_permutation(perm, n)`, `reserve(n)`, `pack_exchange(i, buf)`, `pack_border(i, change_pos, vel_offset, buf)`, `unpack_atom(buf, is_ghost)`, `push_test_atom(tag, pos, radius, mass)`, `get_max_tag()`, `pos_component(i, dim)`. Wire format: `ATOM_PACK_SIZE = 17` f64s per atom (`atom.rs:65`).

`compute_ke(atoms, mask)` — kinetic energy helper, group-maskable (`atom.rs:590`).

### 2. `AtomData` trait and `AtomDataRegistry` (`atom.rs:127`, `atom.rs:163`)

**`AtomData` trait** — the one interface physics tiers implement:

| method | mandatory? | semantics |
|---|---|---|
| `as_any` / `as_any_mut` | yes | upcast for downcasting in registry |
| `truncate(n)` | yes | shrink all per-atom Vecs to `n` (drop ghosts) |
| `swap_remove(i)` | yes | remove atom `i` by swap with last |
| `pack(i, buf)` / `unpack(buf)` | yes | serialize/deserialize for migration + restart |
| `apply_permutation(perm, n)` | yes | spatial reorder |
| `pack_forward(i, buf)` / `unpack_forward(i, buf)` | no-op default | replicate `#[forward]` to ghosts (overwrite) |
| `pack_reverse(i, buf)` / `unpack_reverse(i, buf)` | no-op default | accumulate `#[reverse]` from ghosts (`+=`) |
| `zero(n)` | no-op default | zero `#[zero]` fields for `0..n` atoms |
| `forward_comm_size()` / `reverse_comm_size()` | default 0 | f64s per atom in each comm direction |

Two hard contracts (`atom.rs:40–52`):
- **Forward = overwrite, reverse = accumulate.** `unpack_forward` assigns; `unpack_reverse` does `+=`.
- **`*_comm_size` must equal the number of f64s `pack_*` pushes per atom**, or buffer striding desyncs silently.

Field declaration order is the migration and restart wire layout — reordering fields is a format break (`atom.rs:51`).

**`register_atom_data!(app, value)`** macro (`atom.rs:107`) — call from `Plugin::build()` after `AtomPlugin` is added. Panics if the registry is missing or type is already registered.

**`AtomDataRegistry`** — `TypeId`-keyed `Vec` of `RefCell<Box<dyn AtomData>>` (`atom.rs:163`). Internal `forward_stores` / `reverse_stores` index caches skip no-op extensions in the per-step hot path (`atom.rs:166–171`). Key methods:
- `register<T>(data)` — panics on duplicate type (`atom.rs:191`)
- `get<T>() -> Option<Ref<T>>` / `get_mut<T>() -> Option<RefMut<T>>`
- `expect<T>(ctx) -> Ref<T>` / `expect_mut<T>(ctx) -> RefMut<T>` — panics with context message
- `truncate_all(n)`, `swap_remove_all(i)`, `apply_permutation_all(perm, n)`
- `pack_all(i, buf)`, `unpack_all(buf)`, `pack_forward_all(i, buf)`, `unpack_forward_all(i, buf)`, `pack_reverse_all(i, buf)`, `unpack_reverse_all(i, buf)`
- `zero_all(n)`
- `pack_all_for_restart(nlocal)`, `unpack_all_from_restart(buffers)`

**`AtomPlugin`** (`atom.rs:608`) — registers `Atom` and `AtomDataRegistry` resources; adds `remove_ghost_atoms` (gated on `FullRebuild`) and `zero_all_forces` at `PostInitialIntegration`.

### 3. `CommState` enum (`lib.rs:87`)

```rust
pub enum CommState { FullRebuild, CommunicateOnly }
```

Controls per-step path selection. `FullRebuild` = heavy path (PBC, exchange, borders, neighbor rebuild). `CommunicateOnly` = light path (forward comm only). Default = `FullRebuild`. Systems gate on this via `run_if(in_state(CommState::FullRebuild))`.

### 4. `ParticleSimScheduleSet` phases (`schedule.rs:22`)

14-phase enum for velocity-Verlet DEM/MD scheduling. Systems sort by phase, then topologically within a phase by `before`/`after` labels.

| index | phase | typical use |
|---|---|---|
| 0 | `Setup` | per-step bookkeeping (timestep counter) |
| 1 | `PreInitialIntegration` | pre-kick hooks |
| 2 | `InitialIntegration` | first half-kick (v += ½a·dt) |
| 3 | `PostInitialIntegration` | force zeroing, ghost removal (gated FullRebuild), decide_rebuild |
| 4 | `PreExchange` | shrink-wrap update, PBC wrapping (gated FullRebuild) |
| 5 | `Exchange` | MPI atom migration (gated FullRebuild) |
| 6 | `PreNeighbor` | spatial sort by bin (`sort_atoms_by_bin`), border ghost rebuild |
| 7 | `Neighbor` | bin-based neighbor list build (gated FullRebuild) |
| 8 | `PreForce` | group rebuild, forward comm (ghost position/velocity update) |
| 9 | `Force` | physics force computation |
| 10 | `PostForce` | reverse comm (ghost force → owner accumulation) |
| 11 | `PreFinalIntegration` | pre-second-kick hooks |
| 12 | `FinalIntegration` | second half-kick (v += ½a·dt) |
| 13 | `PostFinalIntegration` | output, end-of-step fixes |

`ScheduleSetupSet` (re-exported from `grass_app`) has `Setup` / `PostSetup` for one-time initialization systems. `verlet_schedule_warnings` (`schedule.rs:65`) is a callback that emits warnings for common misconfigurations (missing Force, asymmetric Verlet, no integrator).

### 5. `Neighbor` resource and `NeighborPlugin` (`neighbor.rs:87`, `neighbor.rs:516`)

Half CSR neighbor list (Newton flag). Key fields:

| field | meaning |
|---|---|
| `skin_fraction` | pair cutoff multiplier: `(r_i + r_j) * skin_fraction` (`neighbor.rs:89`) |
| `neighbor_offsets` | CSR row offsets; `offsets[i]..offsets[i+1]` = neighbors of local atom `i` |
| `neighbor_indices` | flat CSR column indices (local or ghost) |
| `every` | rebuild every N steps (0 = displacement-based only) (`neighbor.rs:125`) |
| `check` | if `every>0`, also check displacement each step (`neighbor.rs:126`) |
| `ghost_cutoff` | comm distance for ghosts = `max_cutoff + 2·displacement_buffer` (`neighbor.rs:129`) |
| `newton` | true = half list + reverse comm; false = full list, no reverse comm (`neighbor.rs:173`) |
| `cached_min_skin` | min cutoff radius cached at rebuild for displacement threshold |
| `cached_uniform_cutoff_sq` | fast-path cutoff² for monodisperse systems; `None` for polydisperse |
| `sort_every` | spatial sort interval for cache locality (0 = disabled) |
| `bin_count` | `[nx, ny, nz]` bins including ghost layers |
| `bin_stencil_forward` | forward-only stencil cell offsets for half-list construction |

**`Neighbor::pairs(nlocal) -> PairIter`** (`neighbor.rs:224`) — primary iteration API. Yields `(i, j)` with `i < nlocal`; `j` may be local or ghost. Uses unsafe unchecked indexing in the hot path. For `newton=true`, each pair appears once; for `newton=false`, the list is full and each pair appears twice (from both atoms' perspectives — actually both lists are iterated, so the pair `(i,j)` and `(j,i)` are not the same — see the newton flag semantics below).

Systems registered: `decide_rebuild` (at `PostInitialIntegration`, `CommunicateOnly` only), `sort_atoms_by_bin` (at `PreNeighbor`), `bin_neighbor_list` (at `Neighbor`, `FullRebuild` only).

`neighbor_setup` (`neighbor.rs:604`) — computes `ghost_cutoff` from `max_skin` + displacement buffer; must run at `PostSetup` after atoms exist. Sets `domain.ghost_cutoff` too.

### 6. `Domain` resource and `DomainPlugin` (`domain.rs:105`, `domain.rs:302`)

| field | meaning |
|---|---|
| `boundaries_low/high` | global box corners |
| `sub_domain_low/high` | this rank's subdomain corners |
| `sub_length` | subdomain edge lengths |
| `size` | global box edge lengths |
| `volume` | global box volume |
| `boundary_type` | `[BoundaryType; 3]` — `Periodic`, `Fixed`, or `ShrinkWrap` |
| `shrink_wrap_padding` | explicit padding for shrink-wrap; 0 = auto (use `ghost_cutoff`) |
| `ghost_cutoff` | comm cutoff; set by `neighbor_setup`, read by `borders` |
| `tilt` | `[xy, xz, yz]` triclinic tilt factors (only `xy` used today — Lees–Edwards) |
| `triclinic` | true if any tilt is non-zero; gates triclinic code paths |
| `boundary_vel` | streaming velocity for Lees–Edwards y-face crossing (`γ̇·Ly, 0, 0`) |
| `sub_lamda_low/high` | subdomain in fractional coordinates; used by triclinic comm/exchange |
| `bounds_changed` | set true by shrink-wrap; cleared by `sort_atoms_by_bin` after bin recompute |

Key methods: `x2lamda(r)`, `lamda2x(lam)`, `is_periodic(d)`, `is_shrink_wrap(d)`, `periodic_flags()`, `update_derived()`.

`decompose_domain(config, comm)` — uniform Cartesian grid decomposition (`domain.rs:235`). First-stage-only (guarded by `run_if(first_stage_only())`); domain bounds set once and mutated by deform/shrink-wrap, not re-read from config (`domain.rs:333`).

`BoundaryType` enum: `Periodic`, `Fixed`, `ShrinkWrap` (`domain.rs:29`).

### 7. Comm (`comm.rs`)

`CommConfig` — `[comm]` TOML section: `processors_x/y/z`.

`CommunicationPlugin` — registers `CommResource`, `CommBuffers`, border ghost comm, forward/reverse comm systems.

`SwapData` — saved sendlist for one border swap (`comm.rs:114`): `send_indices`, `recv_start`, `recv_count`, `to_proc`, `from_proc`, `periodic_offset`.

`CommBuffers` — persistent buffers reused across timesteps to avoid re-allocation (`comm.rs:76`): `border_send_buff`, `exchange_buffs`, `recv_buff`, `reverse_send_buff`, `forward_scratch`, `reverse_scratch`, `border_recv_buff`.

Key systems (not fully shown in this reading but described in module doc `comm.rs:15`):
1. `exchange` — migrate atoms that left subdomain (MPI only, FullRebuild)
2. `borders` — create ghost copies (FullRebuild)
3. `forward_comm` — lightweight ghost position/velocity update (every step, `PreForce`)
4. `reverse_send_force` — accumulate ghost forces back to owners (every step, `PostForce`)

### 8. `Region` enum (`region.rs:40`)

TOML-deserializable spatial primitives (tagged with `type =`):
- `Block { min, max }` — axis-aligned bounding box
- `Sphere { center, radius }`
- `Cylinder { center, radius, axis, lo, hi }` — `axis` is `"x"/"y"/"z"`
- `Cone { center, axis, rad_lo, rad_hi, lo, hi }` — linear frustum
- `Plane { point, normal }` — positive-side halfspace
- `Union { regions }` — OR
- `Intersect { regions }` — AND

Methods: `contains(pos)`, `random_point_inside(rng)`, `closest_point_on_surface(pos) -> SurfaceResult`.

`SurfaceResult { point, normal, distance }` — signed distance (negative = inside) and outward normal (`region.rs:22`).

### 9. Groups (`group.rs`)

`GroupDef` — TOML `[[group]]` entry: `name`, `type` (atom_types), `region`, `dynamic`.

`Group` — named subset with `mask: Vec<bool>` and `count`.

`GroupRegistry` — registry always containing built-in `"all"` group. Key methods: `get(name)`, `expect(name)`, `validate_name(name, ctx)`, `mask_for(group_name)`.

`group_includes(mask, i)` — inline helper for group-filtered loops (`group.rs:103`).

Membership evaluation: type filter AND region filter. Static groups (type-only, no region, `dynamic` unset) lock at first build and skip per-step rebuild. Dynamic groups (region set, or `dynamic = true`) rebuild every step at `PreForce` (`group.rs:173`). Stage overrides can redefine groups by name.

### 10. Bonds (`bond.rs`)

`BondStore` — `AtomData` extension: `bonds: Vec<Vec<BondEntry>>`. Each atom stores its neighbor bond list; both partners of a bond hold a record (A→B and B→A).

`BondEntry { partner_tag, bond_type, r0 }` — partner global tag, bond type, reference length.

`BondStore::has_bond(i, partner_tag)`, `are_excluded(i, j, tags)` — 1-2 and 1-3 pair exclusion check.

### 11. `VirialStress` (`virial.rs:19`)

Full 3×3 symmetric virial: `xx, yy, zz, xy, xz, yz`. Zeroed each step at `PreForce` by `zero_virial_stress`. Sign convention: `dx = pos[j] - pos[i]`, `fx = force on i from j`; pressure `P = NkT/V - trace/(3V)`.

### 12. Prelude / root re-exports (`lib.rs:95–119`)

All modules star-re-exported at crate root: `atom::*`, `bond::*`, `comm::*`, `domain::*`. Explicitly re-exported: `Group`, `GroupDef`, `GroupPlugin`, `GroupRegistry`, `group_includes`, `Region`, `Axis`, `SurfaceResult`. From `grass_io` (re-exported): `RunPlugin`, `RunConfig`, `RunState`, `RunSchedule`, `StageConfig`, `StageOverrides`, `FirstStageOnlyConfigs`, `set_stage_name`, `run_read_input`, `update_cycle`, `validate_stages`, `deep_merge`, `RUN_NAMESPACE`. Also `schedule::*`, `neighbor::*`, `virial::*`, `toml`.

---

## Config / TOML Schema

Every section is loaded via `Config::load::<T>(app, "key")` and validated with `#[serde(deny_unknown_fields)]`. Unknown keys are hard errors.

### `[comm]`
```toml
[comm]
processors_x = 1   # MPI ranks in x (i32, default 1)
processors_y = 1   # MPI ranks in y (i32, default 1)
processors_z = 1   # MPI ranks in z (i32, default 1)
```
Product must equal total MPI rank count. First-stage-only.

### `[domain]`
```toml
[domain]
x_low = 0.0          # (f64, default 0.0; alias: x_lo)
x_high = 1.0         # (f64, default 1.0; alias: x_hi)
y_low = 0.0
y_high = 1.0
z_low = 0.0
z_high = 1.0
boundary_x = "periodic"   # "periodic" | "fixed" | "shrink-wrap"
boundary_y = "periodic"
boundary_z = "periodic"
shrink_wrap_padding = 0.0  # (f64, default 0.0 → auto: uses ghost_cutoff as padding)
```
All values optional (defaults shown). MPI + shrink-wrap panics at startup (`domain.rs:366`). First-stage-only: later stages can deform the box but don't re-read `[domain]` (`domain.rs:333`).

### `[neighbor]`
```toml
[neighbor]
skin_fraction = 1.0    # (f64) pair cutoff multiplier: cutoff = (r_i + r_j) * skin_fraction
bin_size = 1.0         # (f64) minimum bin size; auto-increased if needed for MPI
every = 0              # (usize) rebuild every N steps; 0 = displacement-based only
check = true           # (bool) if every>0, also check displacement each step
sort_every = 1000      # (usize) spatial sort interval for cache locality; 0 = disabled
newton = true          # (bool) true = half list + reverse comm; false = full list
```

### `[[group]]` (array of tables)
```toml
[[group]]
name = "mobile"                                               # (String, required)
type = [1, 2]                                                 # (Vec<u32>, optional) atom_type filter
region = { type = "block", min = [0, 0, 0], max = [5, 5, 5] } # (Region, optional)
dynamic = false                                               # (bool, optional)
                                                              # default: true if region set, false otherwise
```
Built-in `"all"` group always exists; cannot be redefined. Stage overrides can add or replace groups by name.

### `Region` inline TOML formats
```toml
{ type = "block",    min = [x,y,z], max = [x,y,z] }
{ type = "sphere",   center = [x,y,z], radius = r }
{ type = "cylinder", center = [a,b], radius = r, axis = "z", lo = 0.0, hi = 5.0 }
{ type = "cone",     center = [a,b], axis = "z", rad_lo = 0.004, rad_hi = 0.001, lo = 0.0, hi = 0.01 }
{ type = "plane",    point = [x,y,z], normal = [nx,ny,nz] }
{ type = "union",    regions = [...] }
{ type = "intersect",regions = [...] }
```

---

## Key Behaviors, Invariants & Gotchas

### 1. The parallel-array invariant (`lib.rs:17–26`)
Every structural operation — `swap_remove`, `apply_permutation`, `truncate` — must be applied **identically** to the base `Atom` AND every registered `AtomData` extension. Desync = silent indexing bugs. The registry mirrors each op via `*_all`. Any `AtomData` impl must do the same for its own fields; `#[derive(AtomData)]` does it automatically.

**Gotcha**: if you manually resize one Vec in an `AtomData` extension (outside `zero()`) and forget to resize all others identically, indices desync silently. Rule: never resize a column out of band (`SOIL_ATOMDATA_CONTRACT.md:2`).

### 2. Per-step canonical ordering
```
PostInitialIntegration:
  remove_ghost_atoms (FullRebuild only) → zero_all_forces
  → decide_rebuild (CommunicateOnly only)
PreExchange: shrink_wrap → pbc (FullRebuild only)
Exchange: atom migration (FullRebuild only)
PreNeighbor: sort_atoms_by_bin → borders (FullRebuild only)
Neighbor: bin_neighbor_list (FullRebuild only)
PreForce: rebuild_groups → forward_comm
Force: physics
PostForce: reverse_send_force
```
(`atom.rs:614–619`, `neighbor.rs:549–566`, `domain.rs:335–341`, `group.rs:172–174`)

**Gotcha**: `pbc` and `remove_ghost_atoms` are both gated on `FullRebuild`, but `shrink_wrap` runs every step. If you add a pre-exchange system, check whether it needs to be gated on `CommState`.

### 3. Newton flag semantics (`neighbor.rs:65–66`, `neighbor.rs:117`)
`newton = true` (default): half list; each pair appears once with `j > i` or `j` a ghost. Force on ghost `j` must be written — the substrate's reverse comm carries it to `j`'s owner at `PostForce`. `newton = false`: full list; the pair `(i,j)` and `(j,i)` both appear. Force systems must NOT write `atoms.force[j]` for ghost `j` in full-list mode, or they double-count. The tutorial (`write-your-own-physics.md:101–107`) guards this: `if newton || j < nlocal { atoms.force[j] -= ...; }`.

### 4. Never resize AtomData columns out of band (`atom.rs:40–52`, `SOIL_ATOMDATA_CONTRACT.md:2`)
The substrate grows columns only via `unpack` (migration) and `zero` (per-step reset). Any other push/pop on a single column desyncs the parallel arrays. If a new atom is inserted out of band, it must go through a substrate path that touches all columns.

### 5. Ghost overwrite vs. reverse additive (`atom.rs:40–48`)
`unpack_forward` overwrites the ghost's value (called each step at `PreForce`). `unpack_reverse` accumulates (`+=`) into the owner. The force on a ghost is meaningless for anything except reverse comm. Mixing these up is silent divergence in parallel.

### 6. `#[reverse]` must pair with `#[zero]` (`SOIL_ATOMDATA_CONTRACT.md:4`)
A reverse accumulator that is not zeroed each step double-counts across steps. The pattern is `#[reverse] #[zero]`. The only exception: a field that is explicitly reset from within a force system before accumulation starts (rare and fragile).

### 7. `*_comm_size` must match `pack_*` (`atom.rs:49`)
`forward_comm_size()` must return exactly the number of f64s that `pack_forward` pushes per atom. If they differ, ghost buffer striding desyncs and reads garbage. The derive keeps these locked; hand-written impls must too.

### 8. Field declaration order is the wire layout (`atom.rs:51`, `tutorial/field-attributes.md:45`)
Migration buffers and restart files serialize fields in declaration order. Reordering, inserting, or removing a field breaks old restart files and in-flight migration buffers. Treat the field list as a versioned wire format.

### 9. The skin / cutoff / ghost-cutoff chain (`lib.rs:54–65`, `neighbor.rs:604`)
`ghost_cutoff = max_pair_cutoff + 2·displacement_buffer`. If `ghost_cutoff < skinned_pair_cutoff`, pairs are silently missed. `neighbor_setup` computes the ghost cutoff after all atoms are created; must run at `PostSetup`. If `cutoff_radius` is empty at setup (rate-based insertion), ghost cutoff falls back to `bin_size`.

### 10. `decide_rebuild` is an MPI collective (`neighbor.rs:791–816`)
`decide_rebuild` calls `all_reduce_sum_f64` to agree globally on whether to rebuild. If any rank needs a rebuild, all ranks do a `FullRebuild`. This is required for MPI send/recv pattern matching — if ranks take different paths, the collective calls deadlock.

### 11. Spatial sort and ghost sendlists (`neighbor.rs:869–975`)
`sort_atoms_by_bin` only permutes during a `FullRebuild` step. Permuting during `CommunicateOnly` would invalidate the saved ghost sendlists (`swap_data.send_indices`, `recv_start`) without rebuilding them — causing reverse comm to accumulate forces onto the wrong owner atoms (`neighbor.rs:861–879`). After a periodic sort, `all_reduce` ensures all ranks agree and transition to `FullRebuild` together.

### 12. Shrink-wrap is single-process only (`domain.rs:365–368`)
MPI + shrink-wrap panics at startup. MPI support would need `all_reduce_min/max` for global extremes and per-rank subdomain bound updates. Currently no-op with a clear panic message.

### 13. Triclinic box: fractional-coordinate classification (`domain.rs:122–141`)
When `triclinic = true`, atom ownership is determined in fractional (lamda) coordinates where the tilted box maps to the unit cube. PBC wrapping, exchange, and binning all work in lamda space. The tilt `xy` is the only active one today (Lees–Edwards simple shear). Lees–Edwards velocity remap: a y-face crossing subtracts `delta_y · boundary_vel` from the atom's velocity (`domain.rs:559–604`).

### 14. `domain_read_input` and `comm_read_input` are first-stage-only (`domain.rs:333`)
They are guarded by `run_if(first_stage_only())`. Re-running them each stage would clobber box deformation accumulated by `soil_deform`. Per-stage box changes are the job of the deform plugin.

### 15. Group rebuild at `PreForce` not `PostInitialIntegration` (`group.rs:173`)
Dynamic groups (region-based) rebuild at `PreForce`, after ghosts are built but before force computation. This means group masks reflect atom positions at the start of the force phase, not positions after the first half-kick.

### 16. `borders` forward-comm hot path uses persistent buffers (`comm.rs:C7, PERF_COMM_NEIGHBOR.md`)
Per-step forward and reverse comm use `sendrecv_f64_into` with persistent `recv_buff` / `reverse_send_buff`. Full-rebuild `borders` and `exchange` still use the allocating `sendrecv_f64` path (item C7 in `PERF_COMM_NEIGHBOR.md`).

---

## Tutorial Outline

The existing tutorial (`tutorial/write-your-own-physics.md`) covers steps 1–4 for a soft-sphere force. A complete expanded tutorial should cover:

1. **Setting up the substrate** — which plugins to add, in what order; why `AtomPlugin` must come first; the difference between `CorePlugins` (downstream in `dirt_core`) and assembling plugins directly (as in `soil_core/examples/minimal_sim.rs`).
2. **Configuring the TOML** — a minimal `config.toml` with `[domain]`, `[neighbor]`, `[comm]`.
3. **Declaring per-particle state** — `#[derive(AtomData)]`; choosing `#[forward]`, `#[reverse]`, `#[zero]`, or no attribute using the decision tree; the peridynamics worked example (`SOIL_ATOMDATA_CONTRACT.md:5`).
4. **Registering state** — `register_atom_data!(app, ...)` call location and ordering; what panics if you register before `AtomPlugin`.
5. **Writing the force system** — `Res<Atom>`, `Res<Neighbor>`, `Res<AtomDataRegistry>`; `neighbor.pairs(nlocal)`; the `newton` flag guard; why you don't write MPI code.
6. **Scheduling** — `ParticleSimScheduleSet::Force`; using `before`/`after` labels for ordering within a phase; when to use `PreForce` vs. `Force` vs. `PostForce`.
7. **Groups and regions** — filtering loops with `group_includes(mask, i)`; getting a mask from `GroupRegistry::mask_for`.
8. **Multi-stage simulations** — `[[stage]]` config; how group definitions carry across stages; `first_stage_only()`.
9. **MPI checklist** — Newton flag, `decide_rebuild` collective, no per-atom MPI calls in force systems.
10. **Debugging divergence** — how to detect misclassified fields; what `forward = overwrite` and `reverse = additive` mean in practice; restart file format breaks.

---

## Doc Gaps (vs. existing chapters)

The existing docs are strong on the `AtomData` contract and the tutorial. Gaps:

| gap | status | location |
|---|---|---|
| Domain decomposition details (how the box is split, sub_domain bounds) | "still to document" | `reference/internals.md:89` |
| Ghost / halo exchange in detail (border replication, the forward/reverse passes) | "still to document" | `reference/internals.md:92` |
| Atom migration details (how an atom moves across ranks, what data moves) | "still to document" | `reference/internals.md:95` |
| Neighbor list details (binning, the skin, `pairs()`, `newton` flag) | "still to document" | `reference/internals.md:98` |
| Config/TOML reference (every `[section]` and key) | not documented | — |
| `VirialStress` usage and sign convention | not documented | — |
| `BondStore` / `BondPlugin` and pair exclusions | not documented | — |
| Region primitives reference (all types, TOML syntax) | not documented | — |
| Groups reference (`dynamic` default, stage overrides, `mask_for` API) | not documented | — |
| MPI setup and `[comm]` config | not documented | — |
| Triclinic / Lees–Edwards shear integration with substrate | not documented (soil_deform chapter exists but substrate side not covered) | — |
| `CommState` two-path timestep internals | covered in internals.md but thin | — |
| Performance notes and open optimizations | in `PERF_COMM_NEIGHBOR.md` (not in mdBook) | — |
| `push_test_atom` and testing patterns | not documented | — |
| `ScheduleSetupSet` (Setup / PostSetup) vs. `ParticleSimScheduleSet` | not documented | — |
| `verlet_schedule_warnings` and schedule validation | not documented | — |

---

## Suggested Placement (mdBook structure)

```
# The Substrate
  - What SOIL Owns                  (exists — good overview; add VirialStress, BondStore)
  - The AtomData Contract           (exists — strong)
  - Writing an AtomData Extension   (exists)
  - Time Integration                (exists — soil_verlet)
  - Fixes                           (exists — soil_fixes)
  - Box Deformation                 (exists — soil_deform)
+ - Domain & Decomposition          NEW — DomainConfig, BoundaryType, decompose_domain,
                                          Triclinic/Lees-Edwards, shrink-wrap limits
+ - Communication & Ghost Exchange  NEW — CommConfig, CommState, borders/exchange,
                                          forward/reverse comm, SwapData lifecycle
+ - Neighbor Lists                  NEW — NeighborConfig, Neighbor::pairs, newton flag,
                                          skin/ghost-cutoff chain, bin grid, displacement check
+ - Regions                         NEW — all Region variants, TOML syntax, SurfaceResult
+ - Groups                          NEW — GroupDef, dynamic, mask_for, stage overrides
+ - Config Reference                NEW — all TOML sections/keys in one place

# Tutorial
  - Write Your Own Particle Physics  (exists — good)
  - Choosing Field Attributes        (exists — good)
+ - MPI Checklist for Physics Tiers  NEW — newton guard, decide_rebuild, no per-atom MPI

# Reference
  - Substrate Internals              (exists — fill the "still to document" gaps)
  - Crate Map                        (exists)
+ - Performance & Optimization Notes NEW — incorporate PERF_COMM_NEIGHBOR.md into book
```

The biggest single gap is a **Config Reference** chapter listing every TOML section and key with types and defaults — users need this for every new simulation setup and currently have to read Rust source to find it.
