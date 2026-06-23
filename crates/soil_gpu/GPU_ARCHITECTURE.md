# GPU Architecture — generic substrate (soil) vs physics (dirt)

Status: **design for review.**

## Principle

The GPU layer mirrors the existing CPU split:
- **`soil_gpu` = generic GPU substrate.** Anything a *different* particle method (MD,
  peridynamics, SPH, …) would also need lives here: the device, the resident state
  + coherence, integration, neighbor/cell-list build, and the persistent
  per-neighbor-state primitive. No DEM knowledge.
- **`dirt_gpu` = DEM physics only.** Hertz/Mindlin contact force, tangential spring
  *semantics*, particle rotation, DEM wall response. Built *on top of* `soil_gpu`.

If MD or peridynamics wanted GPU, they'd depend on `soil_gpu` and plug in their own
force kernel — exactly as `dirt_gpu` does. This is the test for "does it belong in
soil": *would a non-DEM method reuse it?* If yes → soil.

## The boundary (what goes where)

| Capability | Crate | Why |
|---|---|---|
| `GpuContext` (wgpu device/queue, limits) | **soil_gpu** | generic |
| Device-resident buffers for `Atom` core fields (pos, vel, force, mass, inv_mass, cutoff_radius) | **soil_gpu** | these are soil::Atom fields |
| **Coherence layer** (`DualBuffer` per field: Host/Device/Both + dirty flags; `ensure_host`/`ensure_device`) | **soil_gpu** | generic; any method needs CPU↔GPU sync |
| GPU velocity-Verlet integration (pos/vel/force) | **soil_gpu** | MD uses VV too |
| GPU **cell-list / neighbor build** (positions + cutoff → sorted cell list) | **soil_gpu** | every particle method needs neighbors |
| Generic body force (gravity) | **soil_gpu** | generic |
| **Persistent per-neighbor-state slots** (per-atom fixed capacity, ping-pong, index-keyed, survives rebuilds) | **soil_gpu** | DEM contact springs **and** peridynamics bonds **and** stateful MD pairs all need this |
| Planar **boundary geometry** + signed-distance detection | **soil_gpu** | walls/boundaries are generic |
| Resident-step **framework** + extension API (register device kernels into phases) | **soil_gpu** | the orchestration is generic |
| **Hertz/Mindlin contact force** kernel | **dirt_gpu** | DEM constitutive law |
| Tangential **spring** update (rotate/integrate/Coulomb cap) using the soil slot primitive | **dirt_gpu** | DEM semantics |
| Particle **rotation** (omega/torque/inv_inertia integration; torque from tangential lever) | **dirt_gpu** | DEM rigid-particle (not MD/peri); these are DemAtom fields |
| DEM **wall force response** (Hertz + friction) on soil's boundary geometry | **dirt_gpu** | DEM response |
| `GpuCorePlugins` (substrate plugins) | **soil_gpu** | mirrors `CorePlugins` |
| `GpuGranularPlugins` (DEM force plugins) | **dirt_gpu** | mirrors `GranularDefaultPlugins` |

This is a refactor of today's code: the cell-list build, integration, resident loop,
and the slot primitive currently sit in `dirt_gpu` and **move to `soil_gpu`**; the
Hertz/Mindlin force, rotation, contact-spring semantics, and wall response **stay in
`dirt_gpu`**.

## Coherence layer (soil_gpu)

Per dynamic field (pos, vel, force, omega*, torque*) a `DualBuffer` tracks where the
fresh copy is (Host / Device / Both) with `host_dirty`/`device_dirty` bits — the
Kokkos `DualView` model.
- `ensure_device(field)`: upload iff host-dirty; no-op otherwise. Mark device-fresh.
- `ensure_host(field)`: download iff device-dirty; no-op otherwise. Mark host-fresh.
- writing a side sets the other side dirty.

*(omega/torque live in dirt; soil's coherence primitive is generic over "a field",
so dirt registers its own DualBuffers for omega/torque using soil's primitive.)*

Statics (radius, inv_mass, inv_inertia) upload once. Device-only structures (cell
list, contact-spring slots) never sync — no host equivalent, and CPU fixes don't
touch them.

**Result:** an all-device per-step chain trips zero syncs (resident, ~15×). A CPU
fix that reads `vel` syncs only `vel`, only at that boundary; output reading only
`pos` syncs only `pos`. Same surface API; all-GPU is simply faster. v1 wires the
guards by convention (systems call `ensure_*`); v2 can auto-insert from declared
`{residency, reads, writes}` if the grass scheduler can carry the metadata.

## Resident-step framework + extension API (the crux of generality)

`soil_gpu` owns the resident loop and the device buffers; `dirt_gpu` plugs its force
kernel in. The framework exposes device **phases** — `IntegrateInitial`, `Neighbor`,
`Force`, `IntegrateFinal` — and lets a physics crate register a compute dispatch into
`Force`:

- soil_gpu exposes the resident buffers (pos/vel/force/cutoff_radius + cell-list
  outputs: cell_start, sorted_atoms, atom_cell) as handles (`wgpu::Buffer` or a typed
  accessor) so a downstream kernel can bind them.
- A physics crate builds its own pipeline + bind group referencing **soil's buffers
  + its own** (e.g. dirt: radius, omega, torque, contact slots) and registers a
  `GpuForce` hook: `fn record(&self, pass: &mut ComputePass)`.
- soil's resident loop calls registered Force hooks each step between Neighbor and
  IntegrateFinal, and updates coherence (force → device-dirty).

So MD plugs an LJ kernel into `Force`; peridynamics a bond kernel; dirt the
Hertz/Mindlin kernel. The loop, neighbors, integration, coherence are shared.

## Plugin layering

- **soil_gpu**: `GpuCorePlugins` — registers `GpuState` (buffers + coherence),
  GPU integration, GPU neighbor build, gravity, and the resident loop driver.
- **dirt_gpu**: `GpuGranularPlugins` — registers the Hertz/Mindlin force hook +
  rotation + wall response, on top of `GpuCorePlugins`.
- Existing CPU fixes/plugins are unchanged; they read host `Atom` and trip
  `ensure_host` for the fields they touch (v1: a thin wrapper or explicit call).

## Reusability check (the point of the split)
- **MD**: soil gives device atoms, VV integrate, cell-list neighbors, coherence,
  gravity; MD adds a pair-potential Force hook. The persistent-slot primitive serves
  stateful potentials.
- **Peridynamics**: soil's persistent per-neighbor-state slots == bond state; soil's
  neighbor build == the bond family; peri adds a bond-force hook. Boundary geometry
  reused.
- **DEM (dirt)**: contact springs use the slot primitive; rotation + Hertz/Mindlin +
  wall response are the only DEM-specific additions.

## Refactor + incremental plan
1. **Promote to soil_gpu** (from dirt_gpu): GpuContext (already there), cell-list
   build, VV integration kernels, the resident loop, the persistent-slot primitive,
   boundary geometry, the coherence `DualBuffer`. Define the Force-hook extension API
   + buffer accessors.
2. **Reduce dirt_gpu** to: the Hertz/Mindlin + tangential-spring kernel (using soil's
   slots), rotation, wall response — registered via the Force hook + GpuGranularPlugins.
3. **GpuCorePlugins / GpuGranularPlugins**; CPU fixes trip `ensure_host`.
4. **Prove it**: all-GPU step → zero syncs (~15×); add one CPU fix reading vel →
   only vel syncs, result still matches CPU; existing dirt validation (vs real
   hertz_mindlin) still passes through the new layering.

## Notes / risks
- **Cross-repo dev**: soil_gpu lives in soil, dirt depends on it via the git `dev/gpu`
  branch — during this refactor use a local path patch (or frequent pushes) so dirt
  builds against the in-progress soil_gpu.
- **Buffer sharing across crates** is the main API design effort (exposing soil's
  `wgpu::Buffer`s safely + a stable Force-hook signature). Keep the exposed surface
  minimal and typed.
- **MPI later**: the coherence layer is also where a ghost/halo exchange slots in — a
  halo is just another sync boundary (per-rank device + border sync). Designing
  coherence now keeps multi-GPU/MPI a natural extension, not a rewrite.
- This is the biggest structural change yet; build incrementally so there's always a
  working resident path.
