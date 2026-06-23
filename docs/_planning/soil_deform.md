# Planning: `soil_deform` documentation

Sources read: `crates/soil_deform/src/lib.rs`, `crates/soil_deform/Cargo.toml`,
`crates/soil_deform/README.md`, `docs/src/substrate/deformation.md`,
`docs/src/substrate/what-soil-owns.md`, `docs/src/reference/internals.md`,
`docs/src/SUMMARY.md`, `dirt/examples/bench_lebc_shear/` (README, main.rs,
config.toml, sweep.py).

---

## Purpose

`soil_deform` is the substrate-tier box-deformation driver, analogous to LAMMPS
`fix deform`. It modifies simulation-domain boundaries continuously during a run
and affinely remaps atom positions so particles track the deforming box. It
supports three independent per-axis styles (engineering strain rate, constant
velocity, linear ramp to target) and a fourth mode that drives the box into a
triclinic (tilted) state for Lees–Edwards (LEBC) simple shear. It is the
enabling primitive for triaxial/oedometer compression tests and for the
homogeneous-shear rheometer (`bench_lebc_shear`) that calibrates constitutive
models for MUD/SPH de-fluidization.

---

## Public surface to document

| Item | Kind | File:line |
|---|---|---|
| `DeformPlugin` | `Plugin` impl — the entry point | `lib.rs:235` |
| `DeformConfig` | top-level `[deform]` TOML struct (serde) | `lib.rs:142` |
| `AxisDeformDef` | per-axis TOML struct | `lib.rs:110` |
| `DeformStyle` | runtime enum (`Erate`, `Vel`, `Final`) | `lib.rs:171` |
| `DeformState` | runtime resource (`axes`, `shear_xy_rate`, `shear_initialized`, `remap`, `step`, `initialized`) | `lib.rs:195` |
| `DeformState::has_any()` | convenience predicate | `lib.rs:215` |
| `setup_deform` | setup system (private, but behavior is public contract) | `lib.rs:346` |
| `apply_deform` | update system (private, but behavior is public contract) | `lib.rs:435` |

The crate's prelude exports nothing — consumers call `app.add_plugin(DeformPlugin)`.
The only public surface a DIRT-tier user touches is the TOML config and the
plugin registration call.

---

## Config / TOML schema

Top-level key: `[deform]`

```toml
[deform]
x   = { style = "erate",  rate = -0.001 }          # engineering strain rate
y   = { style = "vel",    velocity = 0.01 }         # constant velocity on faces
z   = { style = "final",  lo = 0.0, hi = 0.02 }    # ramp to target bounds
xy  = { style = "erate",  rate = 50.0 }             # Lees–Edwards shear-strain rate γ̇
remap = true                                         # affine atom remap (default true)
```

### Per-axis keys (`x`, `y`, `z`, `xy`) — type `AxisDeformDef` (`lib.rs:110`)

| Key | Type | Required by | Meaning |
|---|---|---|---|
| `style` | `String` | always | `"erate"`, `"vel"`, or `"final"` (axes); `"erate"` only (xy) |
| `rate` | `f64` | `erate` | engineering strain rate (axis) or shear-strain rate γ̇ (xy), units 1/s |
| `velocity` | `f64` | `vel` | face-separation velocity (positive = expansion), same units as length/time |
| `lo` | `f64` | `final` | target lower bound |
| `hi` | `f64` | `final` | target upper bound |

Struct uses `#[serde(deny_unknown_fields)]` — typos are caught at config parse time.
Missing required fields for the selected style cause a panic at setup (`lib.rs:288–334`).

### `remap` — type `bool`, default `true` (`lib.rs:163`)

When true, atom positions are scaled affinely with the box each step. When false,
box walls move but particles stay put (useful for wall-only tests or re-equilibration
after a hard box insertion).

### Deformation styles

**`erate`** (`lib.rs:484–491`):
`L(t) = L0 * (1 + rate * dt * step)`.
Linear in step count (not a true exponential). Compression = negative rate.
Center-symmetric: both faces move; box center is invariant.

**`vel`** (`lib.rs:492–498`):
`L(t) = L0 + velocity * dt * step`.
Each face moves at `velocity/2`; also center-symmetric.

**`final`** (`lib.rs:499–513`):
Linear interpolation from initial bounds to `(lo, hi)` over the stage's total
steps. Uses `RunState::cycle_count + cycle_remaining` to determine `total_steps`
(`lib.rs:455–464`). Clamps at `frac = 1.0` so it holds the target after the stage
ends rather than overshooting.

**`xy` — Lees–Edwards shear** (`lib.rs:559–586`):
`xy(t) = γ̇ · Ly · t`, box-flip-wrapped into `[−Lx/2, Lx/2]`.
Sets `domain.tilt[0]`, `domain.boundary_vel[0] = γ̇·Ly`, `domain.triclinic = true`.
Only `"erate"` is accepted; other styles panic (`lib.rs:334`).

---

## Key behaviors, invariants, and gotchas

### 1. Center-symmetric resizing (`lib.rs:484–498`)

`erate` and `vel` keep the box center at `(lo_0 + hi_0) / 2`. There is no
single-platen (one-sided compression) mode. The docstring explicitly states this
and directs users to a `[[pin]]` group + physics-tier wall if one-sided loading is
needed (`lib.rs:16–21`).

### 2. Box-flip wrap for LE tilt (`lib.rs:578–579`)

```rust
let raw = rate * dt * step as f64 * ly;
let xy = raw - lx * (raw / lx).round();
```
The tilt is wrapped into `[-Lx/2, Lx/2]` every step. This is an exact relabeling
to an equivalent periodic image — no discontinuity for atoms, but the neighbor
stencil must rebuild every step (which `domain.bounds_changed = true` at `lib.rs:584`
ensures). Failing to rebuild after a tilt wrap is the mode most likely to produce
silently wrong pair lists.

### 3. Streaming-velocity initialization (one-shot, `lib.rs:566–572`)

On the **first** shear step, `v_x += γ̇·(y − y_center)` is imposed across all
local atoms (`shear_initialized` flag, `lib.rs:204`). This gives the system a
linear Couette profile as initial condition, reducing transient settling. It runs
exactly once per stage (flag resets in `setup_deform` at `lib.rs:362`). If the
stage restarts (multi-stage), the profile is re-imposed from the current box
geometry, not the original geometry — this is correct because `y_center` is
computed live from `domain.boundaries_low[1]`.

### 4. Periodic-boundary requirement for xy shear (`lib.rs:366–371`)

`setup_deform` panics if `domain.is_periodic(0)` or `domain.is_periodic(1)` is
false when `shear_xy_rate` is set. The check happens at setup, not at apply time,
so a misconfigured boundary is caught before the first step.

### 5. Stage boundary / time-origin semantics (`lib.rs:346–420`)

`setup_deform` runs at `ScheduleSetupSet::PostSetup` (re-runs every `[[run]]`
stage). At each stage:
- axis `lo_0`/`hi_0` are snapshotted from the current domain bounds (`lib.rs:374–378`);
- `step` resets to 0 (`lib.rs:381`);
- `shear_initialized` resets to `false` (`lib.rs:362`);
- a stage with **no** `[deform]` section produces a default-empty `DeformConfig`,
  clearing all axes (`lib.rs:356–361`).

This means `erate` strain is always measured from the start of the *stage*, not
the start of the run — important for multi-stage protocols (e.g., compress then
shear, as in `bench_lebc_shear`).

### 6. Mandatory neighbor rebuild after every box change (`lib.rs:588–602`)

```rust
domain.bounds_changed = true;
neighbor.last_build_pos.clear();
```
Both flags are set whenever `box_changed`. The README documents the failure mode:
a dilute expanding box with a stale bin grid can index past `bin_total_cells` in
the unchecked neighbor build, causing a SIGSEGV. The rebuild is not optional.

### 7. Single-processor only (`lib.rs:548–553`, README)

Sub-domain bounds are updated identically to global bounds (valid only for
single-rank runs). Full MPI would require re-decomposition after each box change.
The README documents this limitation explicitly.

### 8. Collapse guard (`lib.rs:519–527`)

If any axis deforms to `new_size <= 0.0`, the system panics with a diagnostic
message. There is no graceful recovery — excessive compression rates must be
caught in the input.

### 9. `erate` is linear, not exponential (`lib.rs:484–487`, README)

`L(t) = L0·(1 + rate·dt·step)` is a first-order Taylor approximation to
`L0·exp(rate·t)`. For small rates and short runs the difference is negligible,
but for large `|rate|·T` (e.g., rapid compression) the true volume ratio diverges
from the engineering formula. Users targeting a specific volumetric strain should
verify the accumulated error.

### 10. `domain.boundary_vel` carries the LE velocity jump (`lib.rs:581`)

`domain.boundary_vel[0] = γ̇·Ly` is set every step (it is constant for a fixed
box but is recomputed because `Ly` changes if `y` is also deforming). Ghost/comm
layers that apply the streaming-velocity correction when atoms cross the y-face
read this field. The documentation should make this linkage explicit for anyone
writing a custom ghost-comm layer.

---

## Tutorial outline — run a shear rheometer

Audience: a physicist-tier developer who wants to add LEBC rheometry to a new
DIRT-style physics crate.

1. **Add the plugin**: `app.add_plugin(DeformPlugin)` alongside `CorePlugins`.
2. **Write a two-stage config**:
   - Stage 1 (settle/compress): `[deform]` with `z = { style = "vel", velocity = -V }`.
     No `xy` key → no shear. Particles pack under isotropic compression.
   - Stage 2 (shear): `[deform]` with `xy = { style = "erate", rate = γ̇ }`. No
     axis keys → box volume is fixed. Gravity must be off (`gz = 0.0`).
3. **Periodicity**: set all three boundaries periodic. Confirm `domain.is_periodic(0)
   && domain.is_periodic(1)` or the setup panics.
4. **Record stress**: at `PostFinalIntegration`, compute the virial stress tensor
   from contact forces. `σ_xy` is the shear stress; `p = ⅓ tr σ` is the pressure.
   Subtract the linear streaming profile `v̄_x(y) = γ̇·(y − y_c)` before computing
   the granular temperature `T = ⅓⟨|δv|²⟩`.
5. **Wait for steady state**: strain units `γ = γ̇·t > 10` are typically needed.
   Use the settle stage to reach target `Φ`, then the shear stage to drive to
   steady state before averaging.
6. **Reference**: `dirt/examples/bench_lebc_shear/` is the canonical implementation.
   It uses `config.toml` stages (relax → compress → shear), `main.rs` for the
   stress recorder, and `sweep.py` for parametric sweeps over `Φ` and `γ̇`.

---

## Doc gaps

| Gap | Severity | Notes |
|---|---|---|
| `domain.boundary_vel` linkage to ghost comm | High | Docs say ghosts "pick up the streaming-velocity jump" but never show how — which field carries it, which comm pass reads it, what happens if it's stale |
| `domain.tilt[0]` semantics | Medium | The tilt field is set but never documented in the substrate internals chapter; a reader implementing a custom comm layer won't know how to use it |
| Multi-axis + shear interaction | Medium | Can `z = { style = "erate" }` and `xy = { style = "erate" }` coexist? (Yes, but undocumented; `box_changed` is ORed from both code paths) |
| MPI limitation | Medium | Single-proc-only is noted in README but not in `substrate/deformation.md`; future multi-rank users will be surprised |
| `erate` vs true exponential strain | Low | The linearization is noted in README but not in the mdBook chapter |
| `final` style total-steps lookup | Low | `RunState::cycle_count + cycle_remaining` trick is fragile if the stage index is out of bounds; the fallback to `1` is silent |
| `remap = false` use cases | Low | Not illustrated anywhere; the `no_remap_leaves_atoms_in_place` test covers correctness but no prose explains when you'd want this |

---

## Suggested placement

**Primary**: `docs/src/substrate/deformation.md` — already exists and already
has the right structure (mirroring the crate-level doc comment). This file
should be extended, not replaced. Specific additions:

- A **MPI / single-proc caveat** box after the intro paragraph.
- A **`domain.boundary_vel` and `domain.tilt`** sub-section under "Lees–Edwards
  xy simple shear" explaining which domain fields carry the velocity jump and tilt
  for the ghost-comm layer.
- A **multi-axis + shear** paragraph clarifying that orthogonal `x/y/z` deform
  styles can coexist with `xy` shear and both are applied in the same `apply_deform`
  call (`lib.rs:468–586`).
- A **`erate` linearization** note for users targeting specific volumetric strains.

**Cross-reference**: `docs/src/reference/internals.md` → add a note under
"The two-path timestep" that deform sets `domain.bounds_changed` and `neighbor.last_build_pos.clear()`
to force the `FullRebuild` path, so readers understand why the list always
rebuilds after a deformation step.

**Tutorial hook**: `docs/src/tutorial/write-your-own-physics.md` could link to
the shear-rheometer tutorial outline as the canonical "add a measurement and a
driver" example — it combines `DeformPlugin` (driver) with a custom
`PostFinalIntegration` system (measurement).
