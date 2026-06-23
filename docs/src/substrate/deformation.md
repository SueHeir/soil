# Box Deformation

`soil_deform` continuously modifies the simulation box boundaries during a run,
analogous to LAMMPS `fix deform`. It supports per-axis independent control with
three styles, plus Lees–Edwards `xy` simple shear:

- **erate** — engineering strain rate: `L(t) = L0 * (1 + rate * dt * step)`.
- **vel** — constant velocity on box faces: `L(t) = L0 + velocity * dt * step`.
- **final** — linear ramp to a target value over the stage.
- **xy** — Lees–Edwards simple shear (see below).

The three orthogonal axis styles and the `xy` shear are independent: a stage can
deform `z` and shear `xy` at once, and both are applied in the same per-step
call. `erate` and `vel` take their required parameters (`rate`, `velocity`);
`final` takes `lo`/`hi`. A missing required field for the chosen style panics at
setup, and `#[serde(deny_unknown_fields)]` rejects typos at parse time.

Note that **`erate` is linear in step, not exponential**:
`L(t) = L0·(1 + rate·dt·step)` is the first-order approximation to
`L0·exp(rate·t)`. For small `|rate|·T` the difference is negligible, but if you
are targeting a specific volumetric strain under a large rate or a long run,
check the accumulated error against the true exponential.

Atom positions are remapped proportionally (affine transformation) when the box
changes, ensuring particles stay inside the domain.

> **Single-processor only.** `soil_deform` updates this rank's subdomain bounds
> identically to the global box, which is correct only on a single rank. There is
> no re-decomposition after a box change, so a deforming run under MPI will
> mis-assign atoms. Full multi-rank support would need re-decomposition each time
> the box changes; until then, run deformation on one process.

## Center-symmetric, not one-sided

For `erate` and `vel`, the box is resized **symmetrically about its center**:
*both* faces move so the box center stays put. There is no single-platen mode
here — `vel = v` moves each face by `v/2·dt·step`, not one face by `v·dt·step`.
Use a `[[pin]]` group (see [Fixes](./fixes.md)) plus a velocity-driven wall in a
physics tier if you need a genuinely one-sided platen.

## Configuration

```toml
[deform]
# Constant engineering strain rate on z-axis (compression = negative)
z = { style = "erate", rate = -0.001 }

# Constant velocity on x-axis faces
# x = { style = "vel", velocity = 0.01 }

# Ramp y-axis to target bounds over the run
# y = { style = "final", lo = 0.0, hi = 0.02 }

# Lees–Edwards simple shear: tilt x (flow) as a function of y (gradient)
# xy = { style = "erate", rate = 0.5 }   # shear-strain rate γ̇ (units 1/s)

# Remap atom positions (default: true)
remap = true
```

## Lees–Edwards `xy` simple shear

`xy = { style = "erate", rate = γ̇ }` drives the box into a **triclinic**
(tilted) state to impose simple shear in the x (flow) direction as a function of
y (gradient). Key facts:

- `rate` is a **shear-strain rate** γ̇ (units 1/s), *not* a length rate. The tilt
  grows as `xy(t) = γ̇·L_y·t` and the gradient-direction edge moves at
  `Δv = γ̇·L_y`. Ghost copies wrapping the sheared y-face pick up that
  streaming-velocity jump.
- It **requires periodic x and y boundaries** — `setup_deform` panics otherwise.
- Only `style = "erate"` is supported for `xy` (other styles panic).
- On the first shear step a linear streaming profile `v_x = γ̇·(y − y_center)` is
  imposed once so the system starts near steady simple shear.
- The tilt is box-flip-wrapped into `[−Lx/2, Lx/2]` so the ghost/stencil reach
  stays bounded.

> **The box-flip forces a neighbor rebuild every step.** Because the tilt
> advances (and wraps) each step, every shear step sets `domain.bounds_changed`
> and clears `neighbor.last_build_pos`, which drives the `FullRebuild` path. This
> is not optional: a sheared box with a stale bin grid can index past the
> allocated cells in the unchecked neighbor build and segfault. The cost is that
> Lees–Edwards shear never gets the cheap `CommunicateOnly` step — expect a full
> rebuild on every timestep of a shear stage.

### What the shear writes into `Domain`

The shear driver communicates with the substrate's comm layer through three
`Domain` fields, set every step:

- `domain.tilt[0]` — the current `xy` tilt factor, wrapped into `[−Lx/2, Lx/2]`.
- `domain.triclinic = true` — gates the fractional-coordinate (lamda) code paths
  in PBC wrapping, exchange, and binning.
- `domain.boundary_vel[0] = γ̇·Ly` — the streaming-velocity jump across the
  gradient (y) face. When a ghost copy is built across that periodic y-face, the
  comm layer reads this field and adds the jump to the ghost's velocity, so a
  neighbor on the far side sees the correct relative streaming velocity. It is
  recomputed each step because `Ly` changes if `y` is also deforming.

If you write a custom ghost-comm layer for a new physics tier, those are the
fields it must honor to reproduce Lees–Edwards boundary conditions.

## Scheduling

The deform system runs at `ParticleSimScheduleSet::PreInitialIntegration`,
updating domain bounds and remapping atoms before the Verlet position update.
Every box change sets `domain.bounds_changed` and clears the neighbor list's
last-build positions, forcing a neighbor rebuild on the next step.

## Multi-stage / time-origin semantics

The `[deform]` config is **re-read at the start of every `[[run]]` stage** (the
setup system runs at `ScheduleSetupSet::PostSetup`), so per-stage overrides take
effect. At each stage boundary:

- the current domain bounds are snapshotted as `lo_0` / `hi_0` (the deformation
  origin), so strain is measured from the *start of the stage*, not the start of
  the run;
- the elapsed-step counter `step` is reset to 0;
- a stage with **no** `[deform]` section clears all axes and the xy shear, so
  deformation does not silently persist from a previous stage.

This per-stage reset is what makes multi-stage protocols clean: a typical
rheometry run is *relax → compress → shear*, where the compress stage drives the
box to a target solid fraction and the shear stage (volume fixed, no axis keys)
drives `xy` to steady state. Because strain is measured from each stage's start,
the shear stage's `γ = γ̇·t` is the shear strain since shearing began, not since
the run began.

## Where this is used

DIRT's `bench_lebc_shear` example is the canonical consumer: a homogeneous
Lees–Edwards shear rheometer that records the virial stress tensor (`σ_xy` for
shear stress, `p = ⅓ tr σ` for pressure) and the granular temperature, sweeping
over solid fraction Φ and shear rate γ̇. It is the DEM calibration source for the
MUD/SPH constitutive models. If you are adding shear rheometry to a new
physics tier, that example shows the full pattern: `DeformPlugin` as the driver
plus a custom `PostFinalIntegration` system as the measurement.
