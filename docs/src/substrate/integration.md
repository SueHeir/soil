# Time Integration

`soil_verlet` implements **velocity Verlet** for integrating Newton's equations
of motion. The scheme is split into two phases that bracket the force
calculation each timestep.

**Initial integration** (before forces):

```text
v(t + Δt/2) = v(t)     + (Δt / 2m) · F(t)      // half-step velocity kick
x(t + Δt)   = x(t)     + Δt · v(t + Δt/2)       // full-step position drift
```

**Final integration** (after forces):

```text
v(t + Δt)   = v(t + Δt/2) + (Δt / 2m) · F(t + Δt) // completing velocity kick
```

This "kick-drift-kick" decomposition is symplectic, time-reversible, and
second-order accurate in Δt. It exactly integrates constant-force motion and
conserves energy to O(Δt²) per step for Hamiltonian systems.

## Where this runs

The two systems hook the integration phases of `ParticleSimScheduleSet` (the
canonical per-step order is in [Substrate Internals](../reference/internals.md)):

- `initial_integration` runs at `InitialIntegration`, **before** the force
  phase. It uses **last step's** forces for the half-kick, then does the
  position **drift**.
- `final_integration` runs at `FinalIntegration`, **after** the force phase,
  completing the kick with **this step's** freshly computed forces.

This crate does **not** zero the force arrays — that happens in `soil_core` at
`PostInitialIntegration` (between the drift and the force phase), so the forces
`initial_integration` reads are genuinely the previous step's.

Both systems iterate **local atoms only** (`0..nlocal`); ghosts are read-only
halo copies and are never integrated.

`soil_verlet` integrates **translation only**. Quaternion and angular-velocity
integration for non-spherical particles is a physics-tier responsibility (DIRT's
`dirt_atom`), not the substrate's.

## Adding the integrator

You add the integrator as a plugin. The plugin name is stable —
`VelocityVerletPlugin`, with two constructors:

```rust,ignore
use soil_verlet::VelocityVerletPlugin;

app.add_plugins(VelocityVerletPlugin::new());            // runs in every [[run]] stage
app.add_plugins(VelocityVerletPlugin::for_stage("shear")); // only in the named stage
```

`for_stage("name")` wraps both the initial and final half-kicks with
`run_if(in_stage(name))`. Reach for it when stages need different integrators —
a dynamics stage on velocity Verlet and a minimizer stage on something else, for
instance. The plain `new()` constructor runs in all stages, which is what most
single-stage runs want.

There is no `[verlet]` TOML section. The timestep `Δt` is read from `Atom::dt`,
which `soil_core`'s input and run loop set; the integrator never parses config of
its own.

## Force priming on the first step

`initial_integration` runs *before* any force system on every step, including
step 0. On the first step that means it reads whatever is in `Atom.force` before
a force has been computed. If your physics tier accumulates forces fresh each
step (the usual case), this is harmless after step 0. But for a **constant**
force field — gravity, a constant body force — you must prime `Atom.force`
*before* the run loop starts, or the very first half-kick uses zero force and the
trajectory is shifted by half a step. The `free_fall.rs` example in `soil_verlet`
shows the priming pattern.

## Holding a particle fixed

Both systems scale the force by `inv_mass`. A particle with `inv_mass = 0` (i.e.
infinite mass) therefore never accelerates and never drifts — the idiomatic way
to hold a particle fixed under the integrator without a separate constraint.

For a *hard* positional constraint that also corrects stray drift, use the
`[[pin]]` fix instead — see [Fixes](./fixes.md).

## What the integrator does not do

`soil_verlet` applies **no periodic boundary conditions** and tracks no image
flags. The position drift here can push an atom outside the box; wrapping it back
and updating its `image` count is `soil_core`'s job, done in the **Exchange**
phase of the next full-rebuild step (see
[Substrate Internals](../reference/internals.md)). The integrator also builds no
neighbor list and does no communication — it is purely the kick-drift-kick on
local atoms.
