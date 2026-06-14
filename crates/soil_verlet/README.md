# soil_verlet

Translational velocity Verlet time integration for SOIL particle simulations.

## What it does

Implements the **velocity Verlet** algorithm — a symplectic, time-reversible,
second-order "kick-drift-kick" integrator for Newton's equations of motion. The
scheme is split into two systems that bracket the per-step force calculation:

**Initial integration** (before forces):
```text
v(t + Δt/2) = v(t)     + (Δt / 2m) · F(t)      // half-step velocity kick
x(t + Δt)   = x(t)     + Δt · v(t + Δt/2)       // full-step position drift
```

**Final integration** (after forces):
```text
v(t + Δt)   = v(t + Δt/2) + (Δt / 2m) · F(t + Δt) // completing velocity kick
```

This decomposition is second-order accurate in Δt, exactly integrates
constant-force motion, and keeps energy drift bounded for Hamiltonian systems.
The systems operate on the `Atom` resource from `soil_core`, updating only the
`nlocal` owned particles each step.

## Key types

| Item | Role |
| --- | --- |
| `VelocityVerletPlugin` | Registers both integration systems. `new()` runs in every stage; `for_stage(name)` restricts them to a single `[[run]]` stage. |
| `initial_integration` | Half-step velocity kick + position drift, scheduled at `ParticleSimScheduleSet::InitialIntegration`. |
| `final_integration` | Completing velocity kick, scheduled at `ParticleSimScheduleSet::FinalIntegration`. |

## Usage

```rust,ignore
use soil_verlet::VelocityVerletPlugin;

// All stages (default)
app.add_plugins(VelocityVerletPlugin::new());

// Restrict to one named [[run]] stage
app.add_plugins(VelocityVerletPlugin::for_stage("relaxation"));
```

## License

MIT OR Apache-2.0
