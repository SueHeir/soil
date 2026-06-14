# soil_deform

Continuous simulation-box deformation for SOIL — analogous to LAMMPS `fix deform`.

## What it does

Modifies the simulation domain boundaries during a run, with independent per-axis control. Each axis (`x`, `y`, `z`) can use one of three styles, and atom positions are remapped affinely so particles track the deforming box. Useful for triaxial compression, oedometer tests, and other geomechanics setups.

The three styles:

- **`erate`** — engineering strain rate: `L(t) = L0 * (1 + rate * dt * step)`
- **`vel`** — constant velocity on box faces: `L(t) = L0 + velocity * dt * step`
- **`final`** — linear ramp to target bounds over the stage duration

Per-stage `[deform]` overrides in `[[run]]` blocks are re-read at the start of each stage, so axes that aren't configured for a stage are cleared.

## Key types

| Item | Role |
| --- | --- |
| `DeformPlugin` | Registers `DeformState` plus the setup and update systems |
| `DeformConfig` | Top-level `[deform]` TOML section (`x`/`y`/`z` axes + `remap` flag) |
| `AxisDeformDef` | Per-axis TOML definition (`style` + parameters) |
| `DeformStyle` | Runtime style enum: `Erate`, `Vel`, `Final` |
| `DeformState` | Runtime resource tracking per-axis deformation and step count |

The setup system runs at `ScheduleSetupSet::PostSetup`; the update system runs at `ParticleSimScheduleSet::PreInitialIntegration`, applying deformation before the Verlet position update and triggering a neighbor-list rebuild.

## Configuration

```toml
[deform]
# Uniaxial compression on z-axis (engineering strain rate; negative = compression)
z = { style = "erate", rate = -0.001 }

# Constant velocity on x-axis faces (optional)
# x = { style = "vel", velocity = 0.01 }

# Linear ramp y-axis to target bounds over the run (optional)
# y = { style = "final", lo = 0.0, hi = 0.02 }

# Remap atom positions affinely when the box changes (default: true)
remap = true
```

## Usage

```rust
use soil_deform::DeformPlugin;

app.add_plugin(DeformPlugin);
```

## License

MIT OR Apache-2.0
