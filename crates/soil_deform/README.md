# soil_deform

Continuous simulation-box deformation for SOIL — analogous to LAMMPS `fix deform`.

## What it does

Modifies the simulation domain boundaries during a run, with independent per-axis control. Each axis (`x`, `y`, `z`) can use one of three styles, and atom positions are remapped affinely so particles track the deforming box. The box can also be driven into a triclinic (tilted) state for Lees–Edwards simple shear. Useful for triaxial/oedometer compression, simple-shear rheology, and other geomechanics setups.

The per-axis styles:

- **`erate`** — engineering strain rate: `L(t) = L0 * (1 + rate * dt * step)`
- **`vel`** — constant velocity on box faces: `L(t) = L0 + velocity * dt * step`
- **`final`** — linear ramp to target bounds over the stage duration

Plus a shear style on the `xy` key:

- **`xy = { style = "erate", rate = γ̇ }`** — Lees–Edwards simple shear. `rate` is a **shear-strain rate** γ̇ (units 1/s), not a length rate; the tilt grows as `xy(t) = γ̇·L_y·t` and drives a triclinic box. **Requires periodic x and y** (panics otherwise).

`erate`/`vel` resize the box **center-symmetrically** — both faces move so the center is fixed; this is not a one-sided platen.

Per-stage `[deform]` overrides in `[[run]]` blocks are re-read at the start of each stage: bounds are re-snapshotted as the new strain origin, the step counter resets, and a stage with no `[deform]` section clears all axes (and the shear), so deformation does not persist across stages.

> **Single-proc only.** Deformation updates the global box and remaps all local atoms directly; there is no multi-rank re-decomposition. Every box change forces a neighbor rebuild (`bounds_changed`), which is mandatory — a stale bin grid on a dilute expanding box can index past the grid and SIGSEGV in the unchecked neighbor build. Note `erate` is linear in `step` (`1 + rate·dt·step`), not a true exponential.

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
