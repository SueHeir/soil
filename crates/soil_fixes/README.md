# soil_fixes

Method-agnostic position constraints for [SOIL](https://github.com/SueHeir/soil).

These fixes constrain atom *kinematics* using only the base `Atom` state
(position, velocity, force), with no knowledge of any particular particle
method. DEM-specific fixes that touch rotational state live in DIRT's
`dirt_fixes` crate.

## Fixes

| Fix | TOML key | Description |
|-----|----------|-------------|
| `PinDef` | `[[pin]]` | Hard **translational** position constraint — captures each atom's setup-time position and restores it bit-for-bit every step. |

```toml
[[pin]]
group = "anchor"
```

## `pin` vs `freeze`

- **`pin`** (this crate) is a *positional* constraint: it restores the captured
  position every step, correcting any drift, and touches only translational
  state — so it works for any particle method.
- **`freeze`** (DIRT `dirt_fixes`) is full immobilization: it zeros velocity,
  force, and — for DEM atoms — angular velocity and torque, freezing rotation
  as well.

## Plugin

`SoilFixesPlugin` registers the `[[pin]]` constraint (only when at least one is
configured). It is **not** in any default plugin group (`dirt_core`'s
`CorePlugins` or `dirt_granular`'s `GranularDefaultPlugins`) — add it explicitly:

```rust,ignore
app.add_plugins(CorePlugins)
   .add_plugins(SoilFixesPlugin);
```

## Scheduling

`pin` is the minimal "how to write a fix" template — a fix is a plugin that
registers systems into the per-step Verlet loop
(`soil_core::ParticleSimScheduleSet`):

```text
PreInitialIntegration → InitialIntegration → … → Force → PostForce → … → FinalIntegration
```

`pin` hooks two passes: **`PreInitialIntegration`** (restore position / zero
velocity before the Verlet drift, so the integrator can't move a pinned atom)
and **`PostForce`** (re-enforce after forces, and lazily capture the initial
position on the first step the group mask is populated). Hooking both is why a
pinned atom is bit-for-bit fixed whenever forces are evaluated.
