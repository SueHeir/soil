# Fixes

`soil_fixes` provides fixes that constrain atom *kinematics* using only the base
`Atom` state (position, velocity, force) — no knowledge of any particular
particle method. The DEM-specific fixes (which reach into rotational state such
as `omega` / `torque`) live in DIRT's `dirt_fixes` crate instead.

| Fix | TOML key | Description |
|-----|----------|-------------|
| `pin` | `[[pin]]` | Hard **translational** position constraint — captures position at setup, restores every step |

`SoilFixesPlugin` registers the `[[pin]]` constraint. It is **not** part of any
default plugin group (not in `dirt_core`'s `CorePlugins`, nor in
`dirt_granular`'s `GranularDefaultPlugins`). Add it explicitly:

```rust,ignore
use grass_app::prelude::*;
use soil_fixes::SoilFixesPlugin;

let mut app = App::new();
app.add_plugins(CorePlugins)        // input, comm, domain, neighbor, groups, run loop
   .add_plugins(SoilFixesPlugin);   // ← enables [[pin]]
```

with a `[[pin]]` block in the TOML config naming an atom group:

```toml
[[group]]
name = "anchor"
region = "base"     # some region defined elsewhere

[[pin]]
group = "anchor"
```

## The fix pattern, and why `pin` hooks two phases

`pin` is the minimal "how to write a fix" template: a fix is just a plugin that
registers systems into the per-step Verlet loop. The loop's phases (from
`ParticleSimScheduleSet`, see [Substrate Internals](../reference/internals.md))
run in order:

```text
PreInitialIntegration → InitialIntegration → … → Force → PostForce → … → FinalIntegration
```

`pin` hooks **two** of them:

- **`PreInitialIntegration`** — restore position / zero velocity *before* the
  Verlet drift, so the integrator cannot move a pinned atom this step.
- **`PostForce`** — re-enforce *after* forces are computed (and lazily capture
  the initial position on the first step the group mask is populated), so the
  final half-kick sees `f = 0` and the next step starts from `v = 0`.

Hooking both the pre-drift and post-force passes is why a pinned atom is
bit-for-bit fixed whenever forces are evaluated — which matters for bonded models
where a pinned wall atom is a force neighbor of a flexible one.

## The `[[pin]]` block

A `[[pin]]` table has exactly one key. TOML array-of-tables means you can declare
several, one per group.

| key | type | required | meaning |
|---|---|---|---|
| `group` | string | yes | Name of an already-defined atom group. Validated against the `GroupRegistry` at `PostSetup`; a missing or misspelled name is a hard error at startup. |

`#[serde(deny_unknown_fields)]` means any other key is a parse error. The plugin
is opt-in and zero-cost when no `[[pin]]` is configured: with none present it
registers an empty registry and no per-step systems.

## Lazy capture, tags, and restart

The captured position is taken **lazily** — on the first step the group's mask is
non-empty, not at config-parse time. That deliberately lets startup atom
migration (MPI decomposition) settle before the snapshot is frozen. Once a group
is captured, its snapshot is permanent for the run.

The capture map is keyed by **global atom tag**, not local index, so a pinned
position survives MPI rebalancing: an atom's local index can change when it
migrates between ranks, but its tag does not.

Two consequences worth knowing:

- **Dynamic groups.** If a group's membership changes after capture, atoms that
  *join* later are locked at wherever they first appear in a populated mask (not
  the original setup position), and atoms that *leave* simply fall out of the
  mask and stop being pinned. There is no re-capture and no explicit un-pin.
- **Restart.** `PinState` is **not** serialized into restart files. On restart
  the positions are re-captured from the *restarted* atom positions. If a run was
  checkpointed mid-motion, the pinned positions after restart are the
  mid-motion positions, not the original setup ones.

## `pin` vs `freeze`

`pin` is a *positional* constraint: it captures each atom's setup-time position
and **restores** it bit-for-bit every step, so anything that would move the atom
(Verlet drift, a stray force) is corrected. It touches only translational state,
so it works for any particle method.

For full immobilization that *also* freezes rotation, use DIRT's `[[freeze]]`
fix, which zeros velocity, force, and (for DEM atoms) angular velocity and
torque.

(For the *soft* way to hold a particle still — setting `inv_mass = 0` — see
[Time Integration](./integration.md).)
