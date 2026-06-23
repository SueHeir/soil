# Planning: `soil_verlet` documentation

Crate: `soil_verlet`  
Planned placement: `docs/src/substrate/integration.md` (already exists as a stub; this planning file informs its expansion)

---

## Purpose

Implements translational velocity Verlet (kick-drift-kick) for Newton's equations
of motion. Sits between `soil_core` (which owns the `Atom` resource and schedule)
and physics-tier force crates (e.g. DIRT). It is the only integrator in the SOIL
substrate; rotational integration is out of scope here (owned by physics tiers).

---

## Public surface to document

| Item | Kind | Notes |
|---|---|---|
| `VelocityVerletPlugin` | `struct` (plugin) | The name is confirmed correct. Tutorial `write-your-own-physics.md` uses `VelocityVerletPlugin::new()` — matches `src/lib.rs:80`. |
| `VelocityVerletPlugin::new()` | constructor | Runs in every `[[run]]` stage. `src/lib.rs:87`. |
| `VelocityVerletPlugin::for_stage(name: &str)` | constructor | Restricts both systems to a named `[[run]]` stage. `src/lib.rs:92`. |
| `VelocityVerletPlugin::stage: Option<String>` | pub field | The stage filter; `None` = all stages. `src/lib.rs:82`. |
| `initial_integration` | pub system fn | Half-kick + drift, scheduled at `InitialIntegration`. `src/lib.rs:131`. |
| `final_integration` | pub system fn | Completing half-kick, scheduled at `FinalIntegration`. `src/lib.rs:170`. |

No TOML config schema — `soil_verlet` has no `[verlet]` section or config struct.
`dt` is read from `Atom::dt`, set externally by `soil_core`'s run loop / input layer.

---

## Config / TOML schema

None. The integrator takes `dt` from `Atom::dt` (`src/lib.rs:133`, `src/lib.rs:171`),
which is written by `soil_core`'s input and run plugins. The only user-facing choice
is whether to call `VelocityVerletPlugin::new()` vs. `VelocityVerletPlugin::for_stage("name")`.

---

## Key behaviors, invariants, and gotchas

### Schedule placement (cite: `src/lib.rs:108-116`)

- `initial_integration` → `ParticleSimScheduleSet::InitialIntegration`
- `final_integration` → `ParticleSimScheduleSet::FinalIntegration`

Full canonical per-step order from `docs/src/reference/internals.md:48-55`:

```
Setup
  → Pre / Initial / PostInitialIntegration   ← initial_integration runs at Initial
  → Pre / Exchange
  → Pre / Neighbor
  → Pre / Force / PostForce
  → Pre / Final / PostFinalIntegration        ← final_integration runs at Final
```

### Force zeroing is NOT done here (`src/lib.rs:37-39`)

`soil_core` zeros forces at `PostInitialIntegration` (after drift, before Force phase).
This means `initial_integration` reads genuinely last-step forces for its half-kick.
Physics tiers must accumulate into `Atom.force` with `+=`, not `=`, unless they set
force every step (as the `free_fall.rs` example does for a constant field).

### Ordering vs. force computation

`initial_integration` runs **before** forward comm and force computation;
`final_integration` runs **after** force computation and reverse comm. Force
computation by a physics tier must land in `ParticleSimScheduleSet::Force`.

### Local-only iteration (`src/lib.rs:141`, `src/lib.rs:176`)

Both systems loop `0..nlocal`. Ghost atoms (`nlocal..nlocal+nghost`) are never
integrated — they are read-only halo copies that are rebuilt by the substrate.

### Pinned particles via `inv_mass = 0` (`src/lib.rs:50-55`)

`half_dt_over_m = 0.5 * dt * inv_mass[i]`. When `inv_mass[i] == 0.0`, the atom
gets zero acceleration and zero drift — idiomatic pin under the integrator. For
a hard positional constraint that corrects accumulated floating-point drift, use
`soil_fixes`'s `[[pin]]` fix instead.

### Stage filtering (`src/lib.rs:105-117`)

`VelocityVerletPlugin::for_stage("name")` wraps both systems with
`.run_if(in_stage(name))`. Useful when multiple `[[run]]` stages coexist
(e.g. a dynamics stage and a minimizer stage with a different integrator).
Default (`new()`) runs in every stage.

### Force priming on step 0 (`examples/free_fall.rs:50-53`)

On the very first step, `initial_integration` runs before any force system has
set forces. If forces are not primed (set before `app.run()` is called for
the first time), the first half-kick uses zero force, shifting the trajectory
by half a step. For constant-force fields this requires pre-setting `Atom.force`
before entering the run loop.

### Exactness for constant force

VV is algebraically exact for constant force (parabolic trajectory). The tests
`constant_force_parabolic_trajectory` and `free_particle_constant_velocity`
in `src/lib.rs:365`, `src/lib.rs:326` verify this to machine precision.

### No PBC / image wrap

`soil_verlet` does not apply periodic boundary conditions. PBC wrap and image
tracking live in `soil_core` (applied at the `Exchange` phase). Position drift
here can push atoms outside the box; the substrate corrects that in the next
full-rebuild step.

---

## Tutorial outline

For `docs/src/substrate/integration.md` (already a stub — expand in place):

1. **What the integrator does** — kick-drift-kick math block (already there).
2. **Where it runs** — cite `ParticleSimScheduleSet` phases, point to internals.md.
3. **Usage snippet** — `VelocityVerletPlugin::new()` and `for_stage()`.
4. **Force zeroing and ordering** — explain PostInitialIntegration zeroing, why
   `initial_integration` sees last step's forces.
5. **Pinned particles** — `inv_mass = 0` idiom vs. `soil_fixes`'s `[[pin]]`.
6. **Force priming** — step-0 gotcha, show `free_fall.rs` pattern.
7. **Stage filtering** — when to use `for_stage()`.
8. **What this does not do** — PBC, rotation, neighbor list.

A runnable example (`examples/free_fall.rs`) already exists and can be referenced
as the canonical single-particle parabola check.

---

## Doc gaps

### Plugin name: CONFIRMED CORRECT

The tutorial `docs/src/tutorial/write-your-own-physics.md:159` uses
`VelocityVerletPlugin::new()`. Source confirms this at `src/lib.rs:80, 87`.
No name mismatch.

### The note in the tutorial is slightly over-cautious (`write-your-own-physics.md:186-188`)

> "Confirm them against your checkout, since plugin-group names are the most
> likely thing to have been renamed."

This hedge is appropriate for `CorePlugins` (lives in `dirt_core`, not verified here)
but not for `VelocityVerletPlugin` — that name is stable in `soil_verlet`. The
integration.md chapter should state the name authoritatively.

### `integration.md` stub is accurate but thin

`docs/src/substrate/integration.md` covers the math and phase placement correctly
but omits: stage filtering, force priming, PBC non-responsibility, and the
`inv_mass = 0` vs. `[[pin]]` distinction. All four are documented in `src/lib.rs`
module doc but not yet surfaced for end-users.

### No rotational integration note

The stub does not mention that `soil_verlet` is **translational only**. Quaternion /
angular-velocity integration for non-spherical particles is the physics tier's
responsibility (e.g. `dirt_atom`). Worth one sentence.

---

## Suggested placement

`docs/src/substrate/integration.md` — already exists and is linked from
`what-soil-owns.md:39` and `SUMMARY.md`. Expand in place; no new page needed.
The stub content is correct — add the gaps listed above.

---

## Source cross-references

| File | Lines | Content |
|---|---|---|
| `crates/soil_verlet/src/lib.rs` | 1-55 | Module doc: math, phase names, force-zeroing note, local-only, pinned particles |
| `crates/soil_verlet/src/lib.rs` | 62-119 | `VelocityVerletPlugin` struct, `new()`, `for_stage()`, `Plugin::build()` |
| `crates/soil_verlet/src/lib.rs` | 121-157 | `initial_integration` implementation |
| `crates/soil_verlet/src/lib.rs` | 159-188 | `final_integration` implementation |
| `crates/soil_verlet/examples/free_fall.rs` | 50-53 | Force priming pattern for step-0 correctness |
| `crates/soil_verlet/README.md` | 1-49 | User-facing summary of the same |
| `docs/src/substrate/integration.md` | 1-49 | Existing stub (accurate, needs expansion) |
| `docs/src/substrate/what-soil-owns.md` | 38-39 | Link to integration.md |
| `docs/src/reference/internals.md` | 48-55 | Full `ParticleSimScheduleSet` step order |
| `docs/src/tutorial/write-your-own-physics.md` | 159, 186-188 | `VelocityVerletPlugin::new()` usage; cautionary note on names |
