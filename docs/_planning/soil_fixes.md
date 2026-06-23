# Planning: `soil_fixes` documentation

> Target page: `docs/src/substrate/fixes.md` (already exists and is largely correct — this doc covers gaps and invariants for the next editing pass)

---

## Purpose

`soil_fixes` provides **method-agnostic kinematic constraints** on the base `Atom` state (position, velocity, force). "Method-agnostic" means the crate has no knowledge of DEM, SPH, or any other particle method — it operates only on fields that every `Atom` carries. The single constraint currently implemented is `pin`: a hard translational position lock.

DEM-specific fixes — `freeze` (zeros translational + rotational state), velocity damping, rotational damping — live downstream in DIRT's `dirt_fixes` crate and are explicitly excluded from this crate's scope.

---

## Public surface to document

| Item | Kind | Notes |
|---|---|---|
| `PinDef` | `#[derive(Deserialize)] struct` | The TOML-facing config struct for one `[[pin]]` block. One public field: `group: String`. `#[serde(deny_unknown_fields)]` — unknown TOML keys are a hard error. |
| `PinState` | `struct` (resource) | Runtime capture store: `captured: HashMap<String, HashMap<u32, [f64; 3]>>`. Keys are group name → global atom tag → initial pos. Stored as an app resource when at least one pin is configured. |
| `PinRegistry` | `struct` (resource) | Holds `pins: Vec<PinDef>`, populated from TOML at plugin build time. Always added as a resource, even when empty. |
| `SoilFixesPlugin` | `impl Plugin` | The one public plugin. Reads config, builds `PinRegistry`, conditionally adds `PinState` and three systems. Exposes `default_config()` — a commented-out TOML snippet. |
| Prelude | none | No `prelude` module; users import `SoilFixesPlugin` directly. |

Private API (not public, but document as implementation notes):
- `apply_pin_impl` — shared logic for both pre and post systems (lib.rs:200)
- `apply_pin_pre` / `apply_pin_post` — the two registered systems (lib.rs:234, lib.rs:246)
- `setup_pins` — setup-time validator and rank-0 printer (lib.rs:182)

---

## Config / TOML schema

### `[[pin]]`

```toml
[[group]]
name = "anchor"        # (required) define the group first
region = "base"        # region, atom_types, or dynamic — whatever GroupDef supports

[[pin]]
group = "anchor"       # (required) name of an already-defined atom group
```

**Fields:**

| Key | Type | Required | Meaning |
|---|---|---|---|
| `group` | string | yes | Name of the atom group to pin. Validated against `GroupRegistry` at setup (`ScheduleSetupSet::PostSetup`). A missing or misspelled group name is a hard error at simulation start. |

No other fields. `#[serde(deny_unknown_fields)]` makes extra keys a parse error (lib.rs:113).

**Multiple pins:** TOML array-of-tables means multiple `[[pin]]` blocks are supported, one per group.

**Group semantics:** `pin` pins every atom whose group mask entry is `true`. If the group uses a region, atoms that enter or leave the region after the initial capture are NOT re-pinned or un-pinned — the capture is one-shot (see Key Behaviors).

---

## Key behaviors, invariants & gotchas

### 1. Dual-phase scheduling (lib.rs:169–177)

`pin` registers systems at **two** `ParticleSimScheduleSet` phases:

- **`PreInitialIntegration`** (`apply_pin_pre`) — Restores pos, zeros vel and force **before** the Verlet drift. If the group mask is not yet populated (step 0 edge case), this is effectively a no-op (mask empty → `continue`). Ensures the Verlet kick-drift cannot move a pinned atom.
- **`PostForce`** (`apply_pin_post`) — Re-enforces after forces are computed. Also performs the **lazy capture** on the first step the mask is populated. Guarantees that `FinalIntegration` sees `f=0` on pinned atoms and that the *next* step's `PreInitialIntegration` restores from a valid capture.

Both phases call the same `apply_pin_impl` (lib.rs:200). Hooking both is why a pinned atom is bit-for-bit at its captured position whenever forces are evaluated — relevant for bonded-particle models where a pinned neighbor's position feeds into force calculation on flexible atoms.

### 2. Lazy, one-shot position capture (lib.rs:209–216)

Position is captured on the **first** step where `group.mask` is non-empty. This is intentional: it lets atom migration (MPI domain decomposition at startup) settle before the snapshot is taken. Once captured, the `HashMap` entry for that group is never updated — the snapshot is permanent for the run.

**Gotcha:** if a group is dynamic and atoms join the group after step 0, new members will have their position locked at whenever they first appear in a populated mask, not at step 0. Atoms that leave the group after capture are simply no longer in the mask and stop being pinned.

### 3. Tag-based position lookup (lib.rs:219–228)

The `PinState::captured` map keys on `atoms.tag[i]` (global atom tag, `u32`), not on the local index. This means pinned positions survive MPI atom migration between ranks — after a rebalance, the local index of an atom may change, but its tag is stable. Tested in `pin_lookup_is_tag_based_and_survives_reordering` (lib.rs:357).

### 4. Plugin is opt-in; zero overhead when unconfigured (lib.rs:160–163)

If `config.parse_array::<PinDef>("pin")` returns empty, the plugin adds only `PinRegistry` and returns immediately. No `PinState` resource, no update systems, no per-step cost.

### 5. Plugin ordering requirement (lib.rs:149–152)

`Config` resource must exist before `SoilFixesPlugin::build()` is called. In practice this means `SoilFixesPlugin` must be added after whatever plugin inserts the `Config` resource (typically `CorePlugins` from `dirt_core`).

### 6. `soil_fixes` vs `dirt_fixes` boundary

| Constraint | Crate | Touches rotational state? | Works for all particle methods? |
|---|---|---|---|
| `pin` | `soil_fixes` | No | Yes |
| `freeze` | `dirt_fixes` | Yes (`omega`, `torque`) | No — DEM-only |
| velocity damping | `dirt_fixes` | Yes (angular damping) | No — DEM-only |
| rotational damping | `dirt_fixes` | Yes | No — DEM-only |

The boundary rule: if a fix touches fields beyond `Atom::{pos, vel, force}`, it belongs in `dirt_fixes` or another method-specific tier.

### 7. `pin` vs `inv_mass = 0` (integration.md:42–49)

Setting `inv_mass = 0` prevents acceleration but does not correct drift from floating-point error or forces that bypass the integrator. `pin` is the hard constraint that bit-for-bit restores position. Both approaches hold a particle "fixed" but for different reasons and with different correctness guarantees.

### 8. `setup_pins` prints only on rank 0 (lib.rs:186–191)

Each configured pin prints a one-line summary at startup. This is MPI-rank-gated — only rank 0 prints. Useful for sanity-checking TOML config at the console.

---

## Tutorial outline

A tutorial section for `fixes.md` (or a dedicated tutorial page) should cover:

1. **Minimal working example** — define a `[[group]]` by region, add `[[pin]]`, add `SoilFixesPlugin` to the app. Show the startup print line.
2. **Bonded-particle use case** — explain why pinning a wall atom at pre-drift time matters when it is a force neighbor of a flexible atom.
3. **Writing your own fix** — `pin` as a template: plugin reads config array, stores a registry resource, registers systems at `ParticleSimScheduleSet` phases. Show the pattern in ~30 lines.
4. **`pin` vs `inv_mass = 0` vs `freeze`** — when to use each.
5. **MPI / migration** — brief note that capture is tag-based and survives rebalancing.

---

## Doc gaps

1. **`PinState` is public but undocumented in the current `fixes.md`** — users may inspect or serialize it for restart. Worth a brief note or a `#[doc(hidden)]` decision.
2. **Restart behavior** — `PinState::captured` is not serialized; on restart, positions are re-captured from the restarted atom positions, not from the original setup positions. This could surprise users if restart coordinates differ from the initial run's captured positions. Needs a note or a decision.
3. **Dynamic groups** — the doc says "captures position at setup" but doesn't explain what happens when a dynamic group's membership changes after capture. The code silently ignores new members (no entry in `captured`) and silently un-pins departing members (mask is false). This behavior should be documented explicitly.
4. **Multiple `[[pin]]` blocks** — the reference table says it's allowed (TOML array-of-tables), but the tutorial example shows only one. Add a two-group example.
5. **`SoilFixesPlugin::default_config()`** — the method is implemented (lib.rs:142) and returns a commented-out TOML snippet. Worth surfacing in docs so users know the TOML stub appears in generated configs.
6. **No prelude** — users must import `SoilFixesPlugin` by name. The crate map (reference/crates.md) notes the crate but doesn't mention the import path. A one-liner import example would reduce friction.

---

## Suggested placement

The existing `docs/src/substrate/fixes.md` is the correct home and already covers the essentials. The recommended editing pass:

- Add a **TOML schema table** (the `[[pin]]` field table from §Config above).
- Add a **restart caveat** for `PinState` (gap 2).
- Add a **dynamic groups note** (gap 3).
- Expand the `pin` vs `inv_mass = 0` vs `freeze` table with a "when to use" column.
- Link to a planned `tutorial/writing-a-fix.md` page using `pin` as the minimal template.

The `SUMMARY.md` already lists `fixes.md` under "The Substrate" — no structural changes to the book needed.

---

## File references

| File | Role |
|---|---|
| `crates/soil_fixes/src/lib.rs` | Entire implementation (410 lines) |
| `crates/soil_fixes/Cargo.toml` | Deps: `soil_core`, `grass_app`, `grass_scheduler`, `serde` |
| `crates/soil_fixes/README.md` | Mirrors lib.rs module doc |
| `docs/src/substrate/fixes.md` | The live doc page (already largely correct) |
| `docs/src/SUMMARY.md:11` | `fixes.md` listed under The Substrate |
| `docs/src/substrate/what-soil-owns.md:34,40` | `soil_fixes` row in the substrate crate table |
| `docs/src/reference/crates.md:10` | Crate map row with dirt_fixes cross-reference |
| `docs/src/substrate/integration.md:42–49` | `inv_mass = 0` alternative, cross-links to fixes.md |
