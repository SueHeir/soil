# Planning: `soil_print` documentation

## Purpose

`soil_print` is the output tier of the GRASS→SOIL→DIRT stack.  It gives every
simulation four independent, zero-overhead-when-disabled output subsystems:

- **Thermo** — periodic console metrics line (step, KE, atoms, neighbors, wall
  time, steps/sec, and any value a plugin pushes via `Thermo::set`).
- **Dump** — per-atom snapshots in CSV (`text`), little-endian binary
  (`binary`), or OVITO-native LAMMPS dump (`lammps`), plus any format a plugin
  registers; MPI-gathered to one file per step, or one file per rank.
- **Restart** — checkpoint files (bincode or JSON) that fully round-trip core
  `Atom` fields plus every `AtomData` extension via `pack_all_for_restart` /
  `unpack_all_from_restart`.
- **VTP** — per-rank ParaView `.vtp` files (ASCII PolyData) including positions,
  radii, velocity magnitude, ghost flags, and plugin-registered columns.

All subsystems are wired up by adding one plugin (`PrintPlugin`) and configured
entirely through TOML.

---

## Public surface to document

### Plugin

| Item | Location |
|------|----------|
| `PrintPlugin` (struct + `Plugin` impl) | `lib.rs:678` |

`PrintPlugin::build` loads all four config structs, constructs the `DumpRegistry`
with the three built-in formats pre-registered, and schedules all output systems
in `ParticleSimScheduleSet::PostFinalIntegration` (and one setup system per
subsystem in `ScheduleSetupSet::PostSetup`).  `lib.rs:707–732`.

### Config structs (all `serde::Deserialize`)

| Struct | TOML key | Location |
|--------|----------|----------|
| `ThermoConfig` | `[thermo]` | `lib.rs:96` |
| `DumpConfig` | `[dump]` | `lib.rs:235` |
| `RestartConfig` | `[restart]` | `lib.rs:285` |
| `VtpConfig` | `[vtp]` | `lib.rs:663` |

### Runtime resources

| Resource | Role |
|----------|------|
| `Thermo` | Holds interval, wall-clock timer, parsed columns, user-pushed values map |
| `DumpRegistry` | Registry of scalar/vector/format callbacks; interior-mutable so plugins call `&self` |
| `DumpFrame` | One step's worth of per-atom data handed to format writers |
| `BoxInfo` | Box bounds + periodicity snapshot for dump headers |

### Extension API — the three registration methods

```rust
// All called via shared ref during a plugin's build():
let dump_reg = app.get_resource_ref::<DumpRegistry>().unwrap();

dump_reg.register_scalar("pressure", |atoms, registry| -> Vec<f64> { … });
// lib.rs:566

dump_reg.register_vector("omega", |atoms, registry| -> Vec<[f64; 3]> { … });
// lib.rs:579

dump_reg.register_format("xyz", |frame: &DumpFrame| -> io::Result<()> { … });
// lib.rs:606
```

And the thermo-value push (called from an update system):
```rust
thermo.set("pe", value);   // lib.rs:207
```

### Prelude / re-exports

`soil_print` has no `prelude` module.  Users import types directly:
`use soil_print::{PrintPlugin, DumpRegistry, DumpFrame, Thermo, BoxInfo,
ThermoConfig, DumpConfig, RestartConfig, VtpConfig};`

---

## Config / TOML schema

### `[thermo]`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `columns` | `[String]` (optional) | `["step","atoms","ke","neighbors","walltime","stepps"]` | Ordered list of column names to print.  Use `"compute/group"` syntax (e.g. `"ke/mobile"`) to filter by atom group.  Any name pushed via `Thermo::set` is valid here. |

Built-in compute names: `step`, `atoms`, `ke`, `temp`, `neighbors`, `walltime`,
`stepps`.  When `VirialStress` is present: `virial_xx`, `virial_yy`,
`virial_zz`, `virial_xy`, `virial_xz`, `virial_yz` are published automatically
(`lib.rs:920–944`).

The **print interval** is not a TOML field here — it comes from
`[[run]]` stage `thermo` override key (integer), defaulting to 100
(`lib.rs:764–768`).

### `[dump]`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `interval` | `usize` | `0` | Write every N steps; 0 = disabled |
| `format` | `String` | `"text"` | `"text"` (CSV, `.csv`), `"binary"` (LE binary, `.bin`), `"lammps"` (`.lammpstrj`), or any plugin-registered name |
| `per_rank` | `bool` | `false` | `false`: gather all ranks to rank 0, write `dump/dump_{step}.{ext}`; `true`: each rank writes `dump/dump_{step}_rank{rank}.{ext}` |
| `ghost` | `bool` | `false` | Include ghost (halo) atoms; ghosts are zero-padded for registered columns and flagged with an `IsGhost` column |

Per-stage override key: `dump_interval` (integer) (`lib.rs:1112–1115`).

### `[restart]`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `interval` | `usize` | `0` | Write every N steps; 0 = disabled |
| `format` | `String` | `"bincode"` | `"bincode"` (compact binary, `.bin`) or `"json"` (human-readable, `.json`) |
| `read` | `bool` | `false` | Scan `restart/` at startup, find the highest-numbered file for this rank + format, deserialize and restore all atom state |

Per-stage override key: `restart_interval` (integer) (`lib.rs:1398–1401`).
File naming: `restart/restart_{step}_rank{rank}.{ext}` (`lib.rs:1431`, `1436`).

### `[vtp]`

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `interval` | `usize` | `0` | Write every N steps; 0 = disabled |

Per-stage override key: `vtp_interval` (integer) (`lib.rs:989–992`).
File naming: `{output_dir}/vtp/{step}CYCLE_{rank}RANK.vtp` (`lib.rs:1014`).

### Stage-level overrides (`[[run]]`)

Any `[[run]]` stage may carry these keys in its `overrides` table:

| Key | Effect |
|-----|--------|
| `thermo` | Override thermo print interval for this stage |
| `dump_interval` | Override dump write interval |
| `restart_interval` | Override restart write interval |
| `vtp_interval` | Override VTP write interval |
| `save_at_end` | `true` → write dump + restart at the last step of the stage regardless of intervals |

`save_at_end` is handled by `check_stage_end_save`, which runs
`.before("update_cycle")` so the stage index is still valid when it fires
(`lib.rs:1446–1512`).

---

## Key behaviors, invariants & gotchas

### Scheduling

All output systems run in `ParticleSimScheduleSet::PostFinalIntegration` — after
the second half-kick (`lib.rs:727–731`).  `sync_dump_box` runs in both
`PostSetup` (initial box snapshot) and `PostForce` (updated each step) so format
writers always see current box bounds (`lib.rs:724`, `729`, `1087–1097`).

### Thermo interval source

The thermo interval is **not** read from `[thermo]`; it comes from the `thermo`
key in the current `[[run]]` stage overrides, defaulting to 100 steps
(`lib.rs:764–768`).  `[thermo]` only controls which columns are displayed.

### Restart completeness

`RestartData` round-trips:
- Core `Atom` fields: `natoms`, `total_cycle`, `dt`, `tag`, `atom_type`, all
  three components of `pos`, `vel`, `force`, `mass`, `cutoff_radius`
  (`lib.rs:319–356`).
- All registered `AtomData` extensions: packed into `atom_data_buffers` via
  `registry.pack_all_for_restart(nlocal)` (`lib.rs:382`); restored via
  `registry.unpack_all_from_restart(&data.atom_data_buffers)` (`lib.rs:1617`).

**Field-order is the wire layout.**  Reordering or inserting fields in an
`AtomData` impl changes the restart file format (see
`docs/src/substrate/writing-atomdata.md:47–49`).

Legacy rotational fields (`omega_*`, `torque_*`, `ang_mom_*`, `quaternion`) are
kept as `#[serde(default)]` empty vecs for reading old files but are written as
empty — rotational state now lives in `AtomData` extensions
(`lib.rs:337–355`, `lib.rs:383–395`).

`read_restart` runs **only on the first stage** (`first_stage_only()` guard,
`lib.rs:723`).  Restarting into a later stage is not supported.

Read failure is fatal (`process::exit(1)`) — `lib.rs:1579`, `1585`, `1590`.

### Dump format dispatch

`DumpRegistry` keeps a `Vec` of `(name, fn)` pairs.  `write_format` iterates
**in reverse** and returns the last-registered matching writer, so plugins can
override a built-in format by registering the same name (`lib.rs:631–637`).
Unknown format names print a warning and return without error (`lib.rs:1218–1222`).

### Ghost atom handling in dumps

Registered scalar/vector callbacks receive `atoms.nlocal` atoms.  Ghost atoms
(indices `nlocal..len`) get zero-padded values for registered columns
(`lib.rs:1187–1190`).

### VTP output

Each rank writes its own file (no MPI gather).  The file includes both local and
ghost atoms (`atoms.len()`, not `atoms.nlocal`) (`lib.rs:1018–1019`).  Registered
callbacks are padded to `n` with zeros, not trimmed to `nlocal`, so registered
scalar/vector data for ghost atoms will be `0` unless the callback deliberately
fills beyond `nlocal` (`lib.rs:1044–1071`).

### `check_stage_end_save` timing gotcha

This system runs `.before("update_cycle")`.  The per-stage cycle counter
(`run_state.cycle_count[index]`) reflects cycles completed so far; the system
triggers when `count + 1 == remaining` (the last step) or when
`scheduler_manager.advance_requested` is set (early stage advance)
(`lib.rs:1479–1488`).  Skipped stages (`remaining == 0`) are ignored
(`lib.rs:1475–1477`).

### MPI gather in dumps

With `per_rank = false`, all ranks participate.  Non-root ranks call
`comm.0.send_f64(0, &frame.pack())` and do not write files.  Root receives in
rank order and appends via `push_unpacked` (`lib.rs:1229–1245`).

---

## Tutorial outline: adding output to a simulation

1. Add `PrintPlugin` to the app alongside the physics plugin.
2. Set TOML config for each subsystem (intervals, format).
3. **Custom thermo column**: in the physics plugin's update system, borrow
   `ResMut<Thermo>` and call `thermo.set("pe", value)` every thermo step.
   Add `"pe"` to `[thermo] columns`.
4. **Custom dump/VTP column**: in `build()`, get a shared ref to `DumpRegistry`
   and call `register_scalar` or `register_vector`.  The callback gets
   `(&Atom, &AtomDataRegistry)`.  Use `registry.get::<MyAtomData>()` to reach
   extension fields.
5. **Custom dump format**: call `register_format("myformat", |frame| { … })`.
   The writer creates `{frame.path_stem}.myext`; set `[dump] format = "myformat"`.
6. **Restart**: set `[restart] interval = N` and `[restart] read = true` on the
   second run.  All `AtomData` extensions registered with `AtomPlugin` are
   automatically included.

See `crates/soil_print/examples/output_wiring.rs` for a compiling demonstration
of steps 3–5.

---

## Doc gaps

- **No dedicated `soil_print` page in the mdBook.** Currently only a one-liner
  in `reference/crates.md:8` and a 13-line stub under "Extending output"
  (`docs/src/reference/crates.md:16–32`).
- **Thermo interval source undocumented.** The fact that `thermo` is a stage
  override key, not a `[thermo]` field, is not explained anywhere in docs.
- **`save_at_end` semantics not documented.** The stage-end save behavior,
  including the `.before("update_cycle")` subtlety and the skipped-stage guard,
  has no doc coverage.
- **Legacy restart fields not explained.** The `omega_*` / `quaternion`
  compatibility vecs will confuse readers of `RestartData`; a migration note is
  needed.
- **VTP ghost-padding behavior not documented.** The asymmetry between VTP
  (includes ghosts, callbacks zero-padded beyond `nlocal`) and dump (ghosts
  optional via `ghost = true`) is non-obvious.
- **`DumpRegistry` last-registration-wins override rule** is only in the inline
  doc; it should be in user-facing docs so plugins know they can shadow built-in
  formats.
- **No docs on binary dump wire layout** (u32 count header, u32 tag/type, f64
  everything else, optional u8 IsGhost trailing byte) for readers that parse the
  `.bin` files.
- **Restart read failure is fatal** (`process::exit(1)`) — this should be
  documented so users know a missing or corrupt restart file halts the process
  rather than silently starting fresh.
- **`prelude` absent** — consider adding one or noting explicitly that there is
  none.

---

## Suggested placement

| New page | Location in SUMMARY.md |
|----------|------------------------|
| `output/overview.md` — subsystem table, TOML quickref, `PrintPlugin` usage | New "Output" section, before Reference |
| `output/thermo.md` — columns, group syntax, `Thermo::set`, virial columns, interval-from-stage gotcha | Under "Output" |
| `output/dump.md` — CSV/binary/LAMMPS formats, wire layout, `per_rank`, `ghost`, extension API | Under "Output" |
| `output/restart.md` — completeness contract, field-order invariant, legacy fields, fatal-on-read-failure, `save_at_end` | Under "Output" |
| `output/vtp.md` — per-rank write, ghost inclusion, ParaView workflow | Under "Output" |
| `output/extending.md` — `register_scalar` / `register_vector` / `register_format` / `Thermo::set` with worked examples mirroring `output_wiring.rs` | Under "Output" |

The `reference/crates.md` stub under "Extending output" can be replaced with a
link to `output/extending.md` once that page exists.
