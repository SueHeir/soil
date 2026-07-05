# `soil_print`: Output and Restart Reference

`soil_print` is SOIL's method-agnostic output layer. `PrintPlugin` wires four
subsystems into the app:

| subsystem | TOML section | output |
|---|---|---|
| thermo | `[thermo]` plus the active `[[run]]` stage's `thermo` override | console columns |
| dump | `[dump]` | per-atom snapshots in text, binary, LAMMPS, or a registered format |
| restart | `[restart]` | per-rank checkpoints for later startup restore |
| VTP | `[vtp]` | per-rank ParaView `.vtp` files |

The plugin loads `ThermoConfig`, `DumpConfig`, `RestartConfig`, and `VtpConfig`,
registers the built-in dump formats, and schedules output after final
integration. The source of truth is `crates/soil_print/src/lib.rs`: `PrintPlugin`
builds the resources and systems near `PrintPlugin::build`, while each config
struct documents the accepted TOML keys with `serde(deny_unknown_fields)`.

## Thermo

`[thermo]` selects columns, not the print interval.

```toml
[thermo]
columns = ["step", "atoms", "ke", "temp", "walltime", "stepps"]

[[run]]
steps = 10000
thermo = 100
```

If `columns` is omitted, SOIL prints:

```text
step, atoms, ke, neighbors, walltime, stepps
```

Built-in compute names are `step`, `atoms`, `ke`, `temp`, `neighbors`,
`walltime`, and `stepps`. A column may use `compute/group` syntax, such as
`ke/mobile` or `temp/fluid`, to restrict `atoms`, `ke`, or `temp` to a named
group. Values published with `Thermo::set(name, value)` are also valid column
names. When a `VirialStress` resource exists, `soil_print` publishes
`virial_xx`, `virial_yy`, `virial_zz`, `virial_xy`, `virial_xz`, and
`virial_yz` for use as columns.

The print interval comes from the active run stage's integer `thermo` override.
If that override is absent, `setup_thermo` uses 100 steps. This is intentionally
separate from `[thermo]`, which only controls column selection and ordering.

## Dump

`[dump]` writes per-atom snapshots when its interval fires.

```toml
[dump]
interval = 1000
format = "lammps"  # "text", "binary", "lammps", or a registered format
per_rank = false
ghost = false

[[run]]
steps = 10000
dump_interval = 500
```

The stage override `dump_interval` takes precedence over `[dump].interval`.
`interval = 0` disables periodic dump output.

With `per_rank = false`, every rank builds a frame and sends it to rank 0; rank
0 writes one `dump/dump_{step}.{ext}` file. With `per_rank = true`, each rank
writes `dump/dump_{step}_rank{rank}.{ext}` without a gather. If the app sets an
`output_dir`, dump files live under `{output_dir}/dump/`.

Built-in formats:

| `format` | extension | layout |
|---|---|---|
| `text` | `.csv` | header row, then comma-separated atom rows |
| `binary` | `.bin` | little-endian binary records |
| `lammps` | `.lammpstrj` | LAMMPS dump with timestep, box bounds, and atom table |

All formats start from the same `DumpFrame`: `tag`, `type`, `x/y/z`,
`vx/vy/vz`, `fx/fy/fz`, `radius`, registered scalar columns, registered vector
columns split into `_x/_y/_z` for text and LAMMPS, and optional `IsGhost`.

### Binary Wire Layout

The binary dump format is deliberately simple and little-endian:

```text
u32 atom_count
repeated atom_count times:
  u32 tag
  u32 type
  f64 x, y, z
  f64 vx, vy, vz
  f64 fx, fy, fz
  f64 radius
  f64 each registered scalar, in registration order
  f64 x, y, z for each registered vector, in registration order
  u8 IsGhost, only when [dump].ghost = true
```

The file does not include a self-describing schema for registered columns. A
reader must know the active `DumpRegistry` registrations and whether `ghost`
was enabled for that run.

### Ghost Atoms

Dump output includes only local atoms by default. With `ghost = true`, the frame
uses `atoms.len()` instead of `atoms.nlocal`, appends an `IsGhost` column, and
marks indices `>= nlocal` as `1`. Registered scalar and vector callbacks are
expected to return local-atom values; when ghosts are present, missing registered
values are padded with `0`.

## Restart

`[restart]` writes per-rank checkpoints and can read the latest checkpoint on
startup.

```toml
[restart]
interval = 5000
format = "bincode" # or "json"
read = false

[[run]]
steps = 10000
restart_interval = 2500
```

The stage override `restart_interval` takes precedence over
`[restart].interval`. `interval = 0` disables periodic restart writes.

Each rank writes only its local atoms to:

```text
restart/restart_{step}_rank{rank}.bin   # bincode
restart/restart_{step}_rank{rank}.json  # json
```

If the app sets an `output_dir`, restart files live under
`{output_dir}/restart/`. With `read = true`, `read_restart` scans that directory
for the highest-numbered file matching the current rank and selected format,
then restores it during first-stage setup. If no matching file exists, rank 0
prints a message and the run continues. If the selected file cannot be opened or
deserialized, restart read is fatal: the process prints an error and exits with
status 1 rather than starting from an invalid or partially restored state.

Restart data stores the base `Atom` fields that define local particle state:
`natoms`, `total_cycle`, `dt`, `tag`, `atom_type`, `pos`, `vel`, `force`, `mass`,
and `cutoff_radius`. Coordinates, velocities, forces, masses, and radii are
serialized as `f64` regardless of the build's storage precision, then converted
back to `Real` or `Accum` on restore. Every registered `AtomData` extension is
serialized through `AtomDataRegistry::pack_all_for_restart(nlocal)` and restored
with `unpack_all_from_restart`.

The restart struct still contains legacy rotational fields:

```text
omega_x/y/z
torque_x/y/z
ang_mom_x/y/z
quaternion
```

They are `#[serde(default)]` compatibility fields so older restart files can be
read. New restarts write them empty; rotational or method-specific state belongs
in an `AtomData` extension, not in SOIL's base `Atom`.

## VTP

`[vtp]` writes ASCII ParaView PolyData files.

```toml
[vtp]
interval = 500

[[run]]
steps = 10000
vtp_interval = 250
```

The stage override `vtp_interval` takes precedence over `[vtp].interval`.
`interval = 0` disables VTP output.

Each rank writes its own file:

```text
vtp/{step}CYCLE_{rank}RANK.vtp
```

or `{output_dir}/vtp/{step}CYCLE_{rank}RANK.vtp` when an output directory is set.
There is no MPI gather for VTP. Each file includes local atoms and ghost atoms
(`atoms.len()`), with built-in arrays for position, `Radius`, `Vel_Mag`, and
`IsGhost`. Registered scalar and vector callbacks are written as `Float32` point
data arrays. As with dump ghosts, registered callback data shorter than
`atoms.len()` is padded with zeros, so ghosts usually carry zero values for
plugin columns unless the callback deliberately returns ghost entries.

## `save_at_end`

A run stage with `save_at_end = true` forces one dump and one restart at the end
of that stage, independent of the periodic dump and restart intervals.

```toml
[[run]]
name = "settle"
steps = 5000
save_at_end = true
```

`check_stage_end_save` runs before `update_cycle`, while the current stage index
is still valid. It fires on the last step of the stage (`cycle_count + 1 ==
cycle_remaining`) or when an early stage advance was requested. Skipped stages
with zero remaining cycles do not write. The saved files use the current
`[dump]` and `[restart]` format settings and the current `run_state.total_cycle`.

## Extending Output

Physics tiers extend output through public resources installed by `PrintPlugin`.
The complete compiling example is
[`examples/output_wiring/main.rs`](../../../examples/output_wiring/main.rs).

```rust,ignore
let dump_reg = app.get_resource_ref::<DumpRegistry>().unwrap();

dump_reg.register_scalar("speed", |atoms, registry| {
    (0..atoms.nlocal as usize)
        .map(|i| {
            let v = atoms.vel[i];
            (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
        })
        .collect()
});

dump_reg.register_vector("force", |atoms, registry| {
    (0..atoms.nlocal as usize).map(|i| atoms.force[i]).collect()
});

dump_reg.register_format("xyz", |frame| {
    let mut file = std::fs::File::create(format!("{}.xyz", frame.path_stem))?;
    // write frame.pos, frame.scalars, frame.vectors, frame.box_info, ...
    Ok(())
});
```

`register_scalar` and `register_vector` take `&self`, so a plugin can register
columns from its `build()` method using a shared `DumpRegistry` reference.
Callbacks run only on output steps. They receive `&Atom` and
`&AtomDataRegistry`, which is where method-specific per-atom state should live.

`register_format` adds a named writer selected by `[dump].format`. Built-in
formats (`text`, `binary`, `lammps`) are registered through the same mechanism.
When multiple writers use the same name, the last registration wins, so a
physics tier may intentionally override a built-in format. If `[dump].format`
names no registered writer, SOIL prints a warning and continues without writing
that dump frame.

Custom thermo values use `Thermo::set` from an update system:

```rust,ignore
fn publish_energy(mut thermo: ResMut<Thermo>) {
    thermo.set("pe", 1.234e-3);
}
```

Then add the value name to `[thermo].columns`, for example:

```toml
[thermo]
columns = ["step", "atoms", "ke", "pe"]
```

`Thermo::set` overwrites the previous value, so systems should refresh custom
values whenever thermo output should report them.
