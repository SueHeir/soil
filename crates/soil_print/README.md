# soil_print

Output systems for [SOIL](https://github.com/SueHeir/soil) simulations: periodic console logging, dump files, restart checkpoints, and ParaView visualization.

## What it does

`PrintPlugin` registers four independent output subsystems, each configured by its own TOML section and active only when its interval is non-zero (thermo defaults to every 100 steps):

| System  | TOML section | Description                                                              |
|---------|--------------|--------------------------------------------------------------------------|
| Thermo  | `[thermo]`   | Periodic console metrics with configurable columns                        |
| Dump    | `[dump]`     | Per-atom snapshots in CSV (`text`), `binary`, or `lammps` (`.lammpstrj`)  |
| Restart | `[restart]`  | Checkpoint files (`bincode` or `json`) for resuming simulations           |
| VTP     | `[vtp]`      | ParaView-compatible `.vtp` visualization files                            |

Built-in thermo columns: `step`, `atoms`, `ke`, `temp`, `neighbors`, `walltime`, `stepps`. Any column may use `"compute/group"` syntax (e.g. `"ke/mobile"`) to filter by a named atom group, and any value pushed via `Thermo::set` is also available as a column. When `VirialStress` is present, `virial_xx`…`virial_yz` columns are published automatically.

## Key types

| Type | Role |
|------|------|
| `PrintPlugin` | Registers all four output systems |
| `Thermo` | Runtime thermo state; push custom values via `.set(name, value)` |
| `ThermoConfig` / `DumpConfig` / `RestartConfig` / `VtpConfig` | TOML config structs |
| `DumpRegistry` | Register per-atom scalar/vector columns and custom dump formats for dump + VTP |
| `DumpFrame` / `BoxInfo` | Frame of per-atom data and box bounds handed to format writers |

Dump output supports `per_rank` (one file per MPI rank vs. a single gathered file on rank 0) and `ghost` (include halo atoms, flagged with an `IsGhost` column). Plugins extend dump/VTP columns with `DumpRegistry::register_scalar` / `register_vector`, and add new dump formats with `DumpRegistry::register_format`.

## Configuration

```toml
[thermo]
columns = ["step", "atoms", "ke", "temp", "walltime", "stepps"]

[dump]
interval = 5000
format = "text"     # "text" (CSV), "binary", "lammps", or a plugin-registered format
per_rank = false    # true: one file per rank (no gather)
ghost = false       # include ghost (halo) atoms

[restart]
interval = 10000
format = "bincode"  # or "json"
read = false        # read the latest restart file at startup

[vtp]
interval = 500
```

Per-stage `[[run]]` overrides (`thermo`, `dump_interval`, `restart_interval`, `vtp_interval`) take precedence over these defaults, and `save_at_end = true` on a stage writes dump + restart files when that stage ends.

## Usage

```rust,ignore
use soil_print::PrintPlugin;

app.add_plugins(PrintPlugin);
```

Restart files automatically save and restore all `AtomData` extensions registered with the `AtomDataRegistry`.

## License

MIT OR Apache-2.0
