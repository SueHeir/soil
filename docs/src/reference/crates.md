# Crate Map

| crate | role |
|---|---|
| [`soil_core`](https://github.com/SueHeir/soil/tree/master/crates/soil_core) | base `Atom` + `AtomData` registry, comm, domain decomposition, neighbor, regions, groups |
| [`soil_derive`](https://github.com/SueHeir/soil/tree/master/crates/soil_derive) | `#[derive(AtomData)]` proc macro |
| [`soil_verlet`](https://github.com/SueHeir/soil/tree/master/crates/soil_verlet) | velocity-Verlet translational integration |
| [`soil_print`](https://github.com/SueHeir/soil/tree/master/crates/soil_print) | thermo output, dump files (CSV/binary/LAMMPS/VTP), restart |
| [`soil_deform`](https://github.com/SueHeir/soil/tree/master/crates/soil_deform) | box deformation (strain rate, velocity, target size) + Lees–Edwards `xy` shear |
| [`soil_fixes`](https://github.com/SueHeir/soil/tree/master/crates/soil_fixes) | method-agnostic position constraint `pin` (DEM fixes — `freeze`, velocity damping — live in [dirt_fixes](https://github.com/SueHeir/dirt)) |

Built on [grass](https://github.com/SueHeir/grass). Consumed by
[dirt](https://github.com/SueHeir/dirt) (DEM) and intended for other
particle-method physics tiers.

## Extending output (`soil_print`)

A physics tier adds its own per-atom dump/VTP columns and thermo values through
`soil_print`'s public registration API, all called from a plugin's `build()`:

- `DumpRegistry::register_scalar` / `register_vector` — per-atom columns that
  appear in CSV/LAMMPS dumps and VTP output.
- `DumpRegistry::register_format` — a whole new named dump file format, selected
  by `[dump] format = "<name>"`.
- `Thermo::set(name, value)` — publish a value that becomes a thermo column when
  listed in `[thermo] columns`.

See `crates/soil_print/examples/output_wiring.rs` for a compiling end-to-end use,
and `crates/soil_core/examples/minimal_sim.rs` for assembling the substrate
plugins serially.

> **Three `soil_print` behaviors that surprise people.** A bad restart file is
> **fatal**: if `[restart] read = true` and the file is missing or corrupt, the
> process calls `process::exit(1)` rather than starting fresh. The **thermo
> print interval is not a `[thermo]` key** — it comes from the `thermo` override
> on the active `[[run]]` stage (default 100 steps); `[thermo]` only selects
> *which* columns print. And `save_at_end` (a `[[run]]` override that forces a
> dump + restart on the stage's last step) is scheduled `.before("update_cycle")`
> so the stage index is still valid when it fires — move it after the cycle
> advance and it would attribute the save to the wrong stage.

> *Stub — expand each row into its own page as the API stabilizes.*
