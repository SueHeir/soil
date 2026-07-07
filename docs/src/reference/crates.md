# Crate Map

| crate | role |
|---|---|
| [`soil_core`](https://github.com/SueHeir/soil/tree/main/crates/soil_core) | base `Atom` + `AtomData` registry, comm, domain decomposition, neighbor, regions, groups, [`Octree`](../substrate/octree-near-far.md#what-soil-provides) near/far partitioning |
| [`soil_derive`](https://github.com/SueHeir/soil/tree/main/crates/soil_derive) | `#[derive(AtomData)]` proc macro |
| [`soil_verlet`](https://github.com/SueHeir/soil/tree/main/crates/soil_verlet) | velocity-Verlet translational integration |
| [`soil_print`](https://github.com/SueHeir/soil/tree/main/crates/soil_print) | thermo output, dump files (CSV/binary/LAMMPS/VTP), restart |
| [`soil_deform`](https://github.com/SueHeir/soil/tree/main/crates/soil_deform) | box deformation (strain rate, velocity, target size) + Lees–Edwards `xy` shear |
| [`soil_fixes`](https://github.com/SueHeir/soil/tree/main/crates/soil_fixes) | method-agnostic position constraint `pin` (DEM fixes — `freeze`, velocity damping — live in [dirt_fixes](https://github.com/SueHeir/dirt)) |

Built on [grass](https://github.com/SueHeir/grass). Consumed by
[dirt](https://github.com/SueHeir/dirt) (DEM) and intended for other
particle-method physics tiers.

## `soil_core::tree`

The [`soil_core::tree::Octree`](../substrate/octree-near-far.md#what-soil-provides)
reference covers the public long-range substrate API:
[`build`](../substrate/octree-near-far.md#build),
[`build_in_box`](../substrate/octree-near-far.md#build_in_box),
[`near_pairs`](../substrate/octree-near-far.md#near_pairs),
[`far_field_partition`](../substrate/octree-near-far.md#far_field_partition),
and [`monopoles`](../substrate/octree-near-far.md#monopoles_method).
The mdBook tutorial/reference page is
[Octree Near/Far Partition](../substrate/octree-near-far.md).

## Extending output (`soil_print`)

The dedicated [`soil_print` output and restart reference](../output/soil-print.md)
covers thermo columns and intervals, dump formats and binary layout, restart
read/write behavior, VTP ghost padding, `save_at_end`, and the `DumpRegistry`
extension points. See `examples/output_wiring/main.rs` for a compiling
end-to-end use, and `examples/minimal_sim/main.rs` for assembling the substrate
plugins serially.
