# soil_gpu

GPU storage and particle kernels for SOIL, on the device
[`grass_gpu`](https://github.com/SueHeir/grass/tree/main/crates/grass_gpu) shares. Anything
a different particle method would also need lives here, and physics plugs in from a tier
above, as `dirt_gpu` does. CPU and GPU kernels are separate code.

## Surface

| item | what it does |
|---|---|
| [`GpuState`](src/gpu_state.rs) | atom columns resident on the device, and the velocity Verlet loop, with on-device periodic wrap |
| [`GpuForce`](src/gpu_state.rs) | the hook a physics tier implements to add a force kernel |
| [`CellList`](src/cell_list.rs), [`Grid`](src/cell_list.rs) | neighbor build on the device, in cells that follow the box |
| [`NeighborSlots`](src/neighbor_slots.rs) | persistent per-neighbor state, for contact springs or bonds |
| [`Boundary`](src/boundary.rs) | planar boundary geometry |
| [`ResidentAtoms`](src/resident.rs) | a `GpuState` as an App resource, in step with the CPU `Atom` |

## Keeping CPU systems working

`ResidentAtomsPlugin` registers a bridge with `grass_scheduler`'s `CoherenceRegistry`. A CPU
system that touches `Atom` while the GPU copy is newer triggers a download first, of
positions, velocities, and any extra columns a tier registered with `AtomColumnSync`. It
never has to know a GPU exists. A run with no CPU reader makes no transfers at all, which
`tests/resident_atoms.rs` asserts.

The kernel system takes `ResMut<CoherenceRegistry>`, which tells the scheduler it manages
its own coherence. It calls `begin_kernels` to upload anything the CPU wrote, and
`end_kernels` after launching.

## Precision

The kernels are `f32`, because Apple GPUs have no `f64`. Values are cast on the way up and
down, so results agree with a CPU run within the `f32` band.

## License

MIT OR Apache-2.0
