# SOIL — Substrate for Off-lattice Interacting Lagrangians

The method-agnostic particle substrate in the **GRASS → SOIL → physics** stack.

```
GRASS    framework: App, Plugin, Scheduler, IO, coupling      (no particles)
  └─ SOIL   substrate: Atom, domain decomposition, comm, neighbor lists   (no physics)
       └─ DIRT (DEM) / …   physics: forces, bonds, walls  (rides the substrate)
```

SOIL owns everything every particle method needs regardless of physics — the
base `Atom`, domain decomposition, ghost/halo communication, atom migration,
and neighbor-list construction — and knows nothing about contact forces, bonds,
or damage. Physics tiers extend a particle by registering an `AtomData` column;
the substrate then carries it through every migration, ghost exchange,
permutation, and restart automatically.

It suits explicit, time-stepping particle and grid solvers whose state is
resource-shaped with separable read/write sets. It is not aimed at implicit
global solvers (FEM, spectral, Newton-Krylov) where state is one large coupled
matrix.

## The interface

The one contract a physics tier builds against is the `AtomData` registration
mechanism (`#[forward]` / `#[reverse]` / `#[zero]`). It is documented in
[`docs/SOIL_ATOMDATA_CONTRACT.md`](docs/SOIL_ATOMDATA_CONTRACT.md).

## Crates

| crate | role |
|---|---|
| `soil_core` | base `Atom` + `AtomData` registry, comm, domain decomposition, neighbor, regions, groups |
| `soil_derive` | `#[derive(AtomData)]` proc macro |
| `soil_verlet` | velocity-Verlet translational integration |
| `soil_print` | thermo output, dump files (CSV/binary/VTP), restart |
| `soil_deform` | box deformation (engineering strain rate, velocity, target size) |

Built on [grass](https://github.com/SueHeir/grass). Consumed by
[dirt](https://github.com/SueHeir/dirt) (DEM) and intended for other
particle-method physics tiers.

## License

MIT OR Apache-2.0
