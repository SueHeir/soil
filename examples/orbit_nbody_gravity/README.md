# orbit_nbody_gravity

N-body gravity demonstration built on SOIL octree primitives. Gravity-specific
state, timestepping, and validation remain in the example so `soil_core` stays
particle-method agnostic.

Run from the repository root:

```sh
cargo run --release --example orbit_nbody_gravity
```
