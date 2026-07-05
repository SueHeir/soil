# lj_dimer

Lennard-Jones molecular-dynamics dimer on the SOIL substrate. The example uses
the neighbor list, periodic ghost communication, a custom `AtomData` energy
column, reverse communication, and `soil_verlet`, while keeping the LJ potential
outside the substrate crates.

Run from the repository root:

```sh
cargo run --release --example lj_dimer
```
