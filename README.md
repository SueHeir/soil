# SOIL: Substrate for Off-lattice Interacting Lagrangians

**Distributed particle-simulation infrastructure: extensible particle storage,
spatial decomposition, communication, neighbor search, integration, and output
without prescribing a force law.**

<!-- disclaimer-banner -->
> **Research-software status:** This ecosystem is AI-authored and under active
> evaluation. GRASS, SOIL, and DIRT are the core architecture and DEM
> implementation; repositories prefixed `dev_` are experimental method
> demonstrations outside the author's domain expertise. Treat claims according
> to their linked evidence and documented limitations. See
> [DISCLAIMER.md](DISCLAIMER.md).
<!-- /disclaimer-banner -->

A particle method should begin with its state and interactions, not a new private
implementation of domain decomposition, ghost exchange, migration, neighbor
lists, restart files, and timestep plumbing. SOIL supplies that reusable
substrate while leaving the scientific model to the library built above it.

## What SOIL provides

- **Extensible particle storage** using structure-of-arrays columns registered
  through the `AtomData` contract.
- **Domains and boundaries** including fixed, periodic, shrink-wrapped, and
  triclinic boxes.
- **Spatial decomposition** with local ownership, particle migration, ghost
  particles, and load-balancing support.
- **MPI communication** for border construction, forward field replication,
  reverse accumulation, and multi-hop exchange.
- **Neighbor search** using spatial bins, configurable skin distances, rebuild
  decisions, and per-particle cutoff support.
- **Particle operations** including regions, groups, stable tags, permutation,
  and virial/kinetic-energy infrastructure.
- **Optional common plugins** for velocity-Verlet integration, box deformation,
  method-neutral constraints, thermo output, visualization, dumps, and restart.
- **Precision modes** for double, mixed, and single-precision builds.

SOIL owns no material model or interaction law. Contact mechanics, molecular
potentials, bond laws, and other physics belong in the method-specific tier that
uses the substrate.

## One declaration is the data-movement contract

A method extends each particle with typed columns and declares how those columns
participate in distributed execution:

```rust
use soil_derive::AtomData;

#[derive(AtomData)]
pub struct MethodAtom {
    /// State needed by neighboring ranks: copied from owners to ghosts.
    #[forward]
    pub internal_state: Vec<f64>,

    /// Contributions accumulated on ghosts and returned to their owners.
    #[reverse]
    #[zero]
    pub contribution: Vec<[f64; 3]>,

    /// Owner-only state: migrates with the particle but is not ghosted.
    pub reference_value: Vec<f64>,
}
```

Register the type once, then write the method's kernels as ordinary scheduled
systems. SOIL carries every registered row through insertion, migration,
permutation, ghost construction, communication, and restart.

The field attributes have precise meanings:

- `#[forward]` copies owner values to ghost particles.
- `#[reverse]` reduces ghost contributions back to the owning particle.
- `#[zero]` resets an accumulator before it is reused.
- An unmarked field remains owner-local but still migrates with its particle.

The complete structural invariants and per-step ordering are summarized in the
[`soil_core`](crates/soil_core/README.md) and
[`soil_derive`](crates/soil_derive/README.md) crate references.

## Build a particle method, not another infrastructure stack

A method implemented on SOIL typically supplies:

1. Its additional `AtomData` columns.
2. A force, interaction, or update system over particles and neighbors.
3. A plugin that registers that state and schedules those systems.
4. An integrator and output plugins appropriate to the method.

SOIL supplies the particle store, domain, communication, and neighbor machinery
underneath those choices. [`examples/minimal_sim`](examples/minimal_sim/README.md)
shows the bare plugin assembly, while [`lj_dimer`](examples/lj_dimer/README.md)
adds a complete non-DEM interaction law.

## Local and long-range spatial infrastructure

The mature path is for cutoff-bounded interactions: particles are decomposed by
space, borders produce ghosts, and bin-based neighbor lists find nearby pairs.

`soil_core::Octree` also supplies a physics-agnostic spatial hierarchy with a
near/far partition and monopole aggregation. It is an infrastructure primitive,
not a complete long-range solver. The force kernel, error control, higher
multipoles, and time integration remain the responsibility of the method above
SOIL. The checked `tree_farfield_proof` and `orbit_nbody_gravity` examples show
how to use the tree without moving gravity into the substrate.

## Evidence and boundaries

SOIL's tests focus on invariants that every particle method depends on:

- registered columns remain row-aligned through insertion, sorting, migration,
  and removal;
- neighbor lists match direct reference searches, including triclinic cases;
- forward and reverse communication preserve ownership semantics across ranks;
- periodic, shrink-wrapped, and deforming domains update consistently;
- restart and output paths retain registered particle state.

[`lj_dimer`](examples/lj_dimer/README.md) is a compact end-to-end demonstration
outside DEM: the neighbor list and ghost communication carry a Lennard-Jones
pair across a periodic boundary, and velocity Verlet relaxes it to the analytical
minimum `r = 2^(1/6)σ`, `U = -ε`.

These checks validate substrate behavior. They do not validate an arbitrary
particle method simply because it uses SOIL.

SOIL is not:

- a DEM, molecular-dynamics, SPH, or peridynamics model;
- a mesh or Eulerian field framework;
- a force-law library;
- a promise that every possible particle data structure fits `AtomData`.

## Install and run

You need stable [Rust](https://rustup.rs/). SOIL pulls GRASS during the build.
The quickest start uses the single-process backend with double precision:

```console
git clone https://github.com/SueHeir/soil
cd soil
cargo run --release --example minimal_sim \
  --no-default-features --features precision-double
cargo run --release --example lj_dimer \
  --no-default-features --features precision-double
```

The default features enable MPI and double precision. Use the defaults when an
MPI development toolchain is installed and distributed execution is required.

To build a method on top, depend on the substrate crates you need and assemble
the base plugins (`AtomPlugin`, `DomainPlugin`, `NeighborPlugin`, and
`CommunicationPlugin`). Start from
[`examples/minimal_sim`](examples/minimal_sim/README.md) and add method-specific
state and force systems as a plugin.

## Crate map

| Crate | Role |
|---|---|
| [`soil_core`](crates/soil_core/README.md) | base particle store, `AtomData`, domains, communication, neighbors, regions, groups, and octree |
| [`soil_derive`](crates/soil_derive/README.md) | `#[derive(AtomData)]` and its communication attributes |
| [`soil_verlet`](crates/soil_verlet/README.md) | velocity-Verlet translational integration |
| [`soil_print`](crates/soil_print/README.md) | thermo, VTP and dump output, and restart files |
| [`soil_deform`](crates/soil_deform/README.md) | box deformation, target-size loading, and Lees–Edwards shear |
| [`soil_fixes`](crates/soil_fixes/README.md) | method-neutral position and velocity constraints |

Configuration and runnable guidance live beside the examples. Public API,
parallelism, and model-boundary details live in the crate READMEs and Rust
documentation.

## Ecosystem

The core repositories form a one-way dependency stack:

```text
GRASS   scientific application framework
  └── SOIL   distributed particle infrastructure
        └── DIRT   discrete-element-method physics and applications
```

- [GRASS](https://github.com/SueHeir/grass) provides Apps, scheduling, plugins,
  lifecycle, configuration, and communication abstractions.
- [DIRT](https://github.com/SueHeir/dirt) is the complete DEM implementation
  built on SOIL's particle machinery.

Because SOIL keeps scientific interactions out of the substrate, a method can
also expose its typed state to a larger application. Such composition is an
optional use of the architecture, not SOIL's primary purpose.

## License

MIT OR Apache-2.0
