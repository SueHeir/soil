# SOIL

<!-- disclaimer-banner -->
> **Research-software status:** This ecosystem is AI-authored and under active evaluation. GRASS, SOIL, and DIRT are the core architecture and DEM implementation; repositories prefixed `dev_` are experimental method demonstrations outside the author's domain expertise. Treat claims according to their linked evidence and documented limitations. See [DISCLAIMER.md](DISCLAIMER.md).
<!-- /disclaimer-banner -->


**Write the particle method you need without making it a closed application.
Declare per-particle state once; SOIL carries it through neighbors, migration,
restart, and MPI, while GRASS keeps the resulting solver composable with other
codes.**

A new particle method should begin with its particles and interactions — not a
private implementation of domain decomposition, ghost exchange, atom migration,
neighbor lists, output, and lifecycle.

SOIL — the *Substrate for Off-lattice Interacting Lagrangians* — is the shared
particle substrate for GRASS solvers. A physics code declares its per-particle
state through `AtomData` and writes its kernels as scheduled systems. SOIL handles
the reusable particle infrastructure, and because that infrastructure uses the
same GRASS resource, plugin, and schedule model as the physics, the completed
solver can run alone or participate in a coupled application.

The point is not merely to save communication code once. It is to prevent each
particle method from becoming another isolated executable with a different
internal world.

## From one particle solver to a coupled participant

SOIL owns the particle data model and its migration/halo rules. GRASS holds the
registered `AtomData` resources and schedules their systems; it does not define
what a particle is. A coupling plugin can therefore mount an unchanged SOIL
solver beside a FIELD solver and translate between their public resources.

The local teaching case is
[`hello_dem_cfd`](https://github.com/SueHeir/dev_couple_dem_cfd/tree/main/examples/hello_dem_cfd).
The distributed counterpart is the runnable
[`routed_3x2`](https://github.com/SueHeir/dev_couple_dem_cfd/blob/main/crates/dem_cfd/tests/routed_3x2.rs)
test: one TOML topology assigns 3 DEM and 2 CFD ranks to the same binary, while
the coupling layer uses particle positions and FIELD ownership to address each
record. GRASS transports those records but knows neither particles nor meshes.

Cross-solver access stays on the parent between child ticks or exported
scheduler seams. Systems inside a SOIL child continue to use ordinary
`Res`/`ResMut` for their own resources.

## One declaration is the parallel contract

You extend a particle by declaring a struct of per-particle columns and deriving
`AtomData`. Three field attributes tell the substrate *who needs each field*, and
that is the entire parallel contract — you never touch a send or a receive:

```rust
use soil_derive::AtomData;

#[derive(AtomData)]
pub struct DemAtom {
    /// Read-only state a *neighbor* needs — replicated onto ghosts each step.
    #[forward]
    pub omega: Vec<[f64; 3]>,

    /// A contribution computed *on a ghost* — summed back onto the owner …
    #[reverse]
    #[zero]                  // … and reset to zero at the start of every step.
    pub torque: Vec<[f64; 3]>,

    /// Migrates with the atom across subdomains, but never crosses to a ghost.
    pub radius: Vec<f64>,
}
```

Register that struct once in a plugin's `build()`, write your force law as an
ordinary function over the neighbor list, and add a `soil_verlet` integrator. The
substrate does the rest.

> **If your per-particle state fits the `AtomData` shape, the substrate moves it
> for you and you write no communication code.** `#[forward]` replicates a field
> onto ghosts, `#[reverse]` sums a ghost's contribution back to its owner,
> `#[zero]` clears an accumulator each step — and SOIL carries every column
> through migration, ghost exchange, permutation, and restart automatically,
> serial or across MPI ranks.

The full semantics, invariants, and per-step ordering are in
[The AtomData Contract](docs/SOIL_ATOMDATA_CONTRACT.md).

## When SOIL is worth using

Use SOIL when you are building an off-lattice particle method that may share
infrastructure or couple with other simulation components. If the method will
remain a small standalone calculation forever, a purpose-built loop may be
simpler. SOIL is most valuable when the solver is intended to become one
participant in a larger research-software ecosystem.

DIRT demonstrates a full DEM tier on this substrate. The MD, SPH, and
peridynamics examples test whether the same particle contract survives methods
with different state and interaction semantics. Their purpose here is
architectural: to show that the substrate boundary is not secretly DEM-specific.

## One substrate, many methods

SOIL's claim is that a *single* physics-agnostic contract carries genuinely
different particle methods — not just DEM, which it was first built for. Two
pieces of evidence back that up: one on paper, one you can run.

**Peridynamics drops on with no new substrate primitive.** Peridynamics is
bond-based and Lagrangian, so its per-particle state should map onto the existing
attributes with nothing added to the substrate. It does: scalar `damage` is
replicated state a neighbor reads (`#[forward]`); the force-state and dilatation
accumulators are summed from ghost bonds (`#[reverse]` + `#[zero]`); the
reference configuration migrates with the atom but never ghosts (no attribute).
The one thing peridynamics needs beyond the contract is a *bond/horizon* neighbor
list with a larger fixed cutoff — a neighbor-list strategy configured per-physics
on the SOIL side, **not** a change to `AtomData`. The full worked check is
[Does the Seam Hold? — A Peridynamics Worked Check](docs/src/substrate/peridynamics-check.md).

**A non-DEM method already rides the substrate — and validates to machine
precision.** [`examples/lj_dimer/main.rs`](examples/lj_dimer/main.rs)
is a Lennard-Jones **molecular-dynamics** dimer: a pairwise potential, no contact
model, no rotation, no clumps, no friction. It exercises the whole substrate —
the neighbor list finds the pair, ghost comm carries the partner across a
periodic face (the dimer straddles the boundary, so the interaction is
*ghost-mediated*), a registered `#[reverse] #[zero]` `LjEnergy` column
reverse-sums the potential energy back to its owner, and `VelocityVerletPlugin`
integrates the motion. Relaxed to force balance, it reproduces both closed forms
exactly:

```text
$ cargo run --release --example lj_dimer
  ghosts present at end : 2  (pair is ghost-mediated across the periodic face)
  relaxed separation    : r = 1.122462048 σ
  analytic well minimum : r_min = 2^(1/6) σ = 1.122462048
  |r − r_min|           : 0.000e0
  total potential energy: U = -1.000000000 ε   (from reverse-comm LjEnergy column)
  analytic well depth   : U(r_min) = −ε = -1.000000000
  |U − (−ε)|            : 0.000e0
```

`r = 2^(1/6)σ` and `U = −ε` are the exact Lennard-Jones well minimum and depth.
DEM is not special: the same `AtomData` mechanism that moves DEM's torque moves
MD's per-atom energy.

## When to use it

Reach for SOIL when you're building an **off-lattice particle (Lagrangian)
method** — a force law, an SPH kernel, a peridynamic bond model, or a long-range
treecode — that needs shared particle storage and substrate machinery instead of
hand-written infrastructure. For local-reach methods, SOIL gives you parallel
domain decomposition, halo exchange, migration, and neighbor lists. For
long-range particle methods, `soil_core::Octree` supplies the agnostic spatial
hierarchy and near/far partition; the force kernel, multipole expansion, and
time integration still belong in the physics tier above it.

It is deliberately *not* two things. It is not a mesh/Eulerian framework: a
grid-based finite-volume or finite-element solver, whose state is a field over
cells rather than a cloud of migrating particles, belongs on a mesh substrate on
the same [GRASS](https://github.com/SueHeir/grass) scheduler — not here. And it
owns no physics: no contact model, material, or force law lives in the substrate,
because that is the tier *you* write on top. [DIRT](https://github.com/SueHeir/dirt)
is the worked example of a full one. If your interactions are long-range rather
than cutoff-bounded, start from the tree primitive and keep the physics out of
the substrate: the checked examples below show a generic 1/r treecode proof and
a Barnes-Hut gravity demonstrator built that way.

## The stack

SOIL is the middle tier of a three-repo stack; each tier depends only on the ones
below it and knows nothing about the tiers above:

```
GRASS    framework: App, Plugin, Scheduler, IO, coupling      (no particles)
  └─ SOIL   substrate: Atom, domain decomposition, comm, neighbor lists   (no physics)
       └─ DIRT (DEM) / SPH / peridynamics / …   physics: forces, bonds, walls
```

SOIL covers the local-reach particle substrate: particles interact through
bounded neighbor lists and halo exchange. It also now carries a first long-range
primitive: `soil_core::Octree`, a spatial tree with near/far partitioning and
monopole aggregation. That does not make a gravity, electrostatics, or vortex
solver part of SOIL; it means the shared particle/tree structure lives in the
substrate, while the kernel and method stay above it. Reaching long-range /
global solving is a roadmap axis grown *inside* each substrate — SOIL for
particles, FIELD for meshes — and this tree is an early step on that axis, not a
finished capability.

## How the three fit together

GRASS gives you the `App`/scheduler/coupling; SOIL turns that into a parallel
particle substrate via one `AtomData` contract; DIRT is the proof that a full
LAMMPS-validated physics tier rides it — and the same seams are open for SPH,
peridynamics, or your own method.

One line per tier, worded identically wherever these three repos describe
themselves:

- **[GRASS](https://github.com/SueHeir/grass)** — Build solvers as composable
  plugins instead of a hand-rolled main loop — explicit time-stepping or a
  single implicit global solve, particles or a mesh — and couple several
  together, in-process or across MPI.
- **[SOIL](https://github.com/SueHeir/soil)** — Write your own particle method
  without hand-writing domain decomposition, halo exchange, migration, and
  neighbor lists — declare your state once, SOIL carries it through all of it.
- **[DIRT](https://github.com/SueHeir/dirt)** — A Rust granular-DEM engine that
  resolves every contact individually — cross-checked against LAMMPS and
  closed-form theory.

Built on [GRASS](https://github.com/SueHeir/grass). Consumed by
[DIRT](https://github.com/SueHeir/dirt) (DEM, validated) and by
development-stage sibling physics tiers that ride the same `AtomData` contract
as demonstrations that the substrate is method-agnostic:
`dev_soil_peri` (bond-based peridynamics — a PD point cloud is a soil atom set,
its horizon is a neighbor cutoff, and its PD family is a `BondStore`) and
`dev_soil_sph` (granular SPH — kernels plus a `mu(I)` constitutive update over
the same neighbor machinery). These `dev_` tiers are not yet peer-reviewed; they
exist to show that one particle substrate carries more than one method.

**Where to start:** to *run* granular simulations, start at
[DIRT](https://github.com/SueHeir/dirt), the batteries-included physics tier; to
*write your own* particle method or solver, start at
[SOIL](https://github.com/SueHeir/soil) (the particle substrate) or
[GRASS](https://github.com/SueHeir/grass) (the framework). The full walkthrough
of how the tiers compose — one timestep end to end, and where the seams are — is
the canonical [How the stack fits together](https://sueheir.github.io/grass/stack/how-the-stack-fits-together.html)
page in the GRASS book.

## Install

You need **Rust** (stable, 2021 edition or newer; [rustup.rs](https://rustup.rs)).
You do *not* need to check out GRASS separately — SOIL pulls it during the build.
The fastest way to watch the substrate carry a real method is the LJ-MD dimer:

```bash
git clone https://github.com/SueHeir/soil
cd soil
cargo run --release --example lj_dimer
```

To build your own tier on top, depend on the substrate crates and add the four
plugins (`AtomPlugin`, `DomainPlugin`, `NeighborPlugin`, `CommunicationPlugin`) —
[`examples/minimal_sim/main.rs`](examples/minimal_sim/main.rs)
shows the bare assembly, and the mdBook tutorial
[Write Your Own Particle Physics](docs/src/tutorial/write-your-own-physics.md)
builds a minimal method step by step.

## Crate map

`soil_core` is the substrate proper — depend on it and you get the base `Atom`,
the `AtomData` registry, and all the parallel machinery. The rest are optional
tiers you reach for as needed:

| crate | role |
|---|---|
| [`soil_core`](crates/soil_core/README.md) | base `Atom` + `AtomData` registry, comm, domain decomposition, neighbor, regions, groups |
| [`soil_derive`](crates/soil_derive/README.md) | `#[derive(AtomData)]` proc macro (`#[forward]` / `#[reverse]` / `#[zero]`) |
| [`soil_verlet`](crates/soil_verlet/README.md) | velocity-Verlet translational integration |
| [`soil_print`](crates/soil_print/README.md) | thermo output, dump files (CSV/binary/LAMMPS/VTP), restart |
| [`soil_deform`](crates/soil_deform/README.md) | box deformation (strain rate, velocity, target size) + Lees–Edwards `xy` shear |
| [`soil_fixes`](crates/soil_fixes/README.md) | method-agnostic position constraint `pin` on base `Atom` state (DEM fixes such as `freeze` / velocity damping live in [dirt_fixes](https://github.com/SueHeir/dirt)) |

## Next

- **[The AtomData Contract](docs/SOIL_ATOMDATA_CONTRACT.md)** — the one stable
  interface every physics tier honors; read it before you write a method.
- **[Does the Seam Hold? — A Peridynamics Worked Check](docs/src/substrate/peridynamics-check.md)** —
  the evidence that the tiering is real, not a relabeling.
- **`cargo run --release --example tree_farfield_proof`** — the
  far-field tree primitive checked against direct O(n^2) sums without adding a
  force law to the substrate.
- **`cargo run --release --example orbit_nbody_gravity`** — a
  Barnes-Hut gravity demonstrator that keeps gravity in the example and uses
  SOIL only for the agnostic tree.
- **[Write Your Own Particle Physics](docs/src/tutorial/write-your-own-physics.md)** —
  the step-by-step tutorial. Built on [grass](https://github.com/SueHeir/grass),
  consumed by [dirt](https://github.com/SueHeir/dirt) (DEM).

## License

MIT OR Apache-2.0
