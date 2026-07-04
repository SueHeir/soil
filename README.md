# SOIL

**Write your own particle method — a force law, an SPH kernel, a peridynamic
bond model — without hand-writing domain decomposition, halo exchange, atom
migration, or neighbor lists. Declare your per-particle state once, and SOIL
carries it through all of them.**

SOIL — the *Substrate for Off-lattice Interacting Lagrangians* — is the
method-agnostic particle tier in the **GRASS → SOIL → physics** stack. It owns
the base `Atom`, domain decomposition, ghost/halo communication, atom migration,
and neighbor-list construction — everything *every* particle method needs
regardless of physics — and knows nothing about contact forces, bonds, or
damage. It rides [GRASS](https://github.com/SueHeir/grass) for the
`App`/scheduler/coupling machinery, and [DIRT](https://github.com/SueHeir/dirt)
(granular DEM) is the physics tier that proves it.

## A particle method is one `AtomData` column plus a force function

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
precision.** [`crates/soil_verlet/examples/lj_dimer.rs`](crates/soil_verlet/examples/lj_dimer.rs)
is a Lennard-Jones **molecular-dynamics** dimer: a pairwise potential, no contact
model, no rotation, no clumps, no friction. It exercises the whole substrate —
the neighbor list finds the pair, ghost comm carries the partner across a
periodic face (the dimer straddles the boundary, so the interaction is
*ghost-mediated*), a registered `#[reverse] #[zero]` `LjEnergy` column
reverse-sums the potential energy back to its owner, and `VelocityVerletPlugin`
integrates the motion. Relaxed to force balance, it reproduces both closed forms
exactly:

```text
$ cargo run --release -p soil_verlet --example lj_dimer
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

Reach for SOIL when you're building a **short-range, off-lattice particle
(Lagrangian) method** — a force law, an SPH kernel, a peridynamic bond model —
that needs parallel domain decomposition, halo exchange, migration, and neighbor
lists but shouldn't have to hand-write any of them. You bring the physics; SOIL
brings the parallel plumbing.

It is deliberately *not* two things. It is not a mesh/Eulerian framework: a
grid-based finite-volume or finite-element solver, whose state is a field over
cells rather than a cloud of migrating particles, belongs on a mesh substrate on
the same [GRASS](https://github.com/SueHeir/grass) scheduler — not here. And it
owns no physics: no contact model, material, or force law lives in the substrate,
because that is the tier *you* write on top. [DIRT](https://github.com/SueHeir/dirt)
is the worked example of a full one. If your interactions are long-range (an Ewald
or particle-mesh sum over all pairs) rather than cutoff-bounded, the neighbor
substrate is a starting point, not the whole answer — that solver is unwritten.

## The stack

SOIL is the middle tier of a three-repo stack; each tier depends only on the ones
below it and knows nothing about the tiers above:

```
GRASS    framework: App, Plugin, Scheduler, IO, coupling      (no particles)
  └─ SOIL   substrate: Atom, domain decomposition, comm, neighbor lists   (no physics)
       └─ DIRT (DEM) / SPH / peridynamics / …   physics: forces, bonds, walls
```

## How the three fit together

GRASS gives you the `App`/scheduler/coupling; SOIL turns that into a parallel
particle substrate via one `AtomData` contract; DIRT is the proof that a full
LAMMPS-validated physics tier rides it — and the same seams are open for SPH,
peridynamics, or your own method.

One line per tier, matching how each repo describes itself:

- **[GRASS](https://github.com/SueHeir/grass)** — Build solvers as composable
  plugins instead of a hand-rolled main loop — explicit, time-stepping methods —
  and couple several together, in-process or across MPI. The framework knows
  nothing about particles or physics, and is not proven for implicit global
  solvers (FEM, spectral, Newton-Krylov).
- **[SOIL](https://github.com/SueHeir/soil)** — Write your own particle method
  without hand-writing domain decomposition, halo exchange, migration, and
  neighbor lists — declare your state once, SOIL carries it through all of it.
- **[DIRT](https://github.com/SueHeir/dirt)** — A Rust granular-DEM engine that
  resolves every contact individually — cross-checked against LAMMPS and
  closed-form theory.

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
cargo run --release -p soil_verlet --example lj_dimer
```

To build your own tier on top, depend on the substrate crates and add the four
plugins (`AtomPlugin`, `DomainPlugin`, `NeighborPlugin`, `CommunicationPlugin`) —
[`crates/soil_core/examples/minimal_sim.rs`](crates/soil_core/examples/minimal_sim.rs)
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
- **[Write Your Own Particle Physics](docs/src/tutorial/write-your-own-physics.md)** —
  the step-by-step tutorial. Built on [grass](https://github.com/SueHeir/grass),
  consumed by [dirt](https://github.com/SueHeir/dirt) (DEM).

## License

MIT OR Apache-2.0
