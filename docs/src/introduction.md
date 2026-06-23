# Introduction

**SOIL** — the Substrate for Off-lattice Interacting Lagrangians — is the
method-agnostic particle layer in a three-tier stack. It owns everything *every*
particle method needs regardless of physics, and knows nothing about the physics
itself.

```
GRASS    framework: App, Plugin, Scheduler, IO, coupling      (no particles)
  └─ SOIL   substrate: Atom, domain decomposition, comm, neighbor lists   (no physics)
       └─ DIRT (DEM) / SPH / peridynamics / …   physics: forces, bonds, walls
```

- **[GRASS](https://github.com/SueHeir/grass)** is the framework: `App`,
  `Plugin`, the scheduler, TOML config, and MPI coupling. See the
  [GRASS book](https://sueheir.github.io/grass).
- **SOIL** is the substrate: the base `Atom`, domain decomposition, ghost/halo
  communication, atom migration, and neighbor-list construction.
- **[DIRT](https://github.com/SueHeir/dirt)** is one physics tier built on
  SOIL (Discrete Element Method). See the [DIRT book](https://sueheir.github.io/dirt).

## Who this book is for

If you want to write your **own** particle method — your own force law, an SPH
kernel, a peridynamic bond model — and you don't want to write domain
decomposition, halo exchange, atom migration, and neighbor lists yourself, this
book is for you. SOIL gives you all of that. You supply the physics.

The deal is simple, and it is the whole point of the substrate:

> Declare your per-particle state as an `AtomData` column and tell SOIL who needs
> each field. The substrate then carries that state through every migration,
> ghost exchange, permutation, and restart — automatically. **If your state fits
> the `AtomData` shape, you write no communication code.**

## The shape of the book

- **[What SOIL Owns](./substrate/what-soil-owns.md)** — the base `Atom` and the
  machinery the substrate is responsible for.
- **[The AtomData Contract](./substrate/atomdata-contract.md)** — the one stable
  interface between the substrate and any physics that rides it. This is the
  reference you will keep coming back to.
- **[Write Your Own Particle Physics](./tutorial/write-your-own-physics.md)** —
  a step-by-step tutorial building a minimal physics tier on the substrate.

## Why a substrate at all

The seam between SOIL and a physics tier is a single contract — `AtomData`
registration. Because that contract is physics-agnostic, the same substrate
carries DEM today and could carry SPH or peridynamics tomorrow with **no change
to SOIL**. The [AtomData contract](./substrate/atomdata-contract.md) includes a
worked check that peridynamics drops onto the substrate with no new primitive —
the evidence that the tiering is real and not a relabeling.
