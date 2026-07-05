# Does the Seam Hold? — A Peridynamics Worked Check

SOIL claims to be a *method-agnostic* substrate: the same `Atom`, domain
decomposition, comm, and neighbor machinery that serves DEM today should serve
any other Lagrangian particle method with **no change to the substrate**. That
claim is only worth as much as the evidence behind it. This page is the evidence:
a paper check that peridynamics maps cleanly onto the
[`AtomData` contract](./atomdata-contract.md), followed by a runnable, non-DEM
method that already rides the substrate and validates to machine precision.

## The claim to test

Peridynamics is a bond-based, Lagrangian reformulation of continuum mechanics:
each material point interacts with every other point inside a finite *horizon*
through pairwise (or state-based) force interactions, and bonds accumulate damage
until they break. Structurally that is a short-range, neighbor-driven particle
method — so if the SOIL tiering is real, a peridynamics tier should register a
few `AtomData` columns and get parallel decomposition, halo exchange, and
migration for free, adding **no new substrate primitive**.

## The mapping

Here is the per-particle state a peridynamics tier needs, expressed as a single
`AtomData` struct:

```rust
use soil_derive::AtomData;

#[derive(AtomData)]
pub struct PeriAtom {
    /// Scalar damage φ ∈ [0,1]; neighbors need it to compute the bond force.
    #[forward]
    pub damage: Vec<f64>,

    /// Pairwise force-state contributions summed from ghost bonds, reset/step.
    #[reverse]
    #[zero]
    pub force_state: Vec<[f64; 3]>,

    /// Weighted volume / dilatation accumulator (state-based PD), reset/step.
    #[reverse]
    #[zero]
    pub dilatation: Vec<f64>,

    /// Reference (undeformed) position — migrates with the atom, never a ghost.
    pub ref_pos: Vec<[f64; 3]>,
}
```

Every quantity peridynamics carries maps onto an attribute that already exists:

| peridynamics state | who needs it | attribute |
|---|---|---|
| damage `φ` | a *neighbor* reads it to weight the bond force | `#[forward]` — replicated onto ghosts |
| force-state contribution | computed on a *ghost* bond, owed to the owner | `#[reverse]` + `#[zero]` — ghost-summed, reset each step |
| dilatation (state-based PD) | accumulated from ghost bonds | `#[reverse]` + `#[zero]` |
| reference position `X` | follows the atom across subdomains, never ghosts | *no attribute* — migrates but does not replicate |

`damage` is replicated state a neighbor reads; the force-state and dilatation
accumulators are ghost-summed and reset each step; the reference configuration
migrates with the atom but never crosses the ghost boundary. **No new substrate
primitive is required** — which is exactly the point. If peridynamics had needed
a communication pattern that `#[forward]` / `#[reverse]` / `#[zero]` cannot
express, that would be a hole in the substrate; it does not.

## The one honest caveat

Peridynamics does need one thing beyond the `AtomData` contract: a *bond/horizon*
neighbor list with a larger, fixed cutoff (the horizon), rather than DEM's
contact-radius list that shrinks and grows with particle overlap. That is a
**neighbor-list strategy** configured per-physics on the SOIL side — a knob the
substrate already exposes — not a change to the `AtomData` contract or a new
comm primitive. The seam holds; the horizon is a configuration choice, not a
missing capability.

## The runnable half of the proof

The peridynamics check above is a paper argument. The substrate also carries a
*second, non-DEM particle method today* that you can run and check yourself:
[`examples/lj_dimer/main.rs`](https://github.com/SueHeir/soil/tree/main/examples/lj_dimer/main.rs)
is a Lennard-Jones molecular-dynamics dimer. It uses **no DEM code** — a pairwise
LJ potential and generic viscous drag, no contact model, rotation, clumps, or
friction — and it drives the full substrate:

- the **neighbor list** (`Neighbor::pairs`) finds the interacting pair,
- **ghost comm** carries the partner across a periodic face — the dimer straddles
  the x-boundary, so the interaction is *ghost-mediated*: forward comm replicates
  the ghost position, reverse comm sums the ghost's force and energy back onto the
  owner,
- a registered **`#[reverse] #[zero]` `LjEnergy` column** reverse-sums the
  per-atom potential energy — the same mechanism DEM uses for torque, applied to
  a quantity that has nothing to do with DEM,
- **`VelocityVerletPlugin`** integrates the motion.

Relaxed to force balance, it reproduces both Lennard-Jones closed forms exactly:

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

The relaxed separation equals the LJ well minimum `r_min = 2^(1/6)σ` and the
total potential energy — reconstructed from the reverse-communicated column —
equals the well depth `U(r_min) = −ε`, both to machine precision. Between the
peridynamics mapping and a running LJ-MD example, the case is that the
GRASS → SOIL → physics tiering is real, not a relabeling: DEM is one tenant of
the substrate, not its definition.
