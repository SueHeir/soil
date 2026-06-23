# Write Your Own Particle Physics

This tutorial builds a minimal physics tier on the substrate: a **soft-sphere
pairwise repulsion**. It is intentionally simpler than DIRT's Hertz–Mindlin
contact, but it uses the *same* real substrate API — so when you graduate to a
serious force law, nothing about the plumbing changes.

By the end you will have:

1. Declared per-particle state as an `AtomData` column.
2. Registered it so the substrate carries it through migration and ghosting.
3. Written a force system that iterates the substrate's neighbor list.
4. Wired it all together as a `Plugin`.

You should read [The AtomData Contract](../substrate/atomdata-contract.md) first —
this tutorial assumes the four lifecycle hooks and the field attributes.

## 1. Declare your per-particle state

The base `Atom` already carries `pos`, `vel`, `force`, and `mass`. Our force law
needs one extra per-particle quantity: a radius. We declare it as an `AtomData`
struct.

```rust
use soil_derive::AtomData;

#[derive(AtomData)]
pub struct SoftAtom {
    /// Particle radius. A neighbor needs to know our radius to compute the
    /// overlap, so it must be replicated onto ghosts: `#[forward]`.
    #[forward]
    pub radius: Vec<f64>,
}
```

Why `#[forward]`? When particle *i* on our rank touches particle *j* that lives
on another rank, *j* shows up here as a **ghost**. To compute the overlap we need
*j*'s radius on our rank — so radius is read-only state a neighbor needs, which
is exactly what `#[forward]` means. (See
[Choosing Field Attributes](./field-attributes.md) for the full decision tree.)

We don't need a `#[reverse]` column of our own: the force we compute lands in the
base `Atom.force`, which the substrate already reverse-accumulates and zeros for
us.

## 2. Write the force system

A *system* is an ordinary function whose arguments are typed resources; the
scheduler injects them. Here is the real shape, taken straight from how DIRT
writes its contact force:

```rust
use grass_app::prelude::*;        // Res, ResMut
use soil_core::{Atom, Neighbor, AtomDataRegistry};

pub fn soft_repulsion_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
) {
    // Stiffness of the linear repulsion [N/m]. In a real tier this comes from a
    // material table; hard-coded here to keep the example self-contained.
    const K: f64 = 1.0e5;

    let dem = registry.expect::<SoftAtom>("soft_repulsion_force");
    let nlocal = atoms.nlocal as usize;
    let newton = neighbor.newton;

    // The substrate built this pair list for us — binned, ghost-aware.
    for (i, j) in neighbor.pairs(nlocal) {
        let ri = dem.radius[i];
        let rj = dem.radius[j];
        let sum_r = ri + rj;

        let dx = atoms.pos[j][0] - atoms.pos[i][0];
        let dy = atoms.pos[j][1] - atoms.pos[i][1];
        let dz = atoms.pos[j][2] - atoms.pos[i][2];
        let dist_sq = dx * dx + dy * dy + dz * dz;

        // No overlap → no force.
        if dist_sq >= sum_r * sum_r || dist_sq == 0.0 {
            continue;
        }

        let dist = dist_sq.sqrt();
        let overlap = sum_r - dist;

        // Linear spring: f = K·overlap along the line of centers, pushing apart.
        let inv_dist = 1.0 / dist;
        let fmag = K * overlap;
        let fx = -fmag * dx * inv_dist;
        let fy = -fmag * dy * inv_dist;
        let fz = -fmag * dz * inv_dist;

        // Newton's third law: i gets +f, j gets -f.
        atoms.force[i][0] += fx;
        atoms.force[i][1] += fy;
        atoms.force[i][2] += fz;

        // With Newton's-third-law optimization on, j may be a ghost; the
        // substrate's reverse comm sums that contribution back to j's owner.
        if newton || j < nlocal {
            atoms.force[j][0] -= fx;
            atoms.force[j][1] -= fy;
            atoms.force[j][2] -= fz;
        }
    }
}
```

That is the entire physics. Notice what you did **not** write: no MPI calls, no
ghost packing, no neighbor binning. `neighbor.pairs(nlocal)` hands you the pairs;
writing into `atoms.force[j]` for a ghost `j` is made correct by the substrate's
reverse communication. **The substrate moved the data; you wrote the force law.**

## 3. Register and schedule it as a plugin

A `Plugin` wires your state and systems into the app during its build phase:

```rust
use grass_app::prelude::*;     // App, Plugin
use soil_core::{register_atom_data, ParticleSimScheduleSet};

pub struct SoftSpherePlugin;

impl Plugin for SoftSpherePlugin {
    fn build(&self, app: &mut App) {
        // Register the column once. From now on the substrate carries `radius`
        // through every migration, ghost exchange, permutation, and restart.
        // The `AtomData` derive does not generate a constructor, so build the
        // value directly (its one field is a `Vec`).
        register_atom_data!(app, SoftAtom { radius: Vec::new() });

        // Run the force law in the force phase of the step, after ghosts are
        // fresh and before integration.
        app.add_update_system(
            soft_repulsion_force,
            ParticleSimScheduleSet::Force,
        );
    }
}
```

## 4. Run it

Combine your plugin with the substrate's infrastructure and an integrator:

```rust
use grass_app::prelude::*;

fn main() {
    let mut app = App::new();
    // `CorePlugins` is a convenience plugin group that lives downstream in
    // `dirt_core` (NOT in soil_core); it bundles input/comm/domain/neighbor/run/
    // output. On the substrate alone you would instead add `AtomPlugin`,
    // `DomainPlugin`, `NeighborPlugin`, `CommunicationPlugin`, `InputPlugin`, and
    // `RunPlugin` yourself (see `soil_core/examples/minimal_sim.rs`).
    app.add_plugins(CorePlugins)                 // config, comm, domain decomp, neighbor lists
       .add_plugins(VelocityVerletPlugin::new()) // velocity-Verlet integration (soil_verlet)
       .add_plugins(SoftSpherePlugin);           // ← your physics
    app.start();
}
```

That is a complete particle code. Everything method-agnostic — decomposition,
halo exchange, migration, neighboring, integration, I/O — came from the
substrate and framework. You wrote one `AtomData` struct, one force function, and
one plugin.

## Where the per-step ordering comes from

Recall the canonical ordering from the contract:

```
zero(#[zero])  →  forward(#[forward])  →  compute forces  →  reverse(#[reverse])  →  integrate
```

You scheduled `soft_repulsion_force` into the **force** stage. The substrate runs
`forward` (replicating `radius` onto ghosts) before it, and `reverse` (summing
ghost `force` back to owners) after it — automatically, because you classified
your fields correctly. Get the attributes wrong and parallel runs diverge
silently; that is why the [contract](../substrate/atomdata-contract.md) is worth
reading twice.

> **Note on exact names.** `ParticleSimScheduleSet::Force` and
> `VelocityVerletPlugin` live in `soil_core` / `soil_verlet`; `CorePlugins` lives
> downstream in `dirt_core`. Confirm them against your checkout, since plugin-group
> names are the most likely thing to have been renamed.
