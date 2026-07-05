//! Minimal serial SOIL simulation — no MPI, no physics.
//!
//! Assembles the four substrate plugins every particle code needs
//! ([`AtomPlugin`], [`DomainPlugin`], [`NeighborPlugin`], [`CommunicationPlugin`]
//! with the single-process comm backend), seeds a few atoms by hand, steps the
//! loop, and walks the neighbor pair list the substrate built.
//!
//! There is no force law and no integrator here: this example exists to show the
//! *assembly* — how the method-agnostic pieces snap together — and to give you a
//! runnable starting point. A real tier adds an `AtomData` column, a force
//! system, and an integrator (`soil_verlet`); see the mdBook tutorial
//! "Write Your Own Particle Physics".
//!
//! Run with:
//!
//! ```sh
//! cargo run --example minimal_sim
//! ```

use std::any::TypeId;

use grass_app::prelude::*;
use soil_core::{
    Atom, AtomPlugin, CommunicationPlugin, Domain, DomainPlugin, Neighbor, NeighborPlugin,
};

fn main() {
    let mut app = App::new();

    // The four substrate plugins. `CommunicationPlugin` installs the
    // single-process comm backend by default (no `[comm]` config needed), so this
    // runs serially with no MPI. Each plugin's TOML section falls back to a
    // sensible default when absent, so no config file is required here.
    app.add_plugins(AtomPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(NeighborPlugin)
        .add_plugins(CommunicationPlugin);

    // `prepare()` installs the scheduler manager, organizes systems, and runs the
    // setup phase (populating the default box [0,1]^3 periodic and the neighbor
    // parameters), then moves the loop into its running state.
    app.prepare();

    // Seed a few atoms inside the default unit box. The framework stores
    // resources as `RefCell<Box<dyn Any>>`, so we borrow the `Atom` cell and
    // downcast it. `push_test_atom` appends one local atom with zero
    // velocity/force; the third argument is the per-atom cutoff radius that
    // drives neighbor detection, the fourth is mass.
    {
        let cell = app
            .get_mut_resource(TypeId::of::<Atom>())
            .expect("AtomPlugin registers the Atom resource");
        let mut binder = cell.borrow_mut();
        let atoms = binder.downcast_mut::<Atom>().unwrap();
        atoms.push_test_atom(0, [0.25, 0.5, 0.5], 0.3, 1.0);
        atoms.push_test_atom(1, [0.55, 0.5, 0.5], 0.3, 1.0);
        atoms.push_test_atom(2, [0.85, 0.5, 0.5], 0.3, 1.0);
        atoms.nlocal = 3;
        atoms.natoms = 3;
    }

    // Step the loop a few times. With no integrator the atoms don't move, but the
    // substrate still runs migration, ghost exchange, and neighbor-list rebuilds
    // each step — exactly the machinery a physics tier rides on.
    for _ in 0..3 {
        app.run();
    }

    // Walk the pair list the substrate built. `pairs(nlocal)` yields `(i, j)`
    // index pairs within the neighbor cutoff; `j` may be a local atom or a ghost.
    let atoms = app.get_resource_ref::<Atom>().unwrap();
    let domain = app.get_resource_ref::<Domain>().unwrap();
    let neighbor = app.get_resource_ref::<Neighbor>().unwrap();
    let nlocal = atoms.nlocal as usize;

    println!(
        "box = [{:?} .. {:?}], {} local atoms",
        domain.boundaries_low, domain.boundaries_high, nlocal
    );
    let mut npairs = 0usize;
    for (i, j) in neighbor.pairs(nlocal) {
        npairs += 1;
        println!("pair ({i}, {j})");
    }
    println!("{npairs} neighbor pair(s) within cutoff");
}
