//! Release-build gate for SOIL's currently single-hop atom exchange.
//!
//! Run through `sweep.py`; it starts four x-decomposed MPI ranks and seeds one
//! atom on rank 0 at a coordinate owned by rank 2. A single neighbor exchange can
//! only move that atom to rank 1, so the substrate must stop with a hard,
//! actionable error instead of leaving the atom mis-owned.

use std::{any::TypeId, fs};

use grass_app::prelude::*;
use soil_core::{
    Atom, AtomDataRegistry, AtomPlugin, CommResource, CommunicationPlugin, Config, DomainPlugin,
    NeighborPlugin, ParticleStore,
};

fn main() {
    let cfg_path = std::env::args()
        .nth(1)
        .expect("usage: exchange_multihop_gate <config.toml>");
    let toml =
        fs::read_to_string(&cfg_path).unwrap_or_else(|e| panic!("cannot read {cfg_path}: {e}"));

    let mut app = App::new();
    app.add_resource(Config::from_str(&toml));
    app.add_plugins(AtomPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(NeighborPlugin)
        .add_plugins(CommunicationPlugin);

    app.prepare();

    let (rank, size) = {
        let comm = app.get_resource_ref::<CommResource>().unwrap();
        (comm.rank(), comm.size())
    };
    assert_eq!(size, 4, "exchange_multihop_gate must run with 4 MPI ranks");

    if rank == 0 {
        let atom_cell = app
            .resource_cell(TypeId::of::<Atom>())
            .expect("AtomPlugin registers the Atom resource");
        let registry_cell = app
            .resource_cell(TypeId::of::<AtomDataRegistry>())
            .expect("AtomPlugin registers the AtomDataRegistry resource");
        let mut atom_resource = atom_cell.borrow_mut();
        let registry_resource = registry_cell.borrow();
        let atoms = atom_resource.downcast_mut::<Atom>().unwrap();
        let registry = registry_resource
            .downcast_ref::<AtomDataRegistry>()
            .unwrap();

        ParticleStore::new(atoms, registry)
            .push_default_local(1)
            .expect("seed the deliberately mis-owned atom");
        atoms.tag[0] = 42;
        atoms.pos[0] = [0.625, 0.5, 0.5];
        atoms.cutoff_radius[0] = 0.01;
        atoms.mass[0] = 1.0;
        atoms.inv_mass[0] = 1.0;
    }

    // One full-rebuild step runs PBC, exchange, and then neighbor rebuild. The
    // exchange postcondition should panic before a mis-owned atom reaches bins.
    app.run();
    app.run_cleanup();

    if rank == 0 {
        println!("ERROR: exchange completed even though a two-rank hop was required");
    }
}
