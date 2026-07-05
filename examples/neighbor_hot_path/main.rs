//! Microbenchmark for the SOIL neighbor-list hot path.
//!
//! This example builds a deterministic monodisperse particle lattice, runs the
//! bin-based CSR neighbor-list builder repeatedly, and prints a compact timing
//! line suitable for before/after refactor comparisons.
//!
//! Run from the soil repo root:
//!
//! ```sh
//! cargo run --release --example neighbor_hot_path -- 24 60
//! ```

use std::hint::black_box;
use std::time::Instant;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use soil_core::{
    bin_neighbor_list, neighbor_setup, Atom, CommResource, CommState, Domain, Neighbor,
    NeighborConfig, ParticleSimScheduleSet, ScheduleSetupSet, SingleProcessComm,
};

fn push_lattice_atoms(atoms: &mut Atom, ncell: usize, spacing: f64, radius: f64) {
    let mut tag = 0u32;
    let margin = spacing;
    for ix in 0..ncell {
        for iy in 0..ncell {
            for iz in 0..ncell {
                let pos = [
                    margin + ix as f64 * spacing,
                    margin + iy as f64 * spacing,
                    margin + iz as f64 * spacing,
                ];
                atoms.push_test_atom(tag, pos, radius, 1.0);
                tag += 1;
            }
        }
    }
    atoms.nlocal = tag;
    atoms.natoms = tag as u64;
}

fn main() {
    let ncell = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(24);
    let iterations = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(60);

    let spacing = 0.92;
    let radius = 0.50;
    let length = spacing * (ncell as f64 + 2.0);
    let natoms = ncell * ncell * ncell;

    let mut atoms = Atom::new();
    push_lattice_atoms(&mut atoms, ncell, spacing, radius);

    let mut domain = Domain::new();
    domain.boundaries_low = [0.0, 0.0, 0.0];
    domain.boundaries_high = [length, length, length];
    domain.sub_domain_low = [0.0, 0.0, 0.0];
    domain.sub_domain_high = [length, length, length];
    domain.sub_length = [length, length, length];
    domain.size = [length, length, length];

    let mut neighbor = Neighbor::new();
    neighbor.skin_fraction = 1.0;
    neighbor.bin_min_size = 1.0;
    neighbor.sort_every = 0;

    let mut app = App::new();
    app.add_resource(atoms);
    app.add_resource(neighbor);
    app.add_resource(domain);
    app.add_resource(NeighborConfig {
        skin_fraction: 1.0,
        bin_size: 1.0,
        every: 0,
        check: true,
        sort_every: 0,
        newton: true,
    });
    app.add_resource(CurrentState(CommState::FullRebuild));
    app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
    app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
    app.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
    app.organize_systems();
    app.setup();

    for _ in 0..5 {
        app.run();
    }

    let start = Instant::now();
    let mut pair_visits = 0usize;
    for _ in 0..iterations {
        app.run();
        let neighbor = app.get_resource_ref::<Neighbor>().unwrap();
        pair_visits += black_box(neighbor.neighbor_indices.len());
    }
    let elapsed = start.elapsed();
    let ns_per_build = elapsed.as_nanos() as f64 / iterations as f64;
    let pairs_per_build = pair_visits as f64 / iterations as f64;

    println!(
        "BENCH neighbor_hot_path natoms={} iterations={} pairs_per_build={:.0} ns_per_build={:.0} builds_per_s={:.3}",
        natoms,
        iterations,
        pairs_per_build,
        ns_per_build,
        1.0e9 / ns_per_build
    );
}
