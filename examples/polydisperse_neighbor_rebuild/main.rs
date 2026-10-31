//! Polydisperse neighbor-list regression.
//!
//! The driver seeds the same mixed-radius point cloud on each MPI layout, runs
//! SOIL's normal exchange + neighbor build, and writes the neighbor pairs seen
//! by each rank. `sweep.py` compares the union against an independent O(N^2)
//! brute-force pair set.

use std::any::TypeId;
use std::fs;
use std::io::Write;

use grass_app::prelude::*;
use soil_core::{
    Atom, AtomPlugin, CommResource, CommunicationPlugin, Config, Domain, DomainPlugin, Neighbor,
    NeighborPlugin,
};

const NX: usize = 8;
const NY: usize = 4;
const NZ: usize = 3;
const SPACING: f64 = 1.40;
const JITTER: f64 = 0.09;
const RADII: [f64; 5] = [0.18, 0.28, 0.42, 0.56, 0.70];
const SEED: u64 = 0x5011_2026_0706_4242;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn signed_unit(state: &mut u64) -> f64 {
    let u = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64;
    2.0 * u - 1.0
}

fn global_atoms() -> Vec<(u32, [f64; 3], f64)> {
    let mut out = Vec::with_capacity(NX * NY * NZ);
    let mut rng = SEED;
    for ix in 0..NX {
        for iy in 0..NY {
            for iz in 0..NZ {
                let tag = (ix * NY * NZ + iy * NZ + iz) as u32;
                let pos = [
                    (ix as f64 + 0.5) * SPACING + JITTER * signed_unit(&mut rng),
                    (iy as f64 + 0.5) * SPACING + JITTER * signed_unit(&mut rng),
                    (iz as f64 + 0.5) * SPACING + JITTER * signed_unit(&mut rng),
                ];
                let radius = RADII[(ix + 2 * iy + 3 * iz) % RADII.len()];
                out.push((tag, pos, radius));
            }
        }
    }
    out
}

fn main() {
    let cfg_path = std::env::args()
        .nth(1)
        .expect("usage: polydisperse_neighbor_rebuild <config.toml>");
    let toml =
        fs::read_to_string(&cfg_path).unwrap_or_else(|e| panic!("cannot read {cfg_path}: {e}"));

    let mut app = App::new();
    app.add_resource(Config::from_str(&toml));
    app.add_plugins(AtomPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(NeighborPlugin)
        .add_plugins(CommunicationPlugin);
    app.prepare();

    let (sub_low, sub_high, rank, ranks) = {
        let domain = app.get_resource_ref::<Domain>().unwrap();
        let comm = app.get_resource_ref::<CommResource>().unwrap();
        (
            domain.sub_domain_low,
            domain.sub_domain_high,
            comm.rank(),
            comm.size(),
        )
    };

    {
        let cell = app
            .get_mut_resource(TypeId::of::<Atom>())
            .expect("AtomPlugin registers Atom");
        let mut binder = cell.borrow_mut();
        let atoms = binder.downcast_mut::<Atom>().unwrap();
        let all_atoms = global_atoms();
        let natoms = all_atoms.len() as u64;
        for (tag, pos, radius) in all_atoms {
            let owned = (0..3).all(|d| pos[d] >= sub_low[d] && pos[d] < sub_high[d]);
            if owned {
                // SAFETY: core-only fixture; the layout is set below.
                unsafe { atoms.push_test_atom(tag, pos, radius, 1.0) };
            }
        }
        let local = atoms.tag.len() as u32;
        // SAFETY: no extensions are registered and every pushed row is local.
        unsafe { atoms.set_layout_unchecked(local, 0, natoms) };
    }

    app.run();

    let dir = "examples/polydisperse_neighbor_rebuild/data";
    fs::create_dir_all(dir).ok();
    let pair_path = format!("{dir}/pairs_n{ranks}_rank{rank}.csv");
    let state_path = format!("{dir}/atoms_n{ranks}_rank{rank}.csv");

    {
        let atoms = app.get_resource_ref::<Atom>().unwrap();
        let neighbor = app.get_resource_ref::<Neighbor>().unwrap();

        let mut pf = fs::File::create(pair_path).expect("cannot create pair csv");
        writeln!(pf, "tag_i,tag_j").unwrap();
        for i in 0..atoms.nlocal() as usize {
            let start = neighbor.neighbor_offsets[i] as usize;
            let end = neighbor.neighbor_offsets[i + 1] as usize;
            for &j in &neighbor.neighbor_indices[start..end] {
                let a = atoms.tag[i];
                let b = atoms.tag[j as usize];
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                writeln!(pf, "{lo},{hi}").unwrap();
            }
        }

        let mut sf = fs::File::create(state_path).expect("cannot create atom csv");
        writeln!(sf, "tag,x,y,z,radius,local").unwrap();
        for i in 0..atoms.len() {
            writeln!(
                sf,
                "{},{:.17e},{:.17e},{:.17e},{:.17e},{}",
                atoms.tag[i],
                atoms.pos[i][0],
                atoms.pos[i][1],
                atoms.pos[i][2],
                atoms.cutoff_radius[i],
                i < atoms.nlocal() as usize,
            )
            .unwrap();
        }

        if rank == 0 {
            println!(
                "POLY_NEIGHBOR ranks={ranks} local={} pairs={}",
                atoms.nlocal(),
                neighbor.neighbor_indices.len()
            );
        }
    }

    app.run_cleanup();
}
