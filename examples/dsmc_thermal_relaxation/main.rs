//! One-dimensional DSMC thermal-relaxation demonstrator on SOIL.
//!
//! This example keeps the rarefied-gas physics out of SOIL core: the substrate
//! owns particle state and spatial cell binning, while this file adds only the
//! stochastic collision step. The validation is a homogeneous thermal
//! relaxation test against the published BGK relaxation form
//! `f(t) = f_eq + (f(0) - f_eq) exp(-nu t)` from
//! Bhatnagar, Gross & Krook, Phys. Rev. 94, 511-525 (1954). For Maxwell-molecule
//! DSMC collisions, the temperature anisotropy
//! `A = T_x - (T_y + T_z)/2` relaxes with the same exponential in collision
//! time.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example dsmc_thermal_relaxation -- \
//!   examples/dsmc_thermal_relaxation/cfg.toml
//! ```

use std::{any::TypeId, env, fs, path::PathBuf};

use grass_app::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Deserialize;
use soil_core::{
    Atom, AtomDataRegistry, AtomPlugin, CommunicationPlugin, Config, DomainPlugin, Neighbor,
    NeighborPlugin, ParticleStore,
};

#[derive(Deserialize)]
struct CaseFile {
    dsmc: DsmcConfig,
}

#[derive(Clone, Copy, Deserialize)]
struct DsmcConfig {
    seed: u64,
    cells_x: usize,
    particles_per_cell: usize,
    steps: usize,
    collision_pairs_per_particle: f64,
    tx0: f64,
    ty0: f64,
    tz0: f64,
}

fn gaussian(rng: &mut StdRng, sigma: f64) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn unit_vector(rng: &mut StdRng) -> [f64; 3] {
    let z = 2.0 * rng.random::<f64>() - 1.0;
    let phi = 2.0 * std::f64::consts::PI * rng.random::<f64>();
    let r = (1.0 - z * z).sqrt();
    [r * phi.cos(), r * phi.sin(), z]
}

fn seed_atoms(app: &mut App, cfg: DsmcConfig) {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let cell_width = 1.0 / cfg.cells_x as f64;
    let n = cfg.cells_x * cfg.particles_per_cell;
    let registry_cell = app
        .resource_cell(TypeId::of::<AtomDataRegistry>())
        .expect("AtomPlugin registers AtomDataRegistry before prepare");
    let registry_binder = registry_cell.borrow();
    let registry = registry_binder.downcast_ref::<AtomDataRegistry>().unwrap();
    let cell = app
        .resource_cell(TypeId::of::<Atom>())
        .expect("AtomPlugin registers Atom before prepare");
    let mut binder = cell.borrow_mut();
    let atoms = binder.downcast_mut::<Atom>().unwrap();
    atoms.dt = 1.0;
    let mut store = ParticleStore::new(atoms, registry);
    store.reserve(n);

    for c in 0..cfg.cells_x {
        for p in 0..cfg.particles_per_cell {
            let tag = (c * cfg.particles_per_cell + p) as u32;
            let fx = (p as f64 + 0.5) / cfg.particles_per_cell as f64;
            store.push_default_local(n as u64).unwrap();
            let i = store.atoms().len() - 1;
            drop(store);
            atoms.tag[i] = tag;
            atoms.pos[i] = [
                ((c as f64 + fx) * cell_width) as soil_core::Real,
                (0.5 * cell_width) as soil_core::Real,
                (0.5 * cell_width) as soil_core::Real,
            ];
            atoms.cutoff_radius[i] = (0.5 * cell_width) as soil_core::Real;
            atoms.vel[i] = [
                gaussian(&mut rng, cfg.tx0.sqrt()) as soil_core::Real,
                gaussian(&mut rng, cfg.ty0.sqrt()) as soil_core::Real,
                gaussian(&mut rng, cfg.tz0.sqrt()) as soil_core::Real,
            ];
            store = ParticleStore::new(atoms, registry);
        }
    }
}

fn soil_cells(neighbor: &Neighbor, nlocal: usize) -> Vec<Vec<usize>> {
    let mut cells = Vec::new();
    for c in 0..neighbor.bin_total_cells {
        let start = neighbor.bin_start[c] as usize;
        let end = neighbor.bin_start[c + 1] as usize;
        let mut members = Vec::new();
        for m in start..end {
            let i = neighbor.bin_sorted_atoms[m] as usize;
            if i < nlocal {
                members.push(i);
            }
        }
        if !members.is_empty() {
            cells.push(members);
        }
    }
    cells
}

fn collide_cell(
    atoms: &mut Atom,
    members: &[usize],
    pairs_per_particle: f64,
    rng: &mut StdRng,
) -> usize {
    if members.len() < 2 {
        return 0;
    }
    let ncoll = (pairs_per_particle * members.len() as f64).round() as usize;
    for _ in 0..ncoll {
        let a = rng.random_range(0..members.len());
        let mut b = rng.random_range(0..members.len() - 1);
        if b >= a {
            b += 1;
        }
        let i = members[a];
        let j = members[b];

        let vi = [
            atoms.vel[i][0] as f64,
            atoms.vel[i][1] as f64,
            atoms.vel[i][2] as f64,
        ];
        let vj = [
            atoms.vel[j][0] as f64,
            atoms.vel[j][1] as f64,
            atoms.vel[j][2] as f64,
        ];
        let cm = [
            0.5 * (vi[0] + vj[0]),
            0.5 * (vi[1] + vj[1]),
            0.5 * (vi[2] + vj[2]),
        ];
        let g = [vi[0] - vj[0], vi[1] - vj[1], vi[2] - vj[2]];
        let gmag = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        let dir = unit_vector(rng);
        let gp = [gmag * dir[0], gmag * dir[1], gmag * dir[2]];
        atoms.vel[i] = [
            (cm[0] + 0.5 * gp[0]) as soil_core::Real,
            (cm[1] + 0.5 * gp[1]) as soil_core::Real,
            (cm[2] + 0.5 * gp[2]) as soil_core::Real,
        ];
        atoms.vel[j] = [
            (cm[0] - 0.5 * gp[0]) as soil_core::Real,
            (cm[1] - 0.5 * gp[1]) as soil_core::Real,
            (cm[2] - 0.5 * gp[2]) as soil_core::Real,
        ];
    }
    ncoll
}

fn temperatures(atoms: &Atom) -> [f64; 3] {
    let n = atoms.nlocal() as usize;
    let mut mean = [0.0; 3];
    for v in atoms.vel[..n].iter() {
        mean[0] += v[0] as f64;
        mean[1] += v[1] as f64;
        mean[2] += v[2] as f64;
    }
    mean.iter_mut().for_each(|m| *m /= n as f64);

    let mut t = [0.0; 3];
    for v in atoms.vel[..n].iter() {
        for k in 0..3 {
            let dv = v[k] as f64 - mean[k];
            t[k] += dv * dv;
        }
    }
    t.iter_mut().for_each(|x| *x /= n as f64);
    t
}

fn anisotropy(t: [f64; 3]) -> f64 {
    t[0] - 0.5 * (t[1] + t[2])
}

fn fit_decay(collision_time: &[f64], ratio: &[f64]) -> f64 {
    let n = collision_time.len() as f64;
    let sx: f64 = collision_time.iter().sum();
    let sy: f64 = ratio.iter().map(|r| r.ln()).sum();
    let sxx: f64 = collision_time.iter().map(|x| x * x).sum();
    let sxy: f64 = collision_time
        .iter()
        .zip(ratio)
        .map(|(x, r)| x * r.ln())
        .sum();
    -((n * sxy - sx * sy) / (n * sxx - sx * sx))
}

fn main() {
    let cfg_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/dsmc_thermal_relaxation/cfg.toml"));
    let toml = fs::read_to_string(&cfg_path).expect("read declarative TOML case file");
    let case: CaseFile = toml::from_str(&toml).expect("parse [dsmc] case config");
    let cfg = case.dsmc;

    let mut app = App::new();
    app.add_resource(Config::from_str(&toml));
    app.add_plugins(AtomPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(NeighborPlugin)
        .add_plugins(CommunicationPlugin);
    seed_atoms(&mut app, cfg);
    app.prepare();
    app.run();

    let nlocal = app.get_resource_ref::<Atom>().unwrap().nlocal() as usize;
    let cells = {
        let neighbor = app.get_resource_ref::<Neighbor>().unwrap();
        soil_cells(&neighbor, nlocal)
    };
    assert_eq!(
        cells.len(),
        cfg.cells_x,
        "SOIL binning should expose the 1D DSMC cells"
    );
    assert!(
        cells.iter().all(|c| c.len() == cfg.particles_per_cell),
        "all DSMC cells should carry the configured particle count"
    );

    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0x5eed_c011_1510_0001);
    let initial_a = {
        let atoms = app.get_resource_ref::<Atom>().unwrap();
        anisotropy(temperatures(&atoms))
    };
    let mut collision_time = Vec::with_capacity(cfg.steps + 1);
    let mut ratio = Vec::with_capacity(cfg.steps + 1);
    collision_time.push(0.0);
    ratio.push(1.0);

    let mut total_collisions = 0usize;
    for _ in 0..cfg.steps {
        let cell = app
            .get_mut_resource(TypeId::of::<Atom>())
            .expect("Atom resource remains registered");
        let mut binder = cell.borrow_mut();
        let mut atoms = binder.downcast_mut::<Atom>().unwrap();
        let mut step_collisions = 0usize;
        for cell in &cells {
            step_collisions +=
                collide_cell(&mut atoms, cell, cfg.collision_pairs_per_particle, &mut rng);
        }
        total_collisions += step_collisions;
        let tau = total_collisions as f64 / nlocal as f64;
        collision_time.push(tau);
        ratio.push(anisotropy(temperatures(&atoms)) / initial_a);
    }

    let final_tau = *collision_time.last().unwrap();
    let measured = *ratio.last().unwrap();
    let bgk = (-final_tau).exp();
    let rel_err = ((measured - bgk) / bgk).abs();
    let slope = fit_decay(&collision_time, &ratio);
    let slope_err = (slope - 1.0).abs();

    let atoms = app.get_resource_ref::<Atom>().unwrap();
    let t_final = temperatures(&atoms);

    println!("SOIL DSMC thermal relaxation (Maxwell-molecule collision step)");
    println!("  reference            : BGK exponential relaxation, Bhatnagar-Gross-Krook 1954");
    println!(
        "  SOIL cells           : {} bins x {} particles",
        cells.len(),
        cfg.particles_per_cell
    );
    println!("  particles            : {nlocal}");
    println!("  accepted collisions  : {total_collisions}");
    println!("  collision time tau   : {final_tau:.6}");
    println!(
        "  final temperatures   : Tx={:.6} Ty={:.6} Tz={:.6}",
        t_final[0], t_final[1], t_final[2]
    );
    println!("  DSMC A/A0            : {measured:.6}");
    println!("  BGK  exp(-tau)       : {bgk:.6}");
    println!("  relative error       : {rel_err:.3e}");
    println!("  fitted decay rate    : {slope:.6} (BGK target 1.0)");

    assert!(
        rel_err < 0.20,
        "DSMC thermal relaxation should track BGK within stochastic error"
    );
    assert!(slope_err < 0.08, "fitted relaxation rate should match BGK");
    println!("ALL CHECKS PASSED");
}
