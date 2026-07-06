//! Validation and microbenchmark driver for the SOIL neighbor distance kernel.
//!
//! The checked-in TOML describes deterministic uniform and polydisperse particle
//! clouds. For each case this example builds scalar and SIMD-requested neighbor
//! lists, verifies identical pairs, then reports a compact timing line.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::Deserialize;
use soil_core::{
    bin_neighbor_list, neighbor_setup, Atom, CommResource, CommState, Domain, Neighbor,
    NeighborConfig, NeighborDistanceKernel, ParticleSimScheduleSet, ScheduleSetupSet,
    SingleProcessComm,
};

#[derive(Deserialize)]
struct BenchConfig {
    warmup: usize,
    iterations: usize,
    cases: Vec<CaseConfig>,
}

#[derive(Clone, Deserialize)]
struct CaseConfig {
    name: String,
    layout: Layout,
    ncell: usize,
    spacing: f64,
    radius: f64,
    radius_jitter: Option<f64>,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Layout {
    Uniform,
    Polydisperse,
}

fn radius_for(case: &CaseConfig, tag: usize) -> f64 {
    match case.layout {
        Layout::Uniform => case.radius,
        Layout::Polydisperse => {
            let jitter = case.radius_jitter.unwrap_or(0.18);
            let phase = ((tag as f64 * 12.9898).sin() * 43758.5453).fract();
            case.radius * (1.0 + jitter * (phase * 2.0 - 1.0))
        }
    }
}

fn build_atoms(case: &CaseConfig) -> Atom {
    let mut atoms = Atom::new();
    let mut tag = 0u32;
    let margin = case.spacing;
    for ix in 0..case.ncell {
        for iy in 0..case.ncell {
            for iz in 0..case.ncell {
                let wobble = ((tag as f64 * 0.618_033_988_75).sin()) * 0.035 * case.spacing;
                let pos = [
                    margin + ix as f64 * case.spacing + wobble,
                    margin + iy as f64 * case.spacing - 0.5 * wobble,
                    margin + iz as f64 * case.spacing + 0.25 * wobble,
                ];
                atoms.push_test_atom(tag, pos, radius_for(case, tag as usize), 1.0);
                tag += 1;
            }
        }
    }
    atoms.nlocal = tag;
    atoms.natoms = tag as u64;
    atoms
}

fn build_app(case: &CaseConfig, kernel: NeighborDistanceKernel) -> App {
    let atoms = build_atoms(case);
    let length = case.spacing * (case.ncell as f64 + 2.0);

    let mut domain = Domain::new();
    domain.boundaries_low = [0.0, 0.0, 0.0];
    domain.boundaries_high = [length, length, length];
    domain.sub_domain_low = [0.0, 0.0, 0.0];
    domain.sub_domain_high = [length, length, length];
    domain.sub_length = [length, length, length];
    domain.size = [length, length, length];

    let mut neighbor = Neighbor::new();
    neighbor.skin_fraction = 1.0;
    neighbor.bin_min_size = case.radius * 2.0;
    neighbor.sort_every = 0;
    neighbor.distance_kernel = kernel;

    let mut app = App::new();
    app.add_resource(atoms);
    app.add_resource(neighbor);
    app.add_resource(domain);
    app.add_resource(NeighborConfig {
        skin_fraction: 1.0,
        bin_size: case.radius * 2.0,
        every: 0,
        check: true,
        sort_every: 0,
        newton: true,
        distance_kernel: kernel,
    });
    app.add_resource(CurrentState(CommState::FullRebuild));
    app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
    app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
    app.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
    app.organize_systems();
    app.setup();
    app
}

fn neighbor_pairs(app: &App) -> Vec<(usize, usize)> {
    let neighbor = app.get_resource_ref::<Neighbor>().unwrap();
    let atoms = app.get_resource_ref::<Atom>().unwrap();
    let mut pairs: Vec<(usize, usize)> = neighbor
        .pairs(atoms.nlocal as usize)
        .map(|(i, j)| if i < j { (i, j) } else { (j, i) })
        .collect();
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

fn time_builds(mut app: App, warmup: usize, iterations: usize) -> (Vec<(usize, usize)>, f64) {
    for _ in 0..warmup {
        app.run();
    }
    let first_pairs = neighbor_pairs(&app);

    let start = Instant::now();
    let mut pair_visits = 0usize;
    for _ in 0..iterations {
        app.run();
        let neighbor = app.get_resource_ref::<Neighbor>().unwrap();
        pair_visits += black_box(neighbor.neighbor_indices.len());
    }
    let elapsed = start.elapsed();
    black_box(pair_visits);
    let ns_per_build = elapsed.as_nanos() as f64 / iterations as f64;
    (first_pairs, ns_per_build)
}

fn main() {
    let cfg_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/neighbor_hot_path/config.toml"));
    let cfg_text = std::fs::read_to_string(&cfg_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", cfg_path.display()));
    let cfg: BenchConfig = toml::from_str(&cfg_text)
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", cfg_path.display()));

    let simd_available = NeighborDistanceKernel::Simd.simd_available();
    for case in &cfg.cases {
        let natoms = case.ncell * case.ncell * case.ncell;
        let scalar_app = build_app(case, NeighborDistanceKernel::Scalar);
        let simd_app = build_app(case, NeighborDistanceKernel::Simd);

        let (scalar_pairs, scalar_ns) = time_builds(scalar_app, cfg.warmup, cfg.iterations);
        let (simd_pairs, simd_ns) = time_builds(simd_app, cfg.warmup, cfg.iterations);
        let pairs_match = scalar_pairs == simd_pairs;
        if !pairs_match {
            panic!("SIMD pair set differed from scalar for case {}", case.name);
        }
        let speedup = scalar_ns / simd_ns;
        println!(
            "SIMD_DISTANCE case={} layout={} natoms={} pairs={} simd_available={} pairs_match={} scalar_ns_per_build={:.0} simd_ns_per_build={:.0} speedup={:.3}",
            case.name,
            match case.layout {
                Layout::Uniform => "uniform",
                Layout::Polydisperse => "polydisperse",
            },
            natoms,
            scalar_pairs.len(),
            simd_available,
            pairs_match,
            scalar_ns,
            simd_ns,
            speedup
        );
    }
    println!("ALL CHECKS PASSED");
}
