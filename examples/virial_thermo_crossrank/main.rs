//! SOIL-tier virial thermo reduction validation.
//!
//! The input file declares a small set of pair-force contributions. This
//! example distributes those contributions across MPI ranks, accumulates them
//! through [`soil_core::VirialStress::add_pair`], and lets `soil_print` publish
//! the globally all-reduced `virial_xx..virial_yz` thermo columns. The sweep
//! checks those printed columns against the analytic tensor.

use std::fs;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::Deserialize;
use soil_core::{
    AtomPlugin, CommResource, CommunicationPlugin, Config, DomainPlugin, Input, NeighborPlugin,
    ParticleSimScheduleSet, RunPlugin, VirialStress, VirialStressPlugin,
};
use soil_print::PrintPlugin;

#[derive(Clone, Deserialize)]
struct PairContribution {
    dx: f64,
    dy: f64,
    dz: f64,
    fx: f64,
    fy: f64,
    fz: f64,
}

#[derive(Clone, Deserialize)]
struct VirialCase {
    pair: Vec<PairContribution>,
}

#[derive(Clone, Deserialize)]
struct ExampleConfig {
    virial_case: VirialCase,
}

fn accumulate_declared_pairs(
    case: Res<VirialCase>,
    comm: Res<CommResource>,
    mut virial: ResMut<VirialStress>,
) {
    if !virial.active {
        return;
    }
    let rank = comm.rank() as usize;
    let size = comm.size() as usize;
    for (idx, pair) in case.pair.iter().enumerate() {
        if idx % size == rank {
            virial.add_pair(pair.dx, pair.dy, pair.dz, pair.fx, pair.fy, pair.fz);
        }
    }
}

fn analytic_tensor(case: &VirialCase) -> [f64; 6] {
    let mut out = [0.0; 6];
    for pair in &case.pair {
        out[0] += pair.dx * pair.fx;
        out[1] += pair.dy * pair.fy;
        out[2] += pair.dz * pair.fz;
        out[3] += pair.dx * pair.fy;
        out[4] += pair.dx * pair.fz;
        out[5] += pair.dy * pair.fz;
    }
    out
}

fn main() {
    let cfg_path = std::env::args()
        .nth(1)
        .expect("usage: virial_thermo_crossrank <config.toml>");
    let toml =
        fs::read_to_string(&cfg_path).unwrap_or_else(|e| panic!("cannot read {cfg_path}: {e}"));
    let parsed: ExampleConfig =
        toml::from_str(&toml).unwrap_or_else(|e| panic!("cannot parse {cfg_path}: {e}"));
    let expected = analytic_tensor(&parsed.virial_case);

    let mut app = App::new();
    app.add_resource(Config::from_str(&toml));
    app.add_resource(Input {
        filename: cfg_path.clone(),
        output_dir: std::path::Path::new(&cfg_path)
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.to_string_lossy().into_owned()),
    });
    app.add_resource(parsed.virial_case);
    app.add_plugins(AtomPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(NeighborPlugin)
        .add_plugins(CommunicationPlugin)
        .add_plugins(RunPlugin)
        .add_plugins(VirialStressPlugin)
        .add_plugins(PrintPlugin);
    app.add_update_system(accumulate_declared_pairs, ParticleSimScheduleSet::Force);

    app.prepare();
    let rank = app.get_resource_ref::<CommResource>().unwrap().rank();
    if rank == 0 {
        println!(
            "ANALYTIC virial_xx={:.17e} virial_yy={:.17e} virial_zz={:.17e} \
             virial_xy={:.17e} virial_xz={:.17e} virial_yz={:.17e}",
            expected[0], expected[1], expected[2], expected[3], expected[4], expected[5],
        );
    }

    app.run();
    app.run_cleanup();
}
