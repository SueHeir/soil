//! On several real MPI ranks, a SOIL + FIELD App with `FollowMeshPlugin` must
//! put every rank's particle slab exactly on its mesh partition, so SOIL and
//! FIELD agree on the owner of every point.
//!
//! The test re-launches itself under `mpirun` (skipped if `mpirun` is absent).
#![cfg(feature = "mpi_backend")]

use std::process::Command;

use field_core::{
    FieldRegistryPlugin, OwnerMap, PartitionDirectory, UniformMeshConfig, UniformMeshPlugin,
};
use grass_app::prelude::*;
use grass_io::Config;
use grass_mpi::{CommBackend, CommResource};
use soil_core::{CommunicationPlugin, Domain, DomainPlugin, ParticlePartitionDirectory};
use soil_follow_mesh::{mesh_decomposition, FollowMeshPlugin};

const CHILD_ENV: &str = "SOIL_FOLLOW_MESH_CHILD";
const NRANKS: usize = 6;

fn mesh() -> UniformMeshConfig {
    UniformMeshConfig {
        // Uneven on purpose: 10 cells over 3 ranks is 4 + 3 + 3.
        nx: 10,
        ny: 7,
        nz: 5,
        ng: 1,
        bounds_lo: [0.0, -1.0, 0.5],
        bounds_hi: [2.0, 1.0, 1.5],
        ..Default::default()
    }
}

fn run_body() {
    let cfg = mesh();
    let mut app = App::new();
    // SOIL's own [comm] request (a 6x1x1 grid) must be overridden by the mesh's.
    app.add_resource(Config::from_str(
        "[comm]\nprocessors_x = 6\nprocessors_y = 1\nprocessors_z = 1\n\
         [domain]\nx_low = 0.0\nx_high = 2.0\ny_low = -1.0\ny_high = 1.0\n\
         z_low = 0.5\nz_high = 1.5\nboundary_x = 'fixed'\nboundary_y = 'periodic'\n\
         boundary_z = 'fixed'\n",
    ));
    app.add_plugins(soil_core::AtomPlugin)
        .add_plugins(CommunicationPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(soil_core::BinningPlugin)
        .add_plugins(FieldRegistryPlugin)
        .add_plugins(UniformMeshPlugin::new(cfg.clone()))
        .add_plugins(FollowMeshPlugin);
    app.prepare();

    let comm = app.get_resource_ref::<CommResource>().unwrap();
    let (rank, size) = (comm.rank(), comm.size());
    assert_eq!(size, NRANKS as i32);
    let decomp = mesh_decomposition(&cfg, size);
    assert_ne!(
        decomp,
        [6, 1, 1],
        "test needs the mesh grid to differ from SOIL's request"
    );
    assert_eq!(comm.processor_decomposition(), decomp);

    let domain = app.get_resource_ref::<Domain>().unwrap();
    let field = PartitionDirectory::from_uniform_config(&cfg, decomp);
    let soil = ParticlePartitionDirectory::from_domain(&domain).unwrap();

    // This rank's particle slab is its mesh partition.
    let mine = field
        .extents()
        .iter()
        .find(|e| e.role_rank == rank)
        .expect("this rank owns a mesh partition");
    for axis in 0..3 {
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12;
        assert!(
            close(domain.sub_domain_low[axis], mine.bounds.lo[axis]),
            "rank {rank} axis {axis}"
        );
        assert!(
            close(domain.sub_domain_high[axis], mine.bounds.hi[axis]),
            "rank {rank} axis {axis}"
        );
    }

    // Every cell center has the same owner in SOIL and FIELD.
    let mut checked = 0;
    for i in 0..cfg.nx {
        for j in 0..cfg.ny {
            for k in 0..cfg.nz {
                let p = [
                    (i as f64 + 0.5) * 2.0 / cfg.nx as f64,
                    -1.0 + (j as f64 + 0.5) * 2.0 / cfg.ny as f64,
                    0.5 + (k as f64 + 0.5) / cfg.nz as f64,
                ];
                assert_eq!(
                    OwnerMap::owner_rank(&soil, p),
                    OwnerMap::owner_rank(&field, p),
                    "owner of {p:?}"
                );
                checked += 1;
            }
        }
    }
    if rank == 0 {
        println!(
            "soil_follow_mesh PASS: {size} ranks on grid {decomp:?}, {checked} cell centers agree"
        );
    }
    drop(domain);
    drop(comm);
    app.run_cleanup();
}

#[test]
fn soil_slabs_match_the_mesh_partition_on_six_ranks() {
    if std::env::var_os(CHILD_ENV).is_some() {
        run_body();
        return;
    }
    if Command::new("mpirun").arg("--version").output().is_err() {
        eprintln!("SKIP: `mpirun` not found on PATH");
        return;
    }
    let exe = std::env::current_exe().expect("locate test binary");
    let status = Command::new("mpirun")
        .args(["--oversubscribe", "-np", &NRANKS.to_string()])
        .arg(&exe)
        .args([
            "--exact",
            "soil_slabs_match_the_mesh_partition_on_six_ranks",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ENV, "1")
        .status()
        .expect("spawn mpirun");
    assert!(status.success(), "multi-rank run failed: {status}");
}
