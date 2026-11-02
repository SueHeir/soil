//! Make SOIL split its particles exactly where a FIELD mesh splits its cells.
//!
//! Particle-grid methods (MPM, particle-in-cell) move data between each
//! particle and the grid cells around it every step. That is only local when
//! a particle lives on the same rank as those cells, so SOIL must cut its
//! domain on the mesh's cell faces instead of choosing its own grid.
//!
//! ```ignore
//! app.add_plugins(UniformMeshPlugin::new(grid));
//! app.add_plugins(FollowMeshPlugin); // after the mesh plugin
//! ```
//!
//! Neither SOIL nor FIELD depends on the other: FIELD's split is read from
//! its mesh config with FIELD's own decomposition rule, and handed to SOIL as
//! [`soil_core::FixedCuts`]. Load balancing is off for SOIL while it follows
//! the mesh.

#![warn(missing_docs)]

use field_core::uniform::factor_decomposition;
use field_core::UniformMeshConfig;
use grass_app::prelude::*;
use grass_mpi::CommResource;
use soil_core::FixedCuts;

/// The planes where a [`UniformMeshConfig`] mesh is cut across `decomp`
/// ranks: the same cell faces `UniformMesh` uses, computed the same way, so
/// SOIL's slabs and FIELD's partitions line up exactly.
pub fn mesh_cut_planes(cfg: &UniformMeshConfig, decomp: [i32; 3]) -> [Vec<f64>; 3] {
    let n = [cfg.nx, cfg.ny, cfg.nz];
    let dx = (cfg.bounds_hi[0] - cfg.bounds_lo[0]) / n[0] as f64;
    let edges = |axis: usize, user: &Option<Vec<f64>>| -> Vec<f64> {
        user.clone().unwrap_or_else(|| {
            let (lo, hi) = (cfg.bounds_lo[axis], cfg.bounds_hi[axis]);
            (0..=n[axis])
                .map(|i| lo + i as f64 * (hi - lo) / n[axis] as f64)
                .collect()
        })
    };
    let face = |axis: usize, offset: usize| -> f64 {
        match axis {
            0 => cfg.bounds_lo[0] + offset as f64 * dx,
            1 => edges(1, &cfg.y_edges)[offset],
            _ => edges(2, &cfg.z_edges)[offset],
        }
    };
    std::array::from_fn(|axis| {
        // UniformMesh's split: the first `remainder` ranks get one extra cell.
        let p = decomp[axis] as usize;
        let (base, remainder) = (n[axis] / p, n[axis] % p);
        (0..=p)
            .map(|r| {
                let offset = if r <= remainder {
                    r * (base + 1)
                } else {
                    remainder * (base + 1) + (r - remainder) * base
                };
                face(axis, offset)
            })
            .collect()
    })
}

/// The rank grid FIELD's `UniformMeshPlugin` picks for `nprocs` ranks.
pub fn mesh_decomposition(cfg: &UniformMeshConfig, nprocs: i32) -> [i32; 3] {
    if nprocs <= 1 {
        return [1, 1, 1];
    }
    factor_decomposition(nprocs, [cfg.nx, cfg.ny, cfg.nz])
        .unwrap_or_else(|e| panic!("UniformMesh decomposition failed: {e}"))
}

/// Makes SOIL cut its domain on the FIELD mesh's partition faces.
///
/// Add it after `field_core::UniformMeshPlugin` (it reads the mesh config
/// that plugin installs) and before setup runs.
pub struct FollowMeshPlugin;

impl Plugin for FollowMeshPlugin {
    fn build(&self, app: &mut App) {
        let cfg = app
            .get_resource_ref::<UniformMeshConfig>()
            .expect("FollowMeshPlugin: add field_core::UniformMeshPlugin first")
            .clone();
        let nprocs = app
            .get_resource_ref::<CommResource>()
            .expect("FollowMeshPlugin: no communicator (UniformMeshPlugin adds one)")
            .size();
        let decomp = mesh_decomposition(&cfg, nprocs);
        app.add_resource(FixedCuts {
            planes: mesh_cut_planes(&cfg, decomp),
        });
    }

    fn requires(&self) -> Vec<&str> {
        vec!["mesh"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use field_core::uniform::rank_position;
    use field_core::PartitionDirectory;

    fn config(n: [usize; 3]) -> UniformMeshConfig {
        UniformMeshConfig {
            nx: n[0],
            ny: n[1],
            nz: n[2],
            bounds_lo: [-0.3, 0.0, 1.0],
            bounds_hi: [0.7, 2.0, 2.5],
            ..Default::default()
        }
    }

    /// The planes must be the faces of each rank's FIELD partition (FIELD's
    /// own coupling directory), for uneven splits too.
    #[test]
    fn planes_match_every_ranks_mesh_partition() {
        for (n, nprocs) in [
            ([10, 7, 5], 6),
            ([9, 4, 3], 4),
            ([16, 16, 16], 8),
            ([7, 3, 2], 7),
        ] {
            let cfg = config(n);
            let decomp = mesh_decomposition(&cfg, nprocs);
            let planes = mesh_cut_planes(&cfg, decomp);
            let directory = PartitionDirectory::from_uniform_config(&cfg, decomp);
            for extent in directory.extents() {
                let pos = rank_position(extent.role_rank, decomp);
                for axis in 0..3 {
                    let r = pos[axis] as usize;
                    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * (1.0 + b.abs());
                    assert!(
                        close(planes[axis][r], extent.bounds.lo[axis]),
                        "{n:?}/{nprocs} {extent:?}"
                    );
                    assert!(
                        close(planes[axis][r + 1], extent.bounds.hi[axis]),
                        "{n:?}/{nprocs} {extent:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn serial_run_keeps_one_slab() {
        let cfg = config([4, 4, 4]);
        let planes = mesh_cut_planes(&cfg, mesh_decomposition(&cfg, 1));
        assert_eq!(planes[0], vec![-0.3, 0.7]);
    }
}
