//! Neighbor-list configuration and distance-kernel selection.

use serde::{Deserialize, Serialize};

pub(super) fn default_one_f64() -> f64 {
    1.0
}
pub(super) fn default_zero_usize() -> usize {
    0
}
pub(super) fn default_true() -> bool {
    true
}
pub(super) fn default_sort_every() -> usize {
    1000
}
pub(super) fn default_distance_kernel() -> NeighborDistanceKernel {
    NeighborDistanceKernel::Scalar
}

/// Distance-filter implementation used by the bin neighbor builder.
///
/// `Scalar` is the stable reference path. `Simd` enables a runtime-gated SIMD
/// kernel where supported, falling back to the scalar kernel otherwise.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum NeighborDistanceKernel {
    /// Always use scalar distance checks.
    Scalar,
    /// Use the SIMD distance kernel when the current target supports it.
    Simd,
}

impl NeighborDistanceKernel {
    /// Returns true when the SIMD path is requested and available at runtime.
    pub fn simd_available(self) -> bool {
        matches!(self, NeighborDistanceKernel::Simd) && simd_distance_available()
    }
}

#[cfg(target_arch = "x86_64")]
fn simd_distance_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn simd_distance_available() -> bool {
    false
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
/// TOML `[neighbor]` — neighbor list rebuild and binning settings.
pub struct NeighborConfig {
    /// Multiplier on pairwise cutoff for neighbor skin distance.
    #[serde(default = "default_one_f64")]
    pub skin_fraction: f64,
    /// Minimum bin size for bin-based neighbor lists.
    #[serde(default = "default_one_f64")]
    pub bin_size: f64,
    /// Rebuild every N steps (0 = displacement-based only)
    #[serde(default = "default_zero_usize")]
    pub every: usize,
    /// When true and every > 0, also check displacement threshold (like LAMMPS "check yes")
    #[serde(default = "default_true")]
    pub check: bool,
    /// Sort atoms by spatial bin at the first full rebuild after every N steps,
    /// for cache locality, like LAMMPS `atom_modify sort` (0 = at every full rebuild).
    #[serde(default = "default_sort_every")]
    pub sort_every: usize,
    /// Newton's third law optimization: true = half neighbor list + reverse comm,
    /// false = full neighbor list, no reverse comm.
    #[serde(default = "default_true")]
    pub newton: bool,
    /// Distance-filter kernel for neighbor construction: `scalar` or `simd`.
    #[serde(default = "default_distance_kernel")]
    pub distance_kernel: NeighborDistanceKernel,
    /// Build the pair list at each rebuild (default). `false` keeps the bin grid,
    /// per-bin CSR (`bin_start` / `bin_sorted_atoms`), sorting, exchange and
    /// borders, but leaves the pair list empty: for methods whose interactions
    /// are cell-based rather than pairwise within a cutoff (DSMC, PIC, MPM).
    #[serde(default = "default_true")]
    pub pairs: bool,
}

impl Default for NeighborConfig {
    fn default() -> Self {
        NeighborConfig {
            skin_fraction: 1.0,
            bin_size: 1.0,
            every: 0,
            check: true,
            sort_every: 1000,
            newton: true,
            distance_kernel: NeighborDistanceKernel::Scalar,
            pairs: true,
        }
    }
}
