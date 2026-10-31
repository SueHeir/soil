//! Neighbor list construction for particle simulations.
//!
//! This facade preserves the public neighbor API while keeping the hot path split
//! across focused modules for configuration/state, rebuild policy, bin grid and
//! stencil construction, pair-list construction, spatial sorting, CSR iteration,
//! and tests.

mod config;
mod csr;
mod grid;
mod pair_list;
mod plugin;
mod rebuild;
mod sorting;
mod state;

pub use config::{NeighborConfig, NeighborDistanceKernel};
pub use csr::PairIter;
pub use pair_list::bin_neighbor_list;
pub use plugin::{
    neighbor_read_input, neighbor_setup, DecideRebuildSystem, NeighborPlugin, NeighborSetupSystem,
    SortAtomsSystem, DECIDE_REBUILD, NEIGHBOR_SETUP, SORT_ATOMS,
};
pub use rebuild::decide_rebuild;
pub use sorting::sort_atoms_by_bin;
pub use state::{classify_boundary_atoms, Neighbor};

/// `x.floor() as i32` without the libm call that `f64::floor` lowers to on
/// the baseline x86-64 target (no SSE4.1). Identical for every finite `x` in
/// `i32` range; out-of-range values saturate like the `as` cast, and callers
/// clamp the result to the bin grid anyway.
#[inline(always)]
pub(crate) fn floor_to_i32(x: f64) -> i32 {
    let t = x as i32;
    t - ((t as f64) > x) as i32
}

#[cfg(test)]
mod tests;
