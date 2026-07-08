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
pub use plugin::{neighbor_read_input, neighbor_setup, NeighborPlugin};
pub use rebuild::decide_rebuild;
pub use sorting::sort_atoms_by_bin;
pub use state::{classify_boundary_atoms, Neighbor};

#[cfg(test)]
mod tests;
