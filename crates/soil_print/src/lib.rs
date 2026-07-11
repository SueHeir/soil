//! Output systems for SOIL simulations: thermo printing, CSV/binary dump files,
//! restart file serialization, and VTP (ParaView) visualization output.
//!
//! # Overview
//!
//! This crate provides four output subsystems, each controlled by its own TOML
//! configuration section:
//!
//! | System  | TOML section  | Description                                    |
//! |---------|---------------|------------------------------------------------|
//! | Thermo  | `[thermo]`    | Periodic console output of simulation metrics  |
//! | Dump    | `[dump]`      | Per-atom CSV or binary snapshots                |
//! | Restart | `[restart]`   | Checkpoint files for resuming simulations       |
//! | VTP     | `[vtp]`       | ParaView-compatible `.vtp` visualization files  |
//!
//! All systems are registered automatically when [`PrintPlugin`] is added to the app.
//!
//! # TOML Configuration
//!
//! ```toml
//! [thermo]
//! # Columns to display (optional — defaults to step, atoms, ke, neighbors, walltime, stepps).
//! # Use "compute/group" syntax for group-filtered values, e.g. "ke/mobile".
//! # Built-in columns: step, atoms, ke, temp, neighbors, walltime, stepps.
//! # Any name pushed via Thermo::set() is also available.
//! columns = ["step", "atoms", "ke", "temp", "walltime", "stepps"]
//!
//! [dump]
//! # Write dump every N steps (0 = disabled, default: 0)
//! interval = 1000
//! # Output format: "text" (CSV) or "binary" (little-endian f64/u32)
//! format = "text"
//!
//! [restart]
//! # Write restart every N steps (0 = disabled, default: 0)
//! interval = 5000
//! # File format: "bincode" (compact binary, default) or "json" (human-readable)
//! format = "bincode"
//! # Read the latest restart file at startup (default: false)
//! read = false
//!
//! [vtp]
//! # Write VTP (ParaView) output every N steps (0 = disabled, default: 0)
//! interval = 500
//! ```
//!
//! # Extending Dump / VTP Output
//!
//! Plugins register additional per-atom columns via [`DumpRegistry`]. The
//! registry uses interior mutability, so a plugin takes a shared ref from its
//! `build()`:
//!
//! ```rust,ignore
//! let dump_reg = app.get_resource_ref::<DumpRegistry>().unwrap();
//! dump_reg.register_scalar("pressure", |atoms, registry| {
//!     // Return Vec<f64> of length atoms.nlocal
//!     vec![0.0; atoms.nlocal as usize]
//! });
//! ```
//!
//! See `examples/output_wiring.rs` for a runnable end-to-end demonstration that
//! registers a scalar column, a vector column, a custom dump format, and a
//! thermo value.

#![warn(missing_docs)]

mod config;
mod dump;
mod plugin;
mod registry;
mod restart;
mod thermo;
mod vtp;

pub use dump::{dump_atoms, sync_dump_box, DumpConfig};
pub use plugin::PrintPlugin;
pub use registry::{BoxInfo, DumpFrame, DumpRegistry};
pub use restart::{
    check_stage_end_save, preflight_restart_input, read_restart, write_restart, RestartConfig,
};
pub use thermo::{
    output_virial_to_thermo, print_thermo, setup_thermo, Thermo, ThermoColumn, ThermoConfig,
};
pub use vtp::{print_vtp, VtpConfig};

#[cfg(test)]
mod tests;
