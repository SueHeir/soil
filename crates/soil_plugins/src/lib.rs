//! # soil_plugins — the infrastructure plugin group every SOIL tier shares
//!
//! DIRT (DEM), dev_soil_sph (SPH) and dev_soil_peri (peridynamics) each used to
//! carry a byte-for-byte copy of the same `CorePlugins` group. It contains no
//! physics, so it lives here once and the tiers re-export it.
//!
//! ```rust,ignore
//! use grass_app::prelude::*;
//!
//! let mut app = App::new();
//! app.add_plugins(soil_plugins::CorePlugins)  // input, comm, domain, neighbor, groups, run, output
//!    .add_plugins(MyPhysicsPlugins);          // the tier's integrator + physics
//! app.start();
//! ```

#![warn(missing_docs)]

use grass_app::prelude::*;

/// Core simulation infrastructure (no physics, no integrator).
///
/// Includes, in registration order:
/// - [`InputPlugin`](soil_core::InputPlugin) — CLI parsing and TOML config loading
/// - [`CommunicationPlugin`](soil_core::CommunicationPlugin) — MPI / single-process backend
/// - [`DomainPlugin`](soil_core::DomainPlugin) — domain decomposition, PBC, shrink-wrap
/// - [`NeighborPlugin`](soil_core::NeighborPlugin) — bin-based neighbour lists (and the shared [`TagIndex`](soil_core::TagIndex))
/// - [`GroupPlugin`](soil_core::GroupPlugin) — atom group definitions
/// - [`RunPlugin`](soil_core::RunPlugin) — run / cycle management
/// - [`PrintPlugin`](soil_print::PrintPlugin) — thermo, dump, restart output
///
/// Time integration is left to the tier (e.g.
/// `soil_verlet::VelocityVerletPlugin`), which also applies `[[run]] dt`.
pub struct CorePlugins;

impl PluginGroup for CorePlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(|app: &mut App| {
                app.set_warning_fn(soil_core::verlet_schedule_warnings);
            })
            .add(soil_core::InputPlugin)
            .add(soil_core::CommunicationPlugin)
            .add(soil_core::DomainPlugin)
            .add(soil_core::NeighborPlugin)
            .add(soil_core::GroupPlugin)
            .add(soil_core::RunPlugin)
            .add(soil_print::PrintPlugin)
    }
}
