//! Standard Verlet-style schedule phases for particle simulations (MD/DEM).
//!
//! These enums define the execution order for a velocity-Verlet integration loop.
//! Other domains (e.g. CFD) can define their own schedule phases by implementing
//! [`ScheduleSet`](grass_scheduler::ScheduleSet) on a custom enum (or using
//! `#[derive(ScheduleSet)]`).

/// Execution phase within each timestep (the run loop).
///
/// Systems are sorted first by their `ParticleSimScheduleSet` phase, then topologically within
/// each phase using `before`/`after` constraints. The phases execute in the order
/// listed below, mirroring a standard velocity-Verlet integration cycle:
///
/// | Index | Phase                    | Typical use                          |
/// |-------|--------------------------|--------------------------------------|
/// | 0     | `Setup`                  | Per-step bookkeeping                 |
/// | 1–3   | `Pre/Initial/PostInitialIntegration` | First half-step velocity update |
/// | 4–5   | `Pre/Exchange`           | Particle migration (MPI)             |
/// | 6–7   | `Pre/Neighbor`           | Neighbor-list rebuild                |
/// | 8–10  | `Pre/Force/PostForce`    | Force computation                    |
/// | 11–13 | `Pre/Final/PostFinalIntegration` | Second half-step velocity update |
#[derive(Debug, Clone, Copy, grass_derive::ScheduleSet)]
pub enum ParticleSimScheduleSet {
    /// Per-step bookkeeping (e.g., incrementing timestep counters).
    Setup,
    /// Runs before the initial (first half-step) integration.
    PreInitialIntegration,
    /// First half-step velocity update (velocity-Verlet first kick).
    InitialIntegration,
    /// Runs after the initial integration (e.g., position updates).
    PostInitialIntegration,
    /// Runs before particle exchange / migration.
    PreExchange,
    /// Particle exchange across MPI ranks or periodic boundaries.
    Exchange,
    /// Runs before neighbor-list construction.
    PreNeighbor,
    /// Neighbor-list build / rebuild.
    Neighbor,
    /// Runs before force computation (e.g., zeroing force arrays).
    PreForce,
    /// Main force computation phase.
    Force,
    /// Runs after force computation (e.g., force modifications, diagnostics).
    PostForce,
    /// Runs before the final (second half-step) integration.
    PreFinalIntegration,
    /// Second half-step velocity update (velocity-Verlet second kick).
    FinalIntegration,
    /// Runs after final integration (e.g., output, end-of-step fixes).
    PostFinalIntegration,
}

// `ScheduleSetupSet` lives in `grass_app` since it's a generic 3-phase
// setup ordering enum. Re-exported from `soil_core::lib` so existing
// consumers keep working unchanged.
pub use grass_app::ScheduleSetupSet;

/// Returns Verlet-specific schedule warnings for the given phase names.
///
/// This function is registered as the scheduler's `warning_fn` callback by a
/// downstream plugin group (e.g. `dirt_core`'s `CorePlugins`, which is **not**
/// defined in this crate) so that the generic scheduler can emit
/// particle-simulation warnings without knowing about Verlet phase names directly.
pub fn verlet_schedule_warnings(phase_names: &[&str]) -> Vec<String> {
    let mut warnings = Vec::new();

    let is_standard_schedule = phase_names.iter().any(|n| {
        matches!(
            *n,
            "Force"
                | "InitialIntegration"
                | "FinalIntegration"
                | "Setup"
                | "PreInitialIntegration"
                | "PostInitialIntegration"
                | "PreExchange"
                | "Exchange"
                | "PreNeighbor"
                | "Neighbor"
                | "PreForce"
                | "PostForce"
                | "PreFinalIntegration"
                | "PostFinalIntegration"
        )
    });

    if !is_standard_schedule {
        return warnings;
    }

    let has_phase = |name: &str| -> bool { phase_names.iter().any(|n| *n == name) };

    // Count distinct non-Setup phase names
    let update_count: usize = phase_names
        .iter()
        .filter(|n| **n != "Setup")
        .collect::<std::collections::HashSet<_>>()
        .len();

    // 1. No Force systems
    if update_count > 0 && !has_phase("Force") {
        warnings.push(
            "[Schedule Warning] No systems registered in the Force schedule set. \
             Forces will not be computed. Did you forget a force plugin?"
                .to_string(),
        );
    }

    // 2. Asymmetric Verlet
    let has_initial = has_phase("InitialIntegration") || has_phase("PreInitialIntegration");
    let has_final = has_phase("FinalIntegration") || has_phase("PostFinalIntegration");
    if has_initial && !has_final {
        warnings.push(
            "[Schedule Warning] InitialIntegration has systems but FinalIntegration is empty. \
             This produces an asymmetric Verlet integration."
                .to_string(),
        );
    } else if !has_initial && has_final {
        warnings.push(
            "[Schedule Warning] FinalIntegration has systems but InitialIntegration is empty. \
             This produces an asymmetric Verlet integration."
                .to_string(),
        );
    }

    // 3. No integrator
    if update_count > 2 && !has_initial && !has_final {
        warnings.push(
            "[Schedule Warning] Schedule has update systems but no integrator \
             (neither InitialIntegration nor FinalIntegration has systems). \
             Particles will not move."
                .to_string(),
        );
    }

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use grass_scheduler::prelude::*;

    fn schedule_probe() {}
    fn schedule_consumer() {}

    #[test]
    fn warns_no_force_systems() {
        let phases = vec!["InitialIntegration", "FinalIntegration"];
        let warnings = verlet_schedule_warnings(&phases);
        assert!(
            warnings.iter().any(|w| w.contains("Force")),
            "Expected warning about missing Force systems, got: {:?}",
            warnings
        );
    }

    #[test]
    fn warns_asymmetric_verlet() {
        let phases = vec!["InitialIntegration", "Force"];
        let warnings = verlet_schedule_warnings(&phases);
        assert!(
            warnings.iter().any(|w| w.contains("asymmetric")),
            "Expected warning about asymmetric Verlet, got: {:?}",
            warnings
        );
    }

    #[test]
    fn warns_no_integrator() {
        let phases = vec!["Force", "PostForce", "Exchange"];
        let warnings = verlet_schedule_warnings(&phases);
        assert!(
            warnings.iter().any(|w| w.contains("no integrator")),
            "Expected warning about no integrator, got: {:?}",
            warnings
        );
    }

    #[test]
    fn no_warnings_for_normal_schedule() {
        let phases = vec!["InitialIntegration", "Force", "FinalIntegration"];
        let warnings = verlet_schedule_warnings(&phases);
        assert!(
            warnings.is_empty(),
            "Expected no warnings, got: {:?}",
            warnings
        );
    }

    #[test]
    fn no_warnings_for_custom_schedule() {
        let phases = vec!["PhaseA", "PhaseB", "PhaseC"];
        let warnings = verlet_schedule_warnings(&phases);
        assert!(
            warnings.is_empty(),
            "Expected no warnings for custom schedule, got: {:?}",
            warnings
        );
    }

    #[test]
    fn public_system_keys_preserve_stable_schedule_names() {
        assert_eq!(crate::DOMAIN_READ_INPUT.name(), "domain_read_input");
        assert_eq!(crate::SHRINK_WRAP.name(), "shrink_wrap");
        assert_eq!(crate::PBC.name(), "pbc");
        assert_eq!(crate::NEIGHBOR_SETUP.name(), "neighbor_setup");
        assert_eq!(crate::DECIDE_REBUILD.name(), "decide_rebuild");
        assert_eq!(crate::SORT_ATOMS.name(), "sort_atoms");
        assert_eq!(crate::BORDERS.name(), "borders");
        assert_eq!(crate::EXCHANGE.name(), "exchange");
        assert_eq!(crate::REVERSE_SEND_FORCE.name(), "reverse_send_force");
        assert_eq!(crate::UPDATE_CYCLE.name(), "update_cycle");
    }

    #[test]
    fn required_typed_provider_is_validated() {
        let panic = std::panic::catch_unwind(|| {
            let mut scheduler = Scheduler::default();
            scheduler.add_update_system(
                schedule_probe.requires(crate::PBC),
                ParticleSimScheduleSet::PreExchange,
            );
            scheduler.organize_systems();
        })
        .expect_err("missing required typed provider must fail schedule validation");

        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic");
        assert!(
            message.contains("pbc"),
            "unexpected validation error: {message}"
        );
    }

    #[test]
    fn required_typed_provider_orders_successfully() {
        let mut scheduler = Scheduler::default();
        scheduler.add_update_system(
            schedule_probe.label(crate::PBC),
            ParticleSimScheduleSet::PreExchange,
        );
        scheduler.add_update_system(
            schedule_consumer.requires(crate::PBC),
            ParticleSimScheduleSet::PreExchange,
        );
        scheduler.organize_systems();
    }

    #[test]
    fn substrate_plugins_register_their_required_typed_schedule_seams() {
        let mut app = grass_app::App::new();
        app.add_plugins(crate::AtomPlugin)
            .add_plugins(crate::DomainPlugin)
            .add_plugins(crate::NeighborPlugin)
            .add_plugins(crate::CommunicationPlugin);

        app.prepare();
    }
}
