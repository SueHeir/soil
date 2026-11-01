//! Uniform gravitational body force, `F = m g`, for any SOIL particle method.
//!
//! Lives in SOIL (not a physics tier) because it reads only base [`Atom`]
//! state: DEM (DIRT), SPH and peridynamics all share this one plugin.

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::Deserialize;

use soil_core::{Accum, Atom, Config, ParticleSimScheduleSet};

/// Gravitational acceleration configuration.
///
/// Applies a body force **F** = m **g** to every local atom each timestep,
/// where m is the atom's mass and **g** = `(gx, gy, gz)`.
///
/// # TOML Configuration
///
/// ```toml
/// [gravity]
/// gx = 0.0      # acceleration in x direction (default: 0.0)
/// gy = 0.0      # acceleration in y direction (default: 0.0)
/// gz = -9.81    # acceleration in z direction (default: -9.81)
/// ```
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct GravityConfig {
    /// Gravitational acceleration in x direction. Default: 0.0.
    #[serde(default)]
    pub gx: f64,
    /// Gravitational acceleration in y direction. Default: 0.0.
    #[serde(default)]
    pub gy: f64,
    /// Gravitational acceleration in z direction. Default: −9.81.
    #[serde(default = "default_gravity_gz")]
    pub gz: f64,
}

impl Default for GravityConfig {
    fn default() -> Self {
        GravityConfig {
            gx: 0.0,
            gy: 0.0,
            gz: -9.81,
        }
    }
}

fn default_gravity_gz() -> f64 {
    -9.81
}

/// Plugin that applies a constant gravitational body force (**F** = m**g**)
/// to all local atoms during the Force schedule phase.
pub struct GravityPlugin;

impl Plugin for GravityPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[gravity]
# Gravitational acceleration components (m/s^2)
gx = 0.0
gy = 0.0
gz = -9.81"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::try_load::<GravityConfig>(app, "gravity")
            .unwrap_or_else(|error| panic!("GravityPlugin failed to build: {error}"));
        app.add_update_system(apply_gravity, ParticleSimScheduleSet::Force);
    }

    fn try_build(&self, app: &mut App) -> Result<(), AppError> {
        Config::try_load::<GravityConfig>(app, "gravity")
            .map_err(|error| AppError::message(error.to_string()))?;
        app.add_update_system(apply_gravity, ParticleSimScheduleSet::Force);
        Ok(())
    }
}

/// Applies gravitational body force **F**_i = m_i **g** to all local atoms.
///
/// Ghost atoms (index ≥ `nlocal`) are not affected.
pub fn apply_gravity(mut atoms: ResMut<Atom>, gravity: Res<GravityConfig>) {
    // g = 0 adds nothing; skip the pass (and the write to every force row).
    if gravity.gx == 0.0 && gravity.gy == 0.0 && gravity.gz == 0.0 {
        return;
    }
    for i in 0..atoms.nlocal() as usize {
        atoms.force[i][0] += (atoms.mass[i] as f64 * gravity.gx) as Accum;
        atoms.force[i][1] += (atoms.mass[i] as f64 * gravity.gy) as Accum;
        atoms.force[i][2] += (atoms.mass[i] as f64 * gravity.gz) as Accum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gravity_atom(mass: f64) -> Atom {
        let mut atom = Atom::new();
        atom.dt = 1e-6;
        // This fixture has no registered extensions.
        unsafe { atom.push_test_atom(0, [0.0; 3], 0.001, mass) };
        // SAFETY: test fixture sets the ownership partition directly.
        unsafe { atom.set_layout_unchecked(1, atom.nghost(), 1) };
        atom
    }

    #[test]
    fn gravity_applies_force_equal_to_mg() {
        let mass = 0.5;
        let gz = -9.81;

        let mut app = App::new();
        app.add_resource(make_gravity_atom(mass));
        app.add_resource(GravityConfig {
            gx: 0.0,
            gy: 0.0,
            gz,
        });
        app.add_update_system(apply_gravity, ParticleSimScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        assert!((atom.force[0][0]).abs() < 1e-15);
        assert!((atom.force[0][1]).abs() < 1e-15);
        assert!((atom.force[0][2] - mass * gz).abs() < 1e-15);
    }

    #[test]
    fn gravity_skips_ghost_atoms() {
        let mass = 1.0;
        let gz = -9.81;

        let mut atom = make_gravity_atom(mass);
        // Add a ghost atom
        // This fixture has no registered extensions.
        unsafe { atom.push_test_atom(1, [0.0; 3], 0.001, mass) };
        atom.is_ghost[1] = true;
        // nlocal stays 1, ghost is index 1

        let mut app = App::new();
        app.add_resource(atom);
        app.add_resource(GravityConfig {
            gx: 0.0,
            gy: 0.0,
            gz,
        });
        app.add_update_system(apply_gravity, ParticleSimScheduleSet::Force);
        app.organize_systems();
        app.run();

        let atom = app.get_resource_ref::<Atom>().unwrap();
        // Local atom gets force
        assert!((atom.force[0][2] - mass * gz).abs() < 1e-15);
        // Ghost atom does not
        assert!((atom.force[1][2]).abs() < 1e-15);
    }
}
