//! Single-particle free fall under constant gravity, integrated with
//! [`VelocityVerletPlugin`].
//!
//! Velocity Verlet is *exact* for constant force, so after `N` steps the position
//! must equal the analytic parabola `z = z0 + v0·t + ½·g·t²` to machine
//! precision. This example asserts that, demonstrating:
//!
//! - assembling an `App` with a force system in the `Force` phase plus the
//!   integrator plugin,
//! - the `inv_mass = 0` ⇒ pinned-particle idiom (a second atom that never moves),
//! - that the integrator touches local atoms (`0..nlocal`) only.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example free_fall
//! ```

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use soil_core::{Atom, ParticleSimScheduleSet};
use soil_verlet::VelocityVerletPlugin;

const G: f64 = -9.81;

/// Constant gravity in z: `F = m·g`. Applied to local atoms only. A particle
/// with `inv_mass = 0` (infinite mass) feels a force here but cannot accelerate,
/// so it stays put — the pinned-particle idiom.
///
/// This example has no force-zeroing system (that lives in `soil_core`'s
/// `AtomPlugin`, which we don't add here), so we **set** the force rather than
/// `+=` into it — making the per-step force assignment idempotent.
fn gravity(mut atoms: ResMut<Atom>) {
    let nlocal = atoms.nlocal as usize;
    for i in 0..nlocal {
        let m = atoms.mass[i];
        atoms.force[i] = [0.0, 0.0, m * G];
    }
}

fn main() {
    let mut app = App::new();

    let mut atoms = Atom::new();
    atoms.dt = 1e-3;
    // Atom 0: a normal falling particle.
    atoms.push_test_atom(0, [0.0, 0.0, 10.0], 0.1, 2.0);
    atoms.vel[0] = [0.0, 0.0, 5.0]; // initial upward velocity v0
    // Prime the force at t=0. Velocity Verlet's first half-kick reads the force
    // present at the start of step 1; for the run to match the analytic parabola
    // exactly, F(0) must already be gravity (the integrator runs before our force
    // system on the very first step).
    atoms.force[0] = [0.0, 0.0, atoms.mass[0] * G];
    // Atom 1: pinned via infinite mass (inv_mass = 0) — never moves.
    atoms.push_test_atom(1, [1.0, 0.0, 10.0], 0.1, 1.0);
    atoms.inv_mass[1] = 0.0;
    atoms.nlocal = 2;
    atoms.natoms = 2;

    app.add_resource(atoms);
    // Gravity in the Force phase; the plugin's systems bracket it.
    app.add_update_system(gravity, ParticleSimScheduleSet::Force);
    app.add_plugins(VelocityVerletPlugin::new());
    app.organize_systems();

    let nsteps = 1000;
    for _ in 0..nsteps {
        app.run();
    }

    let atoms = app.get_resource_ref::<Atom>().unwrap();
    let dt = 1e-3;
    let t = nsteps as f64 * dt;

    // Analytic parabola for atom 0.
    let z0 = 10.0;
    let v0 = 5.0;
    let expected_z = z0 + v0 * t + 0.5 * G * t * t;
    let expected_vz = v0 + G * t;

    let got_z = atoms.pos[0][2];
    let got_vz = atoms.vel[0][2];
    println!("t = {t}: z = {got_z:.6} (expected {expected_z:.6}), vz = {got_vz:.6}");

    assert!(
        (got_z - expected_z).abs() < 1e-9,
        "free-fall z should match the exact parabola"
    );
    assert!(
        (got_vz - expected_vz).abs() < 1e-9,
        "free-fall vz should match the exact line"
    );

    // The infinite-mass atom never moved.
    assert_eq!(atoms.pos[1], [1.0, 0.0, 10.0], "inv_mass=0 atom is pinned");
    println!("pinned atom stayed at {:?}", atoms.pos[1]);
    println!("OK");
}
