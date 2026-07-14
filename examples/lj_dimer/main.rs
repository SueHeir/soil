//! Lennard-Jones **molecular dynamics** dimer on the SOIL substrate — a *second*
//! particle method (not DEM) proving the substrate is method-agnostic.
//!
//! Where `free_fall.rs` exercises only single-particle integration, this example
//! rides the **full** particle-method machinery with a pairwise potential:
//!
//! - the **neighbor list** (`Neighbor::pairs`) finds the interacting pair,
//! - **ghost communication** carries the partner across a periodic face (the
//!   dimer straddles the x-boundary, so one atom's neighbor is a *ghost* image;
//!   forward comm replicates its position, reverse comm sums the ghost's force
//!   — and its potential-energy contribution — back onto the owner),
//! - a **registered `AtomData` column** (`LjEnergy`, a `#[reverse] #[zero]`
//!   per-atom potential-energy accumulator) proves a custom, non-DEM column
//!   flows through the substrate's comm path,
//! - **`VelocityVerletPlugin`** integrates the motion.
//!
//! There is **no DEM code** here: no contact model, no rotation/torque, no
//! clumps, no friction — only a Lennard-Jones pair potential and viscous
//! damping (a generic drag, not a granular contact law).
//!
//! # The closed-form check
//!
//! For two atoms interacting through the Lennard-Jones potential
//!
//! ```text
//!   U(r) = 4ε[(σ/r)¹² − (σ/r)⁶]
//! ```
//!
//! the pair force `−dU/dr` vanishes at the well minimum
//!
//! ```text
//!   r_min = 2^(1/6) · σ      and      U(r_min) = −ε   (exactly).
//! ```
//!
//! Starting the dimer off-equilibrium and adding light viscous damping, the
//! velocity-Verlet trajectory relaxes to force balance. This example asserts
//! that the relaxed separation equals `2^(1/6)·σ` and that the total potential
//! energy — reconstructed from the reverse-communicated `LjEnergy` column —
//! equals `−ε`, both to tight tolerance.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example lj_dimer
//! ```

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use soil_core::{
    register_atom_data, Atom, AtomData, AtomDataRegistry, AtomPlugin, CommunicationPlugin, Config,
    Domain, DomainPlugin, Neighbor, NeighborPlugin, ParticleSimScheduleSet,
};
use soil_derive::AtomData;
use soil_verlet::VelocityVerletPlugin;

// ── Lennard-Jones parameters (reduced units: ε = σ = m = 1) ──────────────────
const EPSILON: f64 = 1.0;
const SIGMA: f64 = 1.0;
/// Interaction cutoff — standard 2.5σ. The dimer (~1.12σ) sits well inside, so
/// truncation neither shifts the equilibrium nor the well-depth check.
const CUTOFF: f64 = 2.5 * SIGMA;
/// Viscous drag coefficient `F -= γ·v`. A generic damping that drains kinetic
/// energy so the dimer relaxes to force balance — *not* a granular/DEM law.
const GAMMA: f64 = 3.0;

/// A registered per-atom **potential-energy** column — the non-DEM `AtomData`
/// extension this example rides through the substrate. Marked `#[reverse]` so a
/// ghost's accumulated share is summed back onto its owner, and `#[zero]` so it
/// resets each step. This is exactly the mechanism DEM uses for torque, but the
/// quantity here (potential energy) has nothing to do with DEM.
#[derive(AtomData, Default)]
struct LjEnergy {
    #[reverse]
    #[zero]
    pe: Vec<f64>,
}

/// Lennard-Jones pair forces over the neighbor list, plus viscous damping.
///
/// Runs in the `Force` phase, bracketed by the Verlet half-kicks. Iterates the
/// substrate's half neighbor list; `j` may be a local atom or a ghost image
/// (its position already carries the periodic shift, so `pos[i] − pos[j]` is the
/// correct minimum-image separation with no manual PBC here). Newton's third law
/// writes equal-and-opposite forces onto `i` and `j`; when `j` is a ghost, the
/// substrate's reverse comm returns that contribution to the owner.
fn lj_force_and_damp(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
) {
    let nlocal = atoms.nlocal as usize;
    let ntot = atoms.len();
    let cutoff2 = CUTOFF * CUTOFF;
    let sigma2 = SIGMA * SIGMA;

    let mut energy = registry.expect_mut::<LjEnergy>("lj_force_and_damp");
    // Ghosts get an `LjEnergy` slot from the substrate's migration pack/unpack,
    // and `#[zero]` sizes the column to len() each step; resize defensively so
    // ghost indices are always writable.
    if energy.pe.len() < ntot {
        energy.pe.resize(ntot, 0.0);
    }

    // Pairwise Lennard-Jones over the neighbor list.
    for (i, j) in neighbor.pairs(nlocal) {
        let pi = atoms.pos[i];
        let pj = atoms.pos[j];
        let d = [pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if r2 >= cutoff2 || r2 == 0.0 {
            continue;
        }
        let sr2 = sigma2 / r2;
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;
        // fpair = (−dU/dr)/r, so F_vec = fpair · d  (d = r_i − r_j).
        let fpair = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
        let f = [fpair * d[0], fpair * d[1], fpair * d[2]];
        atoms.force[i][0] += f[0];
        atoms.force[i][1] += f[1];
        atoms.force[i][2] += f[2];
        atoms.force[j][0] -= f[0];
        atoms.force[j][1] -= f[1];
        atoms.force[j][2] -= f[2];
        // Split the pair energy so each partner (owner or ghost) carries half.
        let u = 4.0 * EPSILON * (sr12 - sr6);
        energy.pe[i] += 0.5 * u;
        energy.pe[j] += 0.5 * u;
    }

    // Viscous damping on local atoms only (ghosts are read-only halo copies).
    for i in 0..nlocal {
        let v = atoms.vel[i];
        atoms.force[i][0] -= GAMMA * v[0];
        atoms.force[i][1] -= GAMMA * v[1];
        atoms.force[i][2] -= GAMMA * v[2];
    }
}

/// Minimum-image separation between the two atoms along a periodic box.
fn min_image_separation(atoms: &Atom, domain: &Domain) -> f64 {
    let a = atoms.pos[0];
    let b = atoms.pos[1];
    let mut r2 = 0.0;
    for k in 0..3 {
        let l = domain.boundaries_high[k] - domain.boundaries_low[k];
        let mut dx = a[k] - b[k];
        // wrap into [-L/2, L/2]
        while dx > 0.5 * l {
            dx -= l;
        }
        while dx < -0.5 * l {
            dx += l;
        }
        r2 += dx * dx;
    }
    r2.sqrt()
}

fn main() {
    // ── Declarative config: a periodic box large enough that the 2.5σ ghost
    // halo never wraps onto itself, plus a neighbor skin. Pure data, no logic. ──
    let toml = "\
[domain]
x_low = 0.0
x_high = 10.0
y_low = 0.0
y_high = 10.0
z_low = 0.0
z_high = 10.0
boundary_x = \"periodic\"
boundary_y = \"periodic\"
boundary_z = \"periodic\"

[neighbor]
skin_fraction = 1.3
";

    let mut app = App::new();
    // Config must be present before the plugins' build() so their setup reads it.
    app.add_resource(Config::from_str(toml));

    // The four substrate plugins (single-process comm backend — serial, no MPI).
    app.add_plugins(AtomPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(NeighborPlugin)
        .add_plugins(CommunicationPlugin);

    // Register the non-DEM per-atom column (must come after AtomPlugin).
    register_atom_data!(&mut app, LjEnergy::default());

    // Pairwise LJ + damping in the Force phase; Verlet brackets it.
    app.add_update_system(lj_force_and_damp, ParticleSimScheduleSet::Force);
    app.add_plugins(VelocityVerletPlugin::new());

    // prepare(): organize systems + run setup (domain box, neighbor/ghost cutoff).
    app.prepare();

    // ── Seed the dimer straddling the x = 0 / x = 10 periodic face, so the pair
    // is found *through a ghost image* — exercising forward + reverse ghost comm.
    // Atom 0 at x = 0.5, atom 1 at x = 9.4 → minimum-image separation 1.1σ, just
    // off the r_min ≈ 1.122σ well minimum. ──
    {
        let cell = app
            .get_mut_resource(std::any::TypeId::of::<Atom>())
            .expect("AtomPlugin registers the Atom resource");
        let mut binder = cell.borrow_mut();
        let atoms = binder.downcast_mut::<Atom>().unwrap();
        atoms.dt = 2e-3;
        unsafe { atoms.push_test_atom(0, [0.5, 5.0, 5.0], CUTOFF, 1.0) };
        unsafe { atoms.push_test_atom(1, [9.4, 5.0, 5.0], CUTOFF, 1.0) };
        atoms.nlocal = 2;
        atoms.natoms = 2;
    }

    // Relax to force balance.
    let nsteps = 40_000;
    for _ in 0..nsteps {
        app.run();
    }

    // ── Validate against the closed forms. ──
    let atoms = app.get_resource_ref::<Atom>().unwrap();
    let domain = app.get_resource_ref::<Domain>().unwrap();
    let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();

    let r_final = min_image_separation(&atoms, &domain);
    let r_min = 2.0_f64.powf(1.0 / 6.0) * SIGMA;

    // Total potential energy from the reverse-communicated per-atom column.
    let energy = registry.expect::<LjEnergy>("lj_dimer main");
    let nlocal = atoms.nlocal as usize;
    let total_pe: f64 = energy.pe[..nlocal].iter().sum();
    let expected_pe = -EPSILON;

    let nghost = atoms.nghost;

    println!("SOIL LJ-MD dimer (a non-DEM particle method on the SOIL substrate)");
    println!(
        "  ghosts present at end : {nghost}  (pair is ghost-mediated across the periodic face)"
    );
    println!("  relaxed separation    : r = {r_final:.9} σ");
    println!("  analytic well minimum : r_min = 2^(1/6) σ = {r_min:.9}");
    println!("  |r − r_min|           : {:.3e}", (r_final - r_min).abs());
    println!("  total potential energy: U = {total_pe:.9} ε   (from reverse-comm LjEnergy column)");
    println!("  analytic well depth   : U(r_min) = −ε = {expected_pe:.9}");
    println!(
        "  |U − (−ε)|            : {:.3e}",
        (total_pe - expected_pe).abs()
    );

    // The dimer must have interacted across the periodic boundary via a ghost.
    assert!(
        nghost > 0,
        "expected ghost atoms — the pair should be ghost-mediated"
    );

    // Equilibrium separation equals 2^(1/6)·σ.
    assert!(
        (r_final - r_min).abs() < 1e-6,
        "relaxed separation {r_final} should equal r_min = {r_min} (2^(1/6)·σ)"
    );

    // Well-depth (validates the reverse-communicated energy column too).
    assert!(
        (total_pe - expected_pe).abs() < 1e-6,
        "total potential energy {total_pe} should equal U(r_min) = −ε = {expected_pe}"
    );

    println!("OK — SOIL substrate served a second particle method (Lennard-Jones MD).");
}
