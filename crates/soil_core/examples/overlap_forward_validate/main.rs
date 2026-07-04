//! overlap_forward_validate — SOIL-tier validation of
//! [`soil_core::forward_comm_overlap`] (roadmap step 4, comm/compute overlap).
//!
//! This is a **substrate-level** proof, with no DEM/granular logic: the only
//! physics is a Lennard-Jones pair potential (generic molecular dynamics), so
//! the same overlap machinery it exercises serves DEM, MD, SPH, and
//! Peridynamics alike. It complements dirt's `mpi_loadbalance_validate` (which
//! proves the *DEM* force overlaps) by proving the overlap correct one tier
//! down, at the point substrate itself.
//!
//! # What it proves
//!
//! A periodic LJ gas is seeded identically on every rank layout (a
//! deterministic splitmix64 lattice, so the global system is decomposition-
//! independent) and integrated with velocity Verlet. The per-step force is
//! computed two ways, selected by the `SOIL_OVERLAP` env var:
//!
//! - **baseline** (`SOIL_OVERLAP=0`): the substrate's standard per-step
//!   [`forward_comm`] refreshes the ghost halo (via the `CommunicationPlugin`),
//!   then a single pass computes every local atom's force.
//!
//! - **overlap** (`SOIL_OVERLAP=1`): [`forward_comm_overlap`] posts the ghost
//!   halo exchange and, *while it is in flight*, runs the interior force as its
//!   overlap closure — the atoms whose neighbours are all local
//!   ([`classify_boundary_atoms`] returns `false`), which therefore need no
//!   ghosts. Once the halo lands, the boundary atoms (mask `true`) are computed.
//!
//! The force uses a **full** neighbour list (`newton = false`) and writes only
//! to the owner `force[i]`, so each atom's force is a self-contained sum in
//! neighbour-list order. Splitting the outer loop by the boundary mask therefore
//! partitions the work exactly and preserves per-atom accumulation order: the
//! overlapped force is **bit-identical** to the single-pass force (well inside
//! the FP-associativity floor the acceptance allows).
//!
//! ## (1) Correctness
//! `sweep.py` runs both modes on ≥2 rank layouts, gathers every rank's final
//! `tag,x,y,z,vx,vy,vz`, and asserts the overlapped trajectory matches the
//! baseline to the FP floor — the ghost halo produced by `forward_comm_overlap`
//! reproduces the standard `forward_comm` halo, so the forces (and thus the
//! whole trajectory) agree.
//!
//! ## (2) Overlap
//! The interior force runs strictly *inside* `forward_comm_overlap`, between the
//! `Isend`/`Irecv` posting and `wait_all` (see
//! `grass_mpi::sendrecv_batch_overlap_f64_into`), and the boundary force is
//! issued only *after* the call returns (after the wait). This example asserts
//! the interior closure actually executed before the boundary pass, and prints
//! an `AUDIT` line with the interior/boundary atom counts so the driver can
//! confirm both populations are non-empty on a multi-rank run (genuine
//! compute happening while messages fly).
//!
//! Run (from the soil repo root):
//! ```bash
//! cargo build --release --example overlap_forward_validate
//! SOIL_OVERLAP=1 mpiexec -n 2 target/release/examples/overlap_forward_validate \
//!     crates/soil_core/examples/overlap_forward_validate/cfg_n2.toml
//! # or drive the whole comparison:
//! python3 crates/soil_core/examples/overlap_forward_validate/sweep.py
//! ```

use std::fs;
use std::io::Write as IoWrite;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use soil_core::{
    classify_boundary_atoms, forward_comm_overlap, Accum, Atom, AtomDataRegistry, AtomPlugin,
    CommBuffers, CommResource, CommTopology, CommunicationPlugin, Config, Domain, DomainPlugin,
    Neighbor, NeighborPlugin, ParticleSimScheduleSet, Real,
};

// ── Lennard-Jones parameters (reduced units: ε = σ = m = 1) ──────────────────
const EPSILON: f64 = 1.0;
const SIGMA: f64 = 1.0;
/// Interaction cutoff — standard 2.5σ.
const CUTOFF: f64 = 2.5 * SIGMA;
/// Per-atom neighbour radius. Pair cutoff = (r_i + r_j)·skin_fraction; with
/// r = 1.25σ and the config's `skin_fraction`, the neighbour cutoff is a
/// superset of the 2.5σ LJ cutoff (the hard cutoff is applied in the force).
const ATOM_RADIUS: f64 = 1.25 * SIGMA;

// ── Deterministic system generation (decomposition-independent) ──────────────
// NCELL is chosen large enough that each rank's subdomain is comfortably wider
// than the ghost halo (~2.9σ), so a non-empty *interior* population exists on
// every layout — genuine local compute to overlap with the in-flight halo.
const NCELL: usize = 16; // NCELL³ atoms on a lattice
/// Lattice spacing at the LJ well minimum r_min = 2^(1/6)σ, so the seed sits at
/// force balance and the gas heats gently rather than exploding.
fn lattice_spacing() -> f64 {
    2.0_f64.powf(1.0 / 6.0) * SIGMA
}
const VEL_SCALE: f64 = 0.35; // thermal velocity spread
const DT: f64 = 1.5e-3;
const NSTEPS: usize = 300;
/// Fixed PRNG seed — identical on every rank so the seeded global system is
/// decomposition-independent.
const SEED: u64 = 0x5011_0432_ABCD_1234;

/// splitmix64 — a tiny, fully deterministic PRNG. Every rank and every layout
/// draws the identical sequence, so the seeded global system is byte-for-byte
/// the same regardless of how many ranks it is decomposed across.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
/// Uniform in [-1, 1).
fn signed_unit(state: &mut u64) -> f64 {
    let u = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
    2.0 * u - 1.0
}

/// Audit counters proving the overlap actually happened (overlap mode only).
#[derive(Default)]
struct OverlapAudit {
    interior_ran_before_boundary: bool,
    interior_atoms: usize,
    boundary_atoms: usize,
    steps: usize,
}

/// Lennard-Jones force accumulated onto the **owner** atom only (`force[i]`),
/// over the full neighbour list, for the local atoms `i` selected by `want`.
///
/// Writing only `force[i]` (full list, `newton = false`) makes each atom's force
/// a self-contained sum in neighbour-list order, so partitioning `i` by the
/// boundary mask reproduces the single-pass force exactly.
fn lj_force_pass<F: Fn(usize) -> bool>(atoms: &mut Atom, neighbor: &Neighbor, want: F) {
    let nlocal = atoms.nlocal as usize;
    let cutoff2 = CUTOFF * CUTOFF;
    let sigma2 = SIGMA * SIGMA;
    for (i, j) in neighbor.pairs(nlocal) {
        if !want(i) {
            continue;
        }
        let pi = atoms.pos[i];
        let pj = atoms.pos[j];
        let d = [
            pi[0] as f64 - pj[0] as f64,
            pi[1] as f64 - pj[1] as f64,
            pi[2] as f64 - pj[2] as f64,
        ];
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if r2 >= cutoff2 || r2 == 0.0 {
            continue;
        }
        let sr2 = sigma2 / r2;
        let sr6 = sr2 * sr2 * sr2;
        let sr12 = sr6 * sr6;
        // fpair = (−dU/dr)/r, so F_vec = fpair · d (d = r_i − r_j).
        let fpair = 24.0 * EPSILON * (2.0 * sr12 - sr6) / r2;
        atoms.force[i][0] += (fpair * d[0]) as Accum;
        atoms.force[i][1] += (fpair * d[1]) as Accum;
        atoms.force[i][2] += (fpair * d[2]) as Accum;
    }
}

/// Baseline force: ghosts already refreshed by the substrate's standard
/// `forward_comm` (CommunicationPlugin), so a single pass over every local atom.
fn force_baseline(mut atoms: ResMut<Atom>, neighbor: Res<Neighbor>) {
    lj_force_pass(&mut atoms, &neighbor, |_| true);
}

/// Overlapped force: interior atoms (no ghost neighbours) computed as the
/// [`forward_comm_overlap`] closure *while the halo is in flight*, boundary
/// atoms once it lands.
fn force_overlap(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    comm: Res<CommResource>,
    topo: Res<CommTopology>,
    mut buffers: ResMut<CommBuffers>,
    mut audit: ResMut<OverlapAudit>,
) {
    let nlocal = atoms.nlocal as usize;
    // `mask[i] == true`  → atom i has ≥1 ghost neighbour (needs the halo).
    // `mask[i] == false` → all neighbours local (interior, halo-independent).
    let mask = classify_boundary_atoms(&neighbor, nlocal);

    let mut pool = std::mem::take(&mut buffers.forward_scratch);
    let mut interior_ran = false;
    {
        let neighbor_ref = &*neighbor;
        let mask_ref = &mask;
        // Interior force runs during the in-flight ghost exchange. It reads only
        // local positions (interior atoms have no ghost neighbours), so it is
        // independent of the halo landing.
        let mut interior = |a: &mut Atom| {
            lj_force_pass(a, neighbor_ref, |i| !mask_ref[i]);
            interior_ran = true;
        };
        forward_comm_overlap(&mut atoms, &registry, &topo, &**comm, &mut pool, &mut interior);
    }
    buffers.forward_scratch = pool;

    // Boundary force is issued only AFTER forward_comm_overlap returns — i.e.
    // after the wait — so it reads the freshly-landed ghost positions.
    lj_force_pass(&mut atoms, &neighbor, |i| mask[i]);

    audit.interior_ran_before_boundary = interior_ran;
    audit.interior_atoms = mask.iter().filter(|&&b| !b).count();
    audit.boundary_atoms = mask.iter().filter(|&&b| b).count();
    audit.steps += 1;
    assert!(
        interior_ran,
        "overlap invariant violated: the interior force closure did not execute \
         inside forward_comm_overlap before the boundary pass"
    );
}

// ── Velocity Verlet (kick-drift-kick), inlined so the example stays within the
// soil_core crate (soil_verlet depends on soil_core, so it cannot be a dep). ──
fn initial_integration(mut atoms: ResMut<Atom>) {
    let dt = atoms.dt as Accum;
    let nlocal = atoms.nlocal as usize;
    for i in 0..nlocal {
        let half_dt_over_m = 0.5 * dt * atoms.inv_mass[i] as Accum;
        for k in 0..3 {
            let v = atoms.vel[i][k] as Accum + half_dt_over_m * atoms.force[i][k];
            atoms.vel[i][k] = v as Real;
            atoms.pos[i][k] = (atoms.pos[i][k] as Accum + v * dt) as Real;
        }
    }
}
fn final_integration(mut atoms: ResMut<Atom>) {
    let dt = atoms.dt as Accum;
    let nlocal = atoms.nlocal as usize;
    for i in 0..nlocal {
        let half_dt_over_m = 0.5 * dt * atoms.inv_mass[i] as Accum;
        for k in 0..3 {
            atoms.vel[i][k] = (atoms.vel[i][k] as Accum + half_dt_over_m * atoms.force[i][k]) as Real;
        }
    }
}

fn main() {
    let cfg_path = std::env::args()
        .nth(1)
        .expect("usage: overlap_forward_validate <config.toml>");
    let overlap = std::env::var("SOIL_OVERLAP").map(|s| s == "1").unwrap_or(false);

    let toml = fs::read_to_string(&cfg_path)
        .unwrap_or_else(|e| panic!("cannot read config {cfg_path}: {e}"));
    let mut app = App::new();
    app.add_resource(Config::from_str(&toml));

    app.add_plugins(AtomPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(NeighborPlugin)
        .add_plugins(CommunicationPlugin);

    // Force system (mode-selected) in the Force phase; Verlet brackets it.
    if overlap {
        app.add_resource(OverlapAudit::default());
        app.add_update_system(force_overlap, ParticleSimScheduleSet::Force);
    } else {
        app.add_update_system(force_baseline, ParticleSimScheduleSet::Force);
    }
    app.add_update_system(initial_integration, ParticleSimScheduleSet::InitialIntegration);
    app.add_update_system(final_integration, ParticleSimScheduleSet::FinalIntegration);

    app.prepare();

    // ── Seed the deterministic global system, keeping only atoms this rank owns.
    let (sub_low, sub_high, ranks, rank) = {
        let domain = app.get_resource_ref::<Domain>().unwrap();
        let comm = app.get_resource_ref::<CommResource>().unwrap();
        (
            domain.sub_domain_low,
            domain.sub_domain_high,
            comm.size(),
            comm.rank(),
        )
    };
    let a = lattice_spacing();
    let mut local = 0u32;
    {
        let cell = app
            .get_mut_resource(std::any::TypeId::of::<Atom>())
            .expect("AtomPlugin registers the Atom resource");
        let mut binder = cell.borrow_mut();
        let atoms = binder.downcast_mut::<Atom>().unwrap();
        atoms.dt = DT;
        atoms.natoms = (NCELL * NCELL * NCELL) as u64;
        // One PRNG stream, drawn identically on every rank so all ranks agree on
        // the global system; each rank keeps only atoms inside its subdomain.
        let mut rng: u64 = SEED;
        for gx in 0..NCELL {
            for gy in 0..NCELL {
                for gz in 0..NCELL {
                    let tag = (gx * NCELL * NCELL + gy * NCELL + gz) as u32;
                    // Jitter the lattice slightly so the boundary populations are
                    // non-degenerate, but keep it deterministic.
                    let pos = [
                        (gx as f64 + 0.5) * a + 0.05 * a * signed_unit(&mut rng),
                        (gy as f64 + 0.5) * a + 0.05 * a * signed_unit(&mut rng),
                        (gz as f64 + 0.5) * a + 0.05 * a * signed_unit(&mut rng),
                    ];
                    let vel = [
                        VEL_SCALE * signed_unit(&mut rng),
                        VEL_SCALE * signed_unit(&mut rng),
                        VEL_SCALE * signed_unit(&mut rng),
                    ];
                    let owned = (0..3).all(|k| pos[k] >= sub_low[k] && pos[k] < sub_high[k]);
                    if owned {
                        atoms.push_test_atom(tag, pos, ATOM_RADIUS, 1.0);
                        let idx = atoms.tag.len() - 1;
                        atoms.vel[idx] = [vel[0] as Real, vel[1] as Real, vel[2] as Real];
                        local += 1;
                    }
                }
            }
        }
        atoms.nlocal = local;
    }

    // Time-step the gas.
    for _ in 0..NSTEPS {
        app.run();
    }

    // ── Dump this rank's final state (tag, position, velocity). ──
    let dir = "crates/soil_core/examples/overlap_forward_validate/data";
    fs::create_dir_all(dir).ok();
    let mode = if overlap { "overlap" } else { "baseline" };
    let path = format!("{dir}/state_{mode}_n{ranks}_rank{rank}.csv");
    {
        let atoms = app.get_resource_ref::<Atom>().unwrap();
        let mut f = fs::File::create(&path).expect("cannot create state csv");
        writeln!(f, "tag,x,y,z,vx,vy,vz").unwrap();
        for i in 0..atoms.nlocal as usize {
            writeln!(
                f,
                "{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                atoms.tag[i],
                atoms.pos[i][0], atoms.pos[i][1], atoms.pos[i][2],
                atoms.vel[i][0], atoms.vel[i][1], atoms.vel[i][2],
            )
            .unwrap();
        }
    }

    if rank == 0 {
        println!(
            "MODE={mode} ranks={ranks} nsteps={NSTEPS} natoms={}",
            NCELL * NCELL * NCELL
        );
    }
    if overlap {
        // Global interior/boundary populations, so the driver can confirm both
        // are non-empty (genuine compute overlapping in-flight messages).
        let audit = app.get_resource_ref::<OverlapAudit>().unwrap();
        let comm = app.get_resource_ref::<CommResource>().unwrap();
        let gi = comm.all_reduce_sum_f64(audit.interior_atoms as f64) as usize;
        let gb = comm.all_reduce_sum_f64(audit.boundary_atoms as f64) as usize;
        if rank == 0 {
            println!(
                "AUDIT ranks={ranks} interior_atoms={gi} boundary_atoms={gb} \
                 interior_ran_before_boundary={}",
                audit.interior_ran_before_boundary
            );
        }
    }

    // Run cleanup (drains `finalize_mpi`, i.e. MPI_Finalize) so every rank tears
    // MPI down cleanly and mpiexec exits 0. Must come after the last collective
    // (the audit all-reduce above).
    app.run_cleanup();
}
