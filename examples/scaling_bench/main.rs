//! SOIL parallel-scaling benchmark: a dense, frictionless granular gas run
//! through the stock substrate plugins (Atom, Domain, Neighbor, Communication,
//! velocity Verlet) with a DEM-shaped `AtomData` extension, timed per phase.
//!
//! The extension mirrors the column footprint of DIRT's `DemAtom` (a forward
//! `omega`, a reverse/zeroed `torque`, plus owner-only state) so ghost and
//! migration payloads are realistic. The force law is a linear spring-dashpot
//! normal contact with a small rolling-style torque so both forward and reverse
//! extension fields are exercised.
//!
//! ```sh
//! cargo build --release --example scaling_bench
//! # strong scaling: fixed 48^3 lattice
//! mpirun -n 4 target/release/examples/scaling_bench --global 48 --grid 2x2x1 --steps 400
//! # weak scaling: 24^3 particles per rank
//! mpirun -n 4 target/release/examples/scaling_bench --per-rank 24 --grid 2x2x1 --steps 400
//! ```
//!
//! Rank 0 prints a LAMMPS-style breakdown (min/avg/max over ranks) and a single
//! `SCALING` line for scripts.

use std::time::Instant;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use soil_core::{
    bin_neighbor_list, forward_comm_borders, Atom, AtomData, AtomDataRegistry, AtomPlugin,
    CommResource, CommunicationPlugin, Config, Domain, DomainPlugin, Neighbor, NeighborPlugin,
    ParticleSimScheduleSet, ParticleStore, ScheduleSetupSet, BORDERS, DECIDE_REBUILD,
    DOMAIN_READ_INPUT, EXCHANGE, REVERSE_SEND_FORCE, SORT_ATOMS,
};
use soil_derive::AtomData as DeriveAtomData;
use soil_verlet::VelocityVerletPlugin;

const RADIUS: f64 = 0.5;
const KN: f64 = 2000.0;
const GAMMA_N: f64 = 5.0;

/// DEM-shaped extension (same communicated widths as DIRT's `DemAtom`).
#[derive(DeriveAtomData, Default)]
struct BenchGrain {
    radius: Vec<f64>,
    density: Vec<f64>,
    inv_inertia: Vec<f64>,
    quaternion: Vec<[f64; 4]>,
    #[forward]
    omega: Vec<[f64; 3]>,
    ang_mom: Vec<[f64; 3]>,
    #[reverse]
    #[zero]
    torque: Vec<[f64; 3]>,
    #[forward]
    body_id: Vec<f64>,
}

#[derive(Clone)]
struct BenchCase {
    global: [usize; 3],
    grid: [i32; 3],
    spacing: f64,
    vmax: f64,
    steps: usize,
    warmup: usize,
    skin: f64,
    seed: u64,
}

// ── timers ───────────────────────────────────────────────────────────────────

const T_EXCHANGE: usize = 0;
const T_SORT: usize = 1;
const T_BORDERS: usize = 2;
const T_FORWARD: usize = 3;
const T_NEIGH: usize = 4;
const T_FORCE: usize = 5;
const T_REVERSE: usize = 6;
const T_DECIDE: usize = 7;
const NT: usize = 8;
const NAMES: [&str; NT] = [
    "exchange",
    "sort",
    "borders",
    "forward",
    "neigh_build",
    "force",
    "reverse",
    "decide",
];

#[derive(Default)]
struct Timers {
    on: bool,
    start: [Option<Instant>; NT],
    total: [f64; NT],
    rebuilds: usize,
}

fn begin(t: &mut Timers, k: usize) {
    if t.on {
        t.start[k] = Some(Instant::now());
    }
}
fn end(t: &mut Timers, k: usize) {
    if let Some(s) = t.start[k].take() {
        t.total[k] += s.elapsed().as_secs_f64();
    }
}

macro_rules! timer_pair {
    ($b:ident, $e:ident, $k:expr) => {
        fn $b(mut t: ResMut<Timers>) {
            begin(&mut t, $k);
        }
        fn $e(mut t: ResMut<Timers>) {
            end(&mut t, $k);
        }
    };
}
timer_pair!(b_exchange, e_exchange, T_EXCHANGE);
timer_pair!(b_sort, e_sort, T_SORT);
timer_pair!(b_borders, e_borders, T_BORDERS);
timer_pair!(b_forward, e_forward, T_FORWARD);
fn b_neigh(mut t: ResMut<Timers>) {
    if t.on {
        t.rebuilds += 1;
    }
    begin(&mut t, T_NEIGH);
}
fn e_neigh(mut t: ResMut<Timers>) {
    end(&mut t, T_NEIGH);
}
timer_pair!(b_reverse, e_reverse, T_REVERSE);
timer_pair!(b_decide, e_decide, T_DECIDE);

// ── physics ──────────────────────────────────────────────────────────────────

fn contact_force(
    mut atoms: ResMut<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
    mut t: ResMut<Timers>,
) {
    begin(&mut t, T_FORCE);
    let nlocal = atoms.nlocal() as usize;
    let mut grain = registry.expect_mut::<BenchGrain>("contact_force");
    let grain = &mut *grain;
    let offsets = &neighbor.neighbor_offsets;
    let indices = &neighbor.neighbor_indices;
    let pos = atoms.pos.as_ptr();
    let vel = atoms.vel.as_ptr();
    let force = unsafe { atoms.force.as_mut_ptr() };
    for i in 0..nlocal {
        let (s, e) = (offsets[i] as usize, offsets[i + 1] as usize);
        let pi = unsafe { *pos.add(i) };
        let vi = unsafe { *vel.add(i) };
        let ri = grain.radius[i];
        let wi = grain.omega[i];
        for &j in &indices[s..e] {
            let j = j as usize;
            let pj = unsafe { *pos.add(j) };
            let dx = pi[0] - pj[0];
            let dy = pi[1] - pj[1];
            let dz = pi[2] - pj[2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let rsum = ri + grain.radius[j];
            if r2 >= rsum * rsum || r2 == 0.0 {
                continue;
            }
            let r = r2.sqrt();
            let inv_r = 1.0 / r;
            let n = [dx * inv_r, dy * inv_r, dz * inv_r];
            let vj = unsafe { *vel.add(j) };
            let vn = (vi[0] - vj[0]) * n[0] + (vi[1] - vj[1]) * n[1] + (vi[2] - vj[2]) * n[2];
            let fmag = KN * (rsum - r) - GAMMA_N * vn;
            let f = [fmag * n[0], fmag * n[1], fmag * n[2]];
            unsafe {
                let fi = &mut *force.add(i);
                fi[0] += f[0];
                fi[1] += f[1];
                fi[2] += f[2];
                let fj = &mut *force.add(j);
                fj[0] -= f[0];
                fj[1] -= f[1];
                fj[2] -= f[2];
            }
            // Tiny rolling-style torque so omega/torque traffic is exercised.
            let wj = grain.omega[j];
            for k in 0..3 {
                let tq = 1e-3 * (wj[k] - wi[k]);
                grain.torque[i][k] += tq;
                grain.torque[j][k] -= tq;
            }
        }
    }
    end(&mut t, T_FORCE);
}

/// Rotational half of the integrator (owner rows only).
fn spin(atoms: Res<Atom>, registry: Res<AtomDataRegistry>) {
    let nlocal = atoms.nlocal() as usize;
    let dt = atoms.dt;
    let mut g = registry.expect_mut::<BenchGrain>("spin");
    let g = &mut *g;
    for i in 0..nlocal {
        for k in 0..3 {
            g.omega[i][k] += dt * g.torque[i][k] * g.inv_inertia[i];
        }
    }
}

// ── setup ────────────────────────────────────────────────────────────────────

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

struct CaseRes(BenchCase);

fn seed_lattice(
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    domain: Res<Domain>,
    case: Res<CaseRes>,
) {
    let c = &case.0;
    let total = (c.global[0] * c.global[1] * c.global[2]) as u64;
    let mut mine = Vec::new();
    for ix in 0..c.global[0] {
        for iy in 0..c.global[1] {
            for iz in 0..c.global[2] {
                let p = [
                    (ix as f64 + 0.5) * c.spacing,
                    (iy as f64 + 0.5) * c.spacing,
                    (iz as f64 + 0.5) * c.spacing,
                ];
                if (0..3)
                    .all(|d| p[d] >= domain.sub_domain_low[d] && p[d] < domain.sub_domain_high[d])
                {
                    let tag = (ix * c.global[1] * c.global[2] + iy * c.global[2] + iz) as u64;
                    mine.push((tag, p));
                }
            }
        }
    }
    let mass = 4.0 / 3.0 * std::f64::consts::PI * RADIUS.powi(3);
    let inertia = 0.4 * mass * RADIUS * RADIUS;
    {
        let mut store = ParticleStore::new(&mut atoms, &registry);
        store.reserve(mine.len());
        for _ in 0..mine.len() {
            store.push_default_local(total).expect("push local");
        }
    }
    let mut g = registry.expect_mut::<BenchGrain>("seed_lattice");
    for (i, (tag, p)) in mine.iter().enumerate() {
        // Velocity is a pure function of the global tag, so every decomposition
        // starts from the same state.
        let mut rng = Lcg(c.seed ^ tag.wrapping_mul(0x9E3779B97F4A7C15));
        rng.next();
        let v = [
            c.vmax * (2.0 * rng.next() - 1.0),
            c.vmax * (2.0 * rng.next() - 1.0),
            c.vmax * (2.0 * rng.next() - 1.0),
        ];
        atoms.tag[i] = *tag as u32;
        atoms.pos[i] = *p;
        atoms.vel[i] = v;
        atoms.mass[i] = mass;
        atoms.inv_mass[i] = 1.0 / mass;
        atoms.cutoff_radius[i] = RADIUS;
        g.radius[i] = RADIUS;
        g.density[i] = 1.0;
        g.inv_inertia[i] = 1.0 / inertia;
        g.quaternion[i] = [1.0, 0.0, 0.0, 0.0];
        g.omega[i] = [v[1], v[2], v[0]];
    }
}

fn parse_args() -> BenchCase {
    let mut case = BenchCase {
        global: [32, 32, 32],
        grid: [1, 1, 1],
        spacing: 1.02,
        vmax: 0.5,
        steps: 400,
        warmup: 50,
        skin: 1.15,
        seed: 12345,
    };
    let mut per_rank: Option<[usize; 3]> = None;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut k = 0;
    let triple = |s: &str| -> [usize; 3] {
        let v: Vec<usize> = s.split('x').map(|x| x.parse().unwrap()).collect();
        if v.len() == 1 {
            [v[0]; 3]
        } else {
            [v[0], v[1], v[2]]
        }
    };
    while k < args.len() {
        let val = args.get(k + 1).cloned().unwrap_or_default();
        match args[k].as_str() {
            "--global" => case.global = triple(&val),
            "--per-rank" => per_rank = Some(triple(&val)),
            "--grid" => {
                let g = triple(&val);
                case.grid = [g[0] as i32, g[1] as i32, g[2] as i32];
            }
            "--steps" => case.steps = val.parse().unwrap(),
            "--warmup" => case.warmup = val.parse().unwrap(),
            "--spacing" => case.spacing = val.parse().unwrap(),
            "--vmax" => case.vmax = val.parse().unwrap(),
            "--skin" => case.skin = val.parse().unwrap(),
            "--seed" => case.seed = val.parse().unwrap(),
            other => panic!("unknown argument {other}"),
        }
        k += 2;
    }
    if let Some(n) = per_rank {
        for d in 0..3 {
            case.global[d] = n[d] * case.grid[d] as usize;
        }
    }
    case
}

fn main() {
    let case = parse_args();
    let len = |d: usize| case.global[d] as f64 * case.spacing;
    let toml = format!(
        r#"
[domain]
x_low = 0.0
x_high = {}
y_low = 0.0
y_high = {}
z_low = 0.0
z_high = {}
boundary_x = "periodic"
boundary_y = "periodic"
boundary_z = "periodic"

[comm]
processors_x = {}
processors_y = {}
processors_z = {}

[neighbor]
skin_fraction = {}
every = 0
check = true
sort_every = 1000
newton = true
"#,
        len(0),
        len(1),
        len(2),
        case.grid[0],
        case.grid[1],
        case.grid[2],
        case.skin
    );

    let mut app = App::new();
    app.add_resource(Config::from_str(&toml));
    app.add_plugins(AtomPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(NeighborPlugin)
        .add_plugins(CommunicationPlugin);
    soil_core::register_atom_data!(&mut app, BenchGrain::default());
    app.add_plugins(VelocityVerletPlugin::new());
    app.add_resource(CaseRes(case.clone()));
    app.add_resource(Timers::default());

    app.add_setup_system(
        seed_lattice.after(DOMAIN_READ_INPUT),
        ScheduleSetupSet::Setup,
    );

    // Physics.
    app.add_update_system(contact_force, ParticleSimScheduleSet::Force);
    app.add_update_system(spin, ParticleSimScheduleSet::FinalIntegration);

    // Timers bracketing the substrate systems. Each marker carries the same
    // run condition as the system it brackets and a strict ordering on both
    // sides, so no other system can run between a begin/end pair.
    use soil_core::CommState::{CommunicateOnly, FullRebuild};
    let zero = "soil_core::atom::zero_all_forces";
    app.add_update_system(
        b_decide
            .before(DECIDE_REBUILD)
            .run_if(in_state(CommunicateOnly)),
        ParticleSimScheduleSet::PostInitialIntegration,
    );
    app.add_update_system(
        e_decide
            .after(DECIDE_REBUILD)
            .before(soil_core::remove_ghost_atoms)
            .before(zero),
        ParticleSimScheduleSet::PostInitialIntegration,
    );
    app.add_update_system(
        b_exchange.before(EXCHANGE).run_if(in_state(FullRebuild)),
        ParticleSimScheduleSet::Exchange,
    );
    app.add_update_system(e_exchange.after(EXCHANGE), ParticleSimScheduleSet::Exchange);
    app.add_update_system(
        b_sort.before(SORT_ATOMS),
        ParticleSimScheduleSet::PreNeighbor,
    );
    app.add_update_system(
        e_sort
            .label("e_sort")
            .after(SORT_ATOMS)
            .before(BORDERS)
            .before(forward_comm_borders),
        ParticleSimScheduleSet::PreNeighbor,
    );
    app.add_update_system(
        b_borders
            .after("e_sort")
            .before(BORDERS)
            .run_if(in_state(FullRebuild)),
        ParticleSimScheduleSet::PreNeighbor,
    );
    app.add_update_system(
        e_borders.after(BORDERS),
        ParticleSimScheduleSet::PreNeighbor,
    );
    app.add_update_system(
        b_forward
            .after("e_sort")
            .before(forward_comm_borders)
            .run_if(in_state(CommunicateOnly)),
        ParticleSimScheduleSet::PreNeighbor,
    );
    app.add_update_system(
        e_forward.after(forward_comm_borders),
        ParticleSimScheduleSet::PreNeighbor,
    );
    app.add_update_system(
        b_neigh
            .before(bin_neighbor_list)
            .run_if(in_state(FullRebuild)),
        ParticleSimScheduleSet::Neighbor,
    );
    app.add_update_system(
        e_neigh.after(bin_neighbor_list),
        ParticleSimScheduleSet::Neighbor,
    );
    app.add_update_system(
        b_reverse.before(REVERSE_SEND_FORCE),
        ParticleSimScheduleSet::PostForce,
    );
    app.add_update_system(
        e_reverse.after(REVERSE_SEND_FORCE),
        ParticleSimScheduleSet::PostForce,
    );

    let t_setup = Instant::now();
    app.prepare();
    let setup_s = t_setup.elapsed().as_secs_f64();
    {
        let cell = app
            .get_mut_resource(std::any::TypeId::of::<Atom>())
            .unwrap();
        let mut b = cell.borrow_mut();
        b.downcast_mut::<Atom>().unwrap().dt = 1e-3;
    }

    for _ in 0..case.warmup {
        app.run();
    }
    {
        let cell = app
            .get_mut_resource(std::any::TypeId::of::<Timers>())
            .unwrap();
        let mut b = cell.borrow_mut();
        let t = b.downcast_mut::<Timers>().unwrap();
        *t = Timers::default();
        t.on = true;
    }
    app.get_resource_ref::<CommResource>().unwrap().barrier();
    let t0 = Instant::now();
    for _ in 0..case.steps {
        app.run();
    }
    app.get_resource_ref::<CommResource>().unwrap().barrier();
    let loop_time = t0.elapsed().as_secs_f64();

    report(&app, &case, loop_time, setup_s);
    app.run_cleanup();
}

fn stats(comm: &CommResource, x: f64) -> (f64, f64, f64) {
    let n = comm.size() as f64;
    let sum = comm.all_reduce_sum_f64(x);
    let min = comm.all_reduce_min_f64(x);
    let max = -comm.all_reduce_min_f64(-x);
    (min, sum / n, max)
}

fn report(app: &App, case: &BenchCase, loop_time: f64, setup_s: f64) {
    let comm = app.get_resource_ref::<CommResource>().unwrap();
    let atoms = app.get_resource_ref::<Atom>().unwrap();
    let neighbor = app.get_resource_ref::<Neighbor>().unwrap();
    let t = app.get_resource_ref::<Timers>().unwrap();
    let rank = comm.rank();
    let size = comm.size();
    let natoms = comm.all_reduce_sum_f64(atoms.nlocal() as f64);
    let nghost = stats(&comm, atoms.nghost() as f64);
    let nlocal = stats(&comm, atoms.nlocal() as f64);
    let pairs = comm.all_reduce_sum_f64(neighbor.neighbor_indices.len() as f64);
    let rows: Vec<(f64, f64, f64)> = (0..NT).map(|k| stats(&comm, t.total[k])).collect();
    // Decomposition-independent state checksums (global sums over owners) so a
    // multi-rank run can be compared against the serial one.
    let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
    let grain = registry.expect::<BenchGrain>("report");
    let nl = atoms.nlocal() as usize;
    let mut ke = 0.0;
    let mut spin = 0.0;
    for i in 0..nl {
        let v = atoms.vel[i];
        ke += 0.5 * atoms.mass[i] * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        let w = grain.omega[i];
        spin += w[0] + 2.0 * w[1] + 3.0 * w[2];
    }
    let ke = comm.all_reduce_sum_f64(ke);
    let spin = comm.all_reduce_sum_f64(spin);
    let timed: f64 = t.total.iter().sum();
    let other = stats(&comm, loop_time - timed);
    if rank == 0 {
        let steps = case.steps as f64;
        println!(
            "\nscaling_bench: {} atoms, {} ranks ({}x{}x{}), {} steps, {} rebuilds",
            natoms, size, case.grid[0], case.grid[1], case.grid[2], case.steps, t.rebuilds
        );
        println!("  setup (insert + first neighbor build) {:.3} s", setup_s);
        println!(
            "  nlocal min/avg/max {:.0}/{:.0}/{:.0}  nghost min/avg/max {:.0}/{:.0}/{:.0}  pairs {}",
            nlocal.0, nlocal.1, nlocal.2, nghost.0, nghost.1, nghost.2, pairs
        );
        println!(
            "  loop time {:.4} s  =  {:.3} us/step  =  {:.4e} s/(atom*step)  ({:.3} Matom-step/s)",
            loop_time,
            loop_time / steps * 1e6,
            loop_time * size as f64 / (steps * natoms),
            natoms * steps / loop_time / 1e6,
        );
        println!("  checksum KE={:.15e} spin={:.15e}", ke, spin);
        println!(
            "  {:<12} {:>10} {:>10} {:>10} {:>7}",
            "phase", "min(s)", "avg(s)", "max(s)", "%loop"
        );
        for k in 0..NT {
            let (mn, av, mx) = rows[k];
            println!(
                "  {:<12} {:>10.4} {:>10.4} {:>10.4} {:>6.1}%",
                NAMES[k],
                mn,
                av,
                mx,
                100.0 * av / loop_time
            );
        }
        println!(
            "  {:<12} {:>10.4} {:>10.4} {:>10.4} {:>6.1}%",
            "other",
            other.0,
            other.1,
            other.2,
            100.0 * other.1 / loop_time
        );
        let mut line = format!(
            "SCALING setup_s={:.4} ranks={} grid={}x{}x{} natoms={} steps={} rebuilds={} loop_s={:.6} nghost_avg={:.0}",
            setup_s, size, case.grid[0], case.grid[1], case.grid[2], natoms, case.steps, t.rebuilds, loop_time, nghost.1
        );
        for k in 0..NT {
            line.push_str(&format!(
                " {}_avg={:.6} {}_max={:.6}",
                NAMES[k], rows[k].1, NAMES[k], rows[k].2
            ));
        }
        line.push_str(&format!(" other_avg={:.6}", other.1));
        println!("{line}");
    }
}
