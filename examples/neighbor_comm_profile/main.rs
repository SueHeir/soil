//! Reproducible SOIL neighbor/communication profiling harness.
//!
//! The executable seeds a deterministic lattice, runs the substrate systems
//! unchanged, and records timing, allocation, and communication counters for
//! the rebuild and communicate-only hot paths.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fs;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::Deserialize;
use soil_core::{
    bin_neighbor_list, borders, comm_setup, finalize_mpi, forward_comm_borders, neighbor_setup,
    reverse_send_force, sort_atoms_by_bin, Atom, AtomDataRegistry, CommBackend, CommBuffers,
    CommConfig, CommResource, CommState, CommTopology, Domain, DomainConfig, ForwardCommPrecision,
    Neighbor, NeighborConfig, ParticleSimScheduleSet, ScheduleSetupSet, SendRecvOp,
};
#[cfg(feature = "mpi_backend")]
use soil_core::{get_mpi_world, MpiCommBackend};

const PHASE_COUNT: usize = 6;
const NONE: usize = 0;
const SORT: usize = 1;
const BORDERS: usize = 2;
const FORWARD: usize = 3;
const REVERSE: usize = 4;
const BUILD: usize = 5;

static ACTIVE_PHASE: AtomicUsize = AtomicUsize::new(NONE);
static ALLOC_CALLS: [AtomicU64; PHASE_COUNT] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static ALLOC_BYTES: [AtomicU64; PHASE_COUNT] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static P2P_MESSAGES: [AtomicU64; PHASE_COUNT] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static P2P_F64: [AtomicU64; PHASE_COUNT] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static COLLECTIVES: [AtomicU64; PHASE_COUNT] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = ACTIVE_PHASE.load(Ordering::Relaxed);
        if p != NONE {
            ALLOC_CALLS[p].fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES[p].fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = ACTIVE_PHASE.load(Ordering::Relaxed);
        if p != NONE {
            ALLOC_CALLS[p].fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES[p].fetch_add(new_size as u64, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileCase {
    ncell: usize,
    iterations: usize,
    warmup: usize,
    spacing: f64,
    radius: f64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileConfig {
    case: ProfileCase,
    domain: DomainConfig,
    comm: CommConfig,
    neighbor: NeighborConfig,
}

#[derive(Clone, Copy, Default)]
struct PhaseTiming {
    calls: u64,
    total_ns: u128,
    min_ns: u128,
    max_ns: u128,
}

impl PhaseTiming {
    fn record(&mut self, ns: u128) {
        self.calls += 1;
        self.total_ns += ns;
        self.min_ns = if self.min_ns == 0 {
            ns
        } else {
            self.min_ns.min(ns)
        };
        self.max_ns = self.max_ns.max(ns);
    }
}

struct Profiler {
    start: [Option<Instant>; PHASE_COUNT],
    timing: [PhaseTiming; PHASE_COUNT],
    enabled: bool,
}

impl Default for Profiler {
    fn default() -> Self {
        Self {
            start: [None; PHASE_COUNT],
            timing: [PhaseTiming::default(); PHASE_COUNT],
            enabled: false,
        }
    }
}

fn phase_name(phase: usize) -> &'static str {
    match phase {
        SORT => "sort_atoms_by_bin",
        BORDERS => "borders",
        FORWARD => "forward_comm",
        REVERSE => "reverse_comm",
        BUILD => "neighbor_build",
        _ => "none",
    }
}

fn clear_atomic_counters() {
    for phase in 0..PHASE_COUNT {
        ALLOC_CALLS[phase].store(0, Ordering::Relaxed);
        ALLOC_BYTES[phase].store(0, Ordering::Relaxed);
        P2P_MESSAGES[phase].store(0, Ordering::Relaxed);
        P2P_F64[phase].store(0, Ordering::Relaxed);
        COLLECTIVES[phase].store(0, Ordering::Relaxed);
    }
}

fn start_phase(profiler: &mut Profiler, phase: usize) {
    if profiler.enabled {
        ACTIVE_PHASE.store(phase, Ordering::Relaxed);
        profiler.start[phase] = Some(Instant::now());
    }
}

fn end_phase(profiler: &mut Profiler, phase: usize) {
    if profiler.enabled {
        if let Some(start) = profiler.start[phase].take() {
            profiler.timing[phase].record(start.elapsed().as_nanos());
        }
        ACTIVE_PHASE.store(NONE, Ordering::Relaxed);
    }
}

fn start_sort(mut profiler: ResMut<Profiler>) {
    start_phase(&mut profiler, SORT);
}
fn end_sort(mut profiler: ResMut<Profiler>) {
    end_phase(&mut profiler, SORT);
}
fn start_borders(mut profiler: ResMut<Profiler>) {
    start_phase(&mut profiler, BORDERS);
}
fn end_borders(mut profiler: ResMut<Profiler>) {
    end_phase(&mut profiler, BORDERS);
}
fn start_forward(mut profiler: ResMut<Profiler>) {
    start_phase(&mut profiler, FORWARD);
}
fn end_forward(mut profiler: ResMut<Profiler>) {
    end_phase(&mut profiler, FORWARD);
}
fn start_reverse(mut profiler: ResMut<Profiler>) {
    start_phase(&mut profiler, REVERSE);
}
fn end_reverse(mut profiler: ResMut<Profiler>) {
    end_phase(&mut profiler, REVERSE);
}
fn start_build(mut profiler: ResMut<Profiler>) {
    start_phase(&mut profiler, BUILD);
}
fn end_build(mut profiler: ResMut<Profiler>) {
    end_phase(&mut profiler, BUILD);
}

fn fill_ghost_forces(mut atoms: ResMut<Atom>) {
    let nlocal = atoms.nlocal() as usize;
    for force in &mut atoms.force[nlocal..] {
        *force = [1.0, 2.0, 3.0];
    }
}

struct CountingComm {
    inner: Box<dyn CommBackend>,
}

impl CountingComm {
    fn p2p(messages: u64, f64s: u64) {
        let p = ACTIVE_PHASE.load(Ordering::Relaxed);
        if p != NONE {
            P2P_MESSAGES[p].fetch_add(messages, Ordering::Relaxed);
            P2P_F64[p].fetch_add(f64s, Ordering::Relaxed);
        }
    }

    fn collective() {
        let p = ACTIVE_PHASE.load(Ordering::Relaxed);
        if p != NONE {
            COLLECTIVES[p].fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl CommBackend for CountingComm {
    fn rank(&self) -> i32 {
        self.inner.rank()
    }
    fn size(&self) -> i32 {
        self.inner.size()
    }
    fn processor_decomposition(&self) -> [i32; 3] {
        self.inner.processor_decomposition()
    }
    fn processor_position(&self) -> [i32; 3] {
        self.inner.processor_position()
    }
    fn set_processor_grid(&mut self, decomp: [i32; 3], position: [i32; 3]) {
        self.inner.set_processor_grid(decomp, position);
    }
    fn all_reduce_sum_f64(&self, local: f64) -> f64 {
        Self::collective();
        self.inner.all_reduce_sum_f64(local)
    }
    fn all_reduce_min_f64(&self, local: f64) -> f64 {
        Self::collective();
        self.inner.all_reduce_min_f64(local)
    }
    fn barrier(&self) {
        self.inner.barrier();
    }
    fn send_f64(&self, dest: i32, buf: &[f64]) {
        Self::p2p((dest != -1) as u64, buf.len() as u64);
        self.inner.send_f64(dest, buf);
    }
    fn recv_f64(&self, source: i32) -> Vec<f64> {
        let out = self.inner.recv_f64(source);
        Self::p2p((source != -1) as u64, out.len() as u64);
        out
    }
    fn recv_f64_any(&self) -> Vec<f64> {
        let out = self.inner.recv_f64_any();
        Self::p2p(1, out.len() as u64);
        out
    }
    fn sendrecv_f64(&self, dest: i32, send_buf: &[f64], source: i32) -> Vec<f64> {
        let out = self.inner.sendrecv_f64(dest, send_buf, source);
        Self::p2p(
            (dest != -1) as u64 + (source != -1) as u64,
            send_buf.len() as u64 + out.len() as u64,
        );
        out
    }
    fn sendrecv_f64_into(&self, dest: i32, send_buf: &[f64], source: i32, recv_buf: &mut [f64]) {
        self.inner
            .sendrecv_f64_into(dest, send_buf, source, recv_buf);
        Self::p2p(
            (dest != -1) as u64 + (source != -1) as u64,
            send_buf.len() as u64 + recv_buf.len() as u64,
        );
    }
    fn sendrecv_batch_f64_into(&self, ops: &mut [SendRecvOp<'_>]) {
        self.inner.sendrecv_batch_f64_into(ops);
        let mut messages = 0;
        let mut f64s = 0;
        for op in ops {
            if op.dest != -1 {
                messages += 1;
                f64s += op.send_buf.len() as u64;
            }
            if op.source != -1 {
                messages += 1;
                f64s += op.recv_buf.len() as u64;
            }
        }
        Self::p2p(messages, f64s);
    }
    fn sendrecv_batch_overlap_f64_into(
        &self,
        ops: &mut [SendRecvOp<'_>],
        overlap: &mut dyn FnMut(),
    ) {
        self.inner.sendrecv_batch_overlap_f64_into(ops, overlap);
        let mut messages = 0;
        let mut f64s = 0;
        for op in ops {
            if op.dest != -1 {
                messages += 1;
                f64s += op.send_buf.len() as u64;
            }
            if op.source != -1 {
                messages += 1;
                f64s += op.recv_buf.len() as u64;
            }
        }
        Self::p2p(messages, f64s);
    }
}

fn make_comm() -> Box<dyn CommBackend> {
    #[cfg(feature = "mpi_backend")]
    {
        Box::new(MpiCommBackend::new(get_mpi_world()))
    }
    #[cfg(not(feature = "mpi_backend"))]
    {
        Box::new(soil_core::SingleProcessComm::new())
    }
}

fn push_rank_lattice(atoms: &mut Atom, domain: &Domain, case: &ProfileCase) {
    let margin = case.spacing;
    let mut local = 0u32;
    for ix in 0..case.ncell {
        for iy in 0..case.ncell {
            for iz in 0..case.ncell {
                let pos = [
                    margin + ix as f64 * case.spacing,
                    margin + iy as f64 * case.spacing,
                    margin + iz as f64 * case.spacing,
                ];
                if (0..3).all(|d| {
                    pos[d] >= domain.sub_domain_low[d] && pos[d] < domain.sub_domain_high[d]
                }) {
                    let tag = (ix * case.ncell * case.ncell + iy * case.ncell + iz) as u32;
                    atoms.push_test_atom(tag, pos, case.radius, 1.0);
                    local += 1;
                }
            }
        }
    }
    atoms.nlocal() = local;
    atoms.natoms() = (case.ncell * case.ncell * case.ncell) as u64;
}

fn configure_domain(cfg: &mut ProfileConfig) {
    let length = cfg.case.spacing * (cfg.case.ncell as f64 + 2.0);
    cfg.domain.x_low = 0.0;
    cfg.domain.y_low = 0.0;
    cfg.domain.z_low = 0.0;
    cfg.domain.x_high = length;
    cfg.domain.y_high = length;
    cfg.domain.z_high = length;
}

fn build_app(mut cfg: ProfileConfig) -> App {
    configure_domain(&mut cfg);

    let mut comm = make_comm();
    let decomp = [
        cfg.comm.processors_x,
        cfg.comm.processors_y,
        cfg.comm.processors_z,
    ];
    let rank = comm.rank();
    let pz = cfg.comm.processors_z;
    let py = cfg.comm.processors_y;
    let position = [rank / (py * pz), (rank / pz) % py, rank % pz];
    comm.set_processor_grid(decomp, position);

    let domain = soil_core::decompose_domain(&cfg.domain, &*comm);
    let mut atoms = Atom::new();
    push_rank_lattice(&mut atoms, &domain, &cfg.case);

    let mut neighbor = Neighbor::new();
    neighbor.skin_fraction = cfg.neighbor.skin_fraction;
    neighbor.bin_min_size = cfg.neighbor.bin_size;
    neighbor.every = cfg.neighbor.every;
    neighbor.check = cfg.neighbor.check;
    neighbor.sort_every = cfg.neighbor.sort_every;
    neighbor.newton = cfg.neighbor.newton;

    let mut app = App::new();
    app.add_resource(atoms);
    app.add_resource(AtomDataRegistry::new());
    app.add_resource(domain);
    app.add_resource(neighbor);
    app.add_resource(cfg.neighbor);
    app.add_resource(CommConfig {
        forward_precision: ForwardCommPrecision::F64,
        ..cfg.comm
    });
    app.add_resource(CommResource(Box::new(CountingComm { inner: comm })));
    app.add_resource(CommBuffers::default());
    app.add_resource(CommTopology {
        swap_directions: [[-1, -1, -1], [-1, -1, -1]],
        periodic_swap: [[0.0; 3], [0.0; 3]],
        swap_data: Vec::new(),
        maxneed: [0, 0, 0],
    });
    app.add_resource(CurrentState(CommState::FullRebuild));
    app.add_resource(Profiler::default());

    app.add_setup_system(comm_setup, ScheduleSetupSet::PostSetup);
    app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);

    app.add_update_system(
        start_sort.before("profile_sort"),
        ParticleSimScheduleSet::PreNeighbor,
    );
    app.add_update_system(
        sort_atoms_by_bin
            .label("profile_sort")
            .before("profile_sort_end"),
        ParticleSimScheduleSet::PreNeighbor,
    );
    app.add_update_system(
        end_sort.label("profile_sort_end"),
        ParticleSimScheduleSet::PreNeighbor,
    );

    app.add_update_system(
        start_borders
            .label("profile_borders_start")
            .after("profile_sort_end")
            .run_if(in_state(CommState::FullRebuild)),
        ParticleSimScheduleSet::PreNeighbor,
    );
    app.add_update_system(
        borders
            .label("profile_borders")
            .after("profile_borders_start")
            .before("profile_borders_end")
            .run_if(in_state(CommState::FullRebuild)),
        ParticleSimScheduleSet::PreNeighbor,
    );
    app.add_update_system(
        end_borders
            .label("profile_borders_end")
            .run_if(in_state(CommState::FullRebuild)),
        ParticleSimScheduleSet::PreNeighbor,
    );

    app.add_update_system(
        start_forward
            .label("profile_forward_start")
            .after("profile_sort_end")
            .run_if(in_state(CommState::CommunicateOnly)),
        ParticleSimScheduleSet::PreNeighbor,
    );
    app.add_update_system(
        forward_comm_borders
            .label("profile_forward")
            .after("profile_forward_start")
            .before("profile_forward_end")
            .run_if(in_state(CommState::CommunicateOnly)),
        ParticleSimScheduleSet::PreNeighbor,
    );
    app.add_update_system(
        end_forward
            .label("profile_forward_end")
            .run_if(in_state(CommState::CommunicateOnly)),
        ParticleSimScheduleSet::PreNeighbor,
    );

    app.add_update_system(fill_ghost_forces, ParticleSimScheduleSet::PreForce);
    app.add_update_system(
        start_reverse.before("profile_reverse"),
        ParticleSimScheduleSet::PostForce,
    );
    app.add_update_system(
        reverse_send_force
            .label("profile_reverse")
            .before("profile_reverse_end"),
        ParticleSimScheduleSet::PostForce,
    );
    app.add_update_system(
        end_reverse.label("profile_reverse_end"),
        ParticleSimScheduleSet::PostForce,
    );

    app.add_update_system(
        start_build
            .before("profile_build")
            .run_if(in_state(CommState::FullRebuild)),
        ParticleSimScheduleSet::Neighbor,
    );
    app.add_update_system(
        bin_neighbor_list
            .label("profile_build")
            .before("profile_build_end")
            .run_if(in_state(CommState::FullRebuild)),
        ParticleSimScheduleSet::Neighbor,
    );
    app.add_update_system(
        end_build.label("profile_build_end"),
        ParticleSimScheduleSet::Neighbor,
    );
    app.add_cleanup(finalize_mpi);
    app.organize_systems();
    app.setup();
    app
}

fn force_full_rebuild(app: &mut App) {
    let cell = app
        .get_mut_resource(std::any::TypeId::of::<CurrentState<CommState>>())
        .unwrap();
    let mut binder = cell.borrow_mut();
    let state = binder
        .downcast_mut::<CurrentState<CommState>>()
        .expect("CurrentState<CommState> resource is present");
    state.0 = CommState::FullRebuild;
}

fn print_summary(app: &App, cfg: &ProfileConfig) {
    let comm = app.get_resource_ref::<CommResource>().unwrap();
    let rank = comm.rank();
    let size = comm.size();
    let atoms = app.get_resource_ref::<Atom>().unwrap();
    let neighbor = app.get_resource_ref::<Neighbor>().unwrap();
    let profiler = app.get_resource_ref::<Profiler>().unwrap();
    for phase in [SORT, BORDERS, FORWARD, REVERSE, BUILD] {
        let t = profiler.timing[phase];
        let calls = t.calls.max(1);
        println!(
            "PROFILE rank={} size={} layout={}x{}x{} phase={} calls={} total_ns={} mean_ns={} min_ns={} max_ns={} alloc_calls={} alloc_bytes={} p2p_messages={} p2p_f64={} collectives={} nlocal={} nghost={} pairs={} iterations={} warmup={}",
            rank,
            size,
            cfg.comm.processors_x,
            cfg.comm.processors_y,
            cfg.comm.processors_z,
            phase_name(phase),
            t.calls,
            t.total_ns,
            t.total_ns / calls as u128,
            t.min_ns,
            t.max_ns,
            ALLOC_CALLS[phase].load(Ordering::Relaxed),
            ALLOC_BYTES[phase].load(Ordering::Relaxed),
            P2P_MESSAGES[phase].load(Ordering::Relaxed),
            P2P_F64[phase].load(Ordering::Relaxed),
            COLLECTIVES[phase].load(Ordering::Relaxed),
            atoms.nlocal(),
            atoms.nghost(),
            neighbor.neighbor_indices.len(),
            cfg.case.iterations,
            cfg.case.warmup,
        );
    }
}

fn main() {
    let cfg_path = std::env::args()
        .nth(1)
        .expect("usage: neighbor_comm_profile <config.toml>");
    let raw =
        fs::read_to_string(&cfg_path).unwrap_or_else(|e| panic!("cannot read {cfg_path}: {e}"));
    let cfg: ProfileConfig =
        toml::from_str(&raw).unwrap_or_else(|e| panic!("cannot parse {cfg_path}: {e}"));

    let mut app = build_app(cfg.clone());
    for _ in 0..cfg.case.warmup {
        force_full_rebuild(&mut app);
        app.run();
        app.run();
    }

    {
        let profiler = app
            .get_mut_resource(std::any::TypeId::of::<Profiler>())
            .unwrap();
        let mut profiler = profiler.borrow_mut();
        let profiler = profiler.downcast_mut::<Profiler>().unwrap();
        *profiler = Profiler::default();
        profiler.enabled = true;
    }
    clear_atomic_counters();

    for _ in 0..cfg.case.iterations {
        force_full_rebuild(&mut app);
        app.run();
        app.run();
    }

    {
        let comm = app.get_resource_ref::<CommResource>().unwrap();
        comm.barrier();
    }
    print_summary(&app, &cfg);
    app.run_cleanup();
}
