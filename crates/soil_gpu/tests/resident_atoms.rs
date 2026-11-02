//! Atoms stepped on the GPU, read and written by ordinary CPU systems in a real schedule.

use std::any::TypeId;
use std::sync::Mutex;

use grass_app::prelude::*;
use grass_gpu::GpuDevice;
use grass_scheduler::prelude::*;
use grass_scheduler::CoherenceRegistry;
use soil_core::{Atom, AtomDataRegistry, Real};
use soil_gpu::resident::to_gpu;
use soil_gpu::{GpuReal, GpuState, Grid, ResidentAtoms, ResidentAtomsPlugin};

const DT: f32 = 1.0e-3;
const WINDOW: usize = 10;
const G: f32 = -9.81;

#[derive(Debug, Clone, Copy, grass_derive::ScheduleSet)]
enum Phase {
    Kernels,
    Cpu,
}

fn gpu_step(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    mut resident: ResMut<ResidentAtoms>,
    mut coherence: ResMut<CoherenceRegistry>,
) {
    if resident.state.is_none() {
        let n = atoms.len();
        let pos = to_gpu(&atoms.pos, n);
        let grid = Grid::from_positions(&pos, 1.0);
        let device = GpuDevice::default_device().unwrap();
        let mut state = GpuState::new(device, n, grid);
        state.set_params(DT as GpuReal, [0.0, 0.0, G as GpuReal]);
        state.set_state(&pos, &to_gpu(&atoms.vel, n), &vec![1.0; n]);
        state.run_steps(WINDOW);
        resident.state = Some(state);
    } else {
        let stale_force = resident.begin_kernels(&mut coherence, &atoms, &registry);
        let state = resident.state.as_mut().unwrap();
        if stale_force {
            state.run_steps(WINDOW);
        } else {
            state.run_steps_continue(WINDOW);
        }
    }
    ResidentAtoms::end_kernels(&mut coherence);
}

/// Keeps the last z it saw. It has never heard of a GPU.
#[derive(Default)]
struct Seen(Vec<Real>);
fn reader(atoms: Res<Atom>, mut seen: ResMut<Seen>) {
    seen.0.push(atoms.pos[0][2]);
}

/// Halves every velocity. It has never heard of a GPU either.
fn writer(mut atoms: ResMut<Atom>) {
    for v in atoms.vel.iter_mut() {
        v[2] *= 0.5;
    }
}

/// The device and its transfer counts are shared by the process, so tests take turns.
static TURN: Mutex<()> = Mutex::new(());

fn app_with_one_atom() -> Option<App> {
    if let Err(e) = GpuDevice::default_device() {
        eprintln!("skipping: {e}");
        return None;
    }
    let mut app = App::new();
    let mut atoms = Atom::new();
    atoms.dt = DT as f64;
    // SAFETY: this registry has no extensions to keep aligned.
    unsafe { atoms.push_test_atom(1, [0.0, 0.0, 100.0], 0.5, 1.0) };
    app.add_resource(atoms);
    app.add_resource(AtomDataRegistry::new());
    app.add_plugins(ResidentAtomsPlugin);
    app.add_resource(Seen::default());
    app.add_update_system(gpu_step, Phase::Kernels);
    Some(app)
}

#[test]
fn cpu_reader_sees_the_gpu_trajectory_every_window() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut app) = app_with_one_atom() else {
        return;
    };
    app.add_update_system(reader, Phase::Cpu);
    app.organize_systems();
    for _ in 0..3 {
        app.run();
    }
    let seen = app.get_resource_ref::<Seen>().unwrap();
    let t = |w: f32| w * WINDOW as f32 * DT;
    for (w, z) in seen.0.iter().enumerate() {
        // Velocity Verlet is exact for a constant force.
        let tw = t(w as f32 + 1.0);
        let exact = 100.0 + 0.5 * G * tw * tw;
        assert!(
            ((*z as f32) - exact).abs() < 1e-3,
            "window {w}: {z} vs {exact}"
        );
    }
    let coherence = app.get_resource_ref::<CoherenceRegistry>().unwrap();
    assert_eq!(coherence.syncs(TypeId::of::<Atom>()), 3);
}

#[test]
fn a_run_with_no_cpu_reader_makes_no_transfers_after_the_first_upload() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut app) = app_with_one_atom() else {
        return;
    };
    app.organize_systems();
    app.run();
    let device = app.get_resource_ref::<GpuDevice>().unwrap().clone();
    let before = device.counts().snapshot();
    for _ in 0..5 {
        app.run();
    }
    let moved = device.counts().snapshot().since(&before);
    assert_eq!((moved.uploads, moved.downloads), (0, 0));
}

#[test]
fn a_cpu_write_reaches_the_gpu() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let Some(mut app) = app_with_one_atom() else {
        return;
    };
    app.add_update_system(writer, Phase::Cpu);
    app.organize_systems();
    app.run();
    app.run();
    // The write must have been uploaded: after two halvings and two windows of
    // free fall, the speed is well below what uninterrupted free fall would give.
    let atoms_v = app.get_resource_ref::<Atom>().unwrap().vel[0][2] as f32;
    let window = WINDOW as f32 * DT;
    let expected = ((G * window) * 0.5 + G * window) * 0.5;
    assert!((atoms_v - expected).abs() < 1e-4, "{atoms_v} vs {expected}");
}
