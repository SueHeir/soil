//! Schedule integration: a drop-in GPU replacement for the CPU velocity Verlet
//! systems.
//!
//! [`VelocityVerletGpuPlugin`] registers GPU integration systems at the
//! `InitialIntegration` and `FinalIntegration` schedule phases, mirroring
//! [`soil_verlet::VelocityVerletPlugin`]. If no GPU adapter is available it
//! transparently falls back to the CPU systems.
//!
//! ## Host is the source of truth (for now)
//!
//! The device buffers are **resident** (allocated once, reused across steps and
//! only re-created when the local atom count changes), but each integration
//! phase still uploads inputs and downloads results. This is deliberate: in the
//! full schedule, `pbc`, `exchange`, `deform` and `pin` all mutate host
//! positions *between* integration phases, so the host array is authoritative
//! and naive residency would silently diverge. On Apple Silicon's unified memory
//! these transfers are cheap.
//!
//! The per-phase sync is the cost that a future milestone removes — once force,
//! neighbor and PBC also run on the GPU, the upload/download collapses and the
//! buffers become truly resident. The plumbing here is what that builds on.

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use soil_core::{Atom, ParticleSimScheduleSet};

use crate::{GpuContext, VerletGpu};

/// Resource holding the GPU device and the resident velocity-Verlet integrator.
///
/// `verlet` is created lazily on the first integration step (once atoms exist)
/// and re-created if the local atom count changes.
pub struct GpuVerlet {
    ctx: Option<GpuContext>,
    verlet: Option<VerletGpu>,
}

impl GpuVerlet {
    fn new(ctx: GpuContext) -> Self {
        Self { ctx: Some(ctx), verlet: None }
    }

    /// Ensure the resident integrator is sized for the current local atom count,
    /// (re)creating its buffers if needed. Returns false if there is no usable
    /// GPU integrator (no adapter, or zero atoms).
    fn ensure_sized(&mut self, atoms: &Atom) -> bool {
        let n = atoms.nlocal as usize;
        if n == 0 {
            return false;
        }
        let needs_build = match &self.verlet {
            Some(v) => v.len() != n,
            None => true,
        };
        if needs_build {
            let Some(ctx) = self.ctx.clone() else { return false };
            self.verlet = Some(VerletGpu::new(ctx, atoms));
        }
        self.verlet.is_some()
    }
}

/// First half of velocity Verlet on the GPU: upload current state, dispatch the
/// kick+drift kernel, download updated positions and velocities back to the host
/// so downstream CPU systems (force, comm, output) see them.
fn gpu_initial_integration(mut atoms: ResMut<Atom>, mut state: ResMut<GpuVerlet>) {
    if !state.ensure_sized(&atoms) {
        return;
    }
    let gpu = state.verlet.as_mut().unwrap();
    gpu.upload(&atoms);
    gpu.step_initial();
    gpu.download_into(&mut atoms);
}

/// Second half of velocity Verlet on the GPU: upload the freshly computed forces
/// (and current state), dispatch the completing kick, download velocities.
fn gpu_final_integration(mut atoms: ResMut<Atom>, mut state: ResMut<GpuVerlet>) {
    if !state.ensure_sized(&atoms) {
        return;
    }
    let gpu = state.verlet.as_mut().unwrap();
    gpu.upload(&atoms);
    gpu.step_final();
    gpu.download_vel_into(&mut atoms);
}

/// GPU velocity Verlet integration plugin — a drop-in replacement for
/// [`soil_verlet::VelocityVerletPlugin`].
///
/// Registers GPU integration at `InitialIntegration` / `FinalIntegration` when a
/// GPU adapter is available; otherwise falls back to the CPU systems so the
/// plugin is safe to add on any machine.
pub struct VelocityVerletGpuPlugin;

impl Plugin for VelocityVerletGpuPlugin {
    fn build(&self, app: &mut App) {
        match GpuContext::new() {
            Some(ctx) => {
                println!("VelocityVerletGpu: integrating on GPU adapter: {}", ctx.adapter_info);
                app.add_resource(GpuVerlet::new(ctx));
                app.add_update_system(gpu_initial_integration, ParticleSimScheduleSet::InitialIntegration);
                app.add_update_system(gpu_final_integration, ParticleSimScheduleSet::FinalIntegration);
            }
            None => {
                eprintln!(
                    "VelocityVerletGpu: no GPU adapter found — falling back to CPU velocity Verlet"
                );
                app.add_update_system(
                    soil_verlet::initial_integration,
                    ParticleSimScheduleSet::InitialIntegration,
                );
                app.add_update_system(
                    soil_verlet::final_integration,
                    ParticleSimScheduleSet::FinalIntegration,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constant gravity force written to the host each step. Reads host mass,
    /// writes host force — exercising the upload/download plumbing in the real
    /// schedule (initial -> force -> final).
    fn gravity(mut atoms: ResMut<Atom>) {
        let n = atoms.nlocal as usize;
        for i in 0..n {
            let m = atoms.mass[i] as f64;
            atoms.force[i] = [0.0 as _, (m * -9.81) as _, 0.0 as _];
        }
    }

    fn make_atoms() -> Atom {
        let mut a = Atom::new();
        a.dt = 1e-3;
        a.push_test_atom(0, [0.5, 1.0, -0.2], 0.1, 1.0);
        a.push_test_atom(1, [-0.4, 0.6, 0.3], 0.1, 2.5);
        a.vel[0] = [0.1, 0.0, -0.2];
        a.vel[1] = [0.0, 0.3, 0.1];
        a.nlocal = 2;
        a.natoms = 2;
        a
    }

    /// The GPU plugin, run through the real App schedule, must match the CPU
    /// velocity Verlet path within f32 tolerance. (If no GPU is present the
    /// plugin falls back to CPU, in which case this compares CPU to CPU — still
    /// a valid invariant.)
    #[test]
    fn gpu_plugin_matches_cpu_schedule() {
        let steps = 500;

        let mut gpu_app = App::new();
        gpu_app.add_resource(make_atoms());
        gpu_app.add_plugins(VelocityVerletGpuPlugin);
        gpu_app.add_update_system(gravity, ParticleSimScheduleSet::Force);
        gpu_app.organize_systems();

        let mut cpu_app = App::new();
        cpu_app.add_resource(make_atoms());
        cpu_app.add_update_system(
            soil_verlet::initial_integration,
            ParticleSimScheduleSet::InitialIntegration,
        );
        cpu_app.add_update_system(
            soil_verlet::final_integration,
            ParticleSimScheduleSet::FinalIntegration,
        );
        cpu_app.add_update_system(gravity, ParticleSimScheduleSet::Force);
        cpu_app.organize_systems();

        for _ in 0..steps {
            gpu_app.run();
            cpu_app.run();
        }

        let ga = gpu_app.get_resource_ref::<Atom>().unwrap();
        let ca = cpu_app.get_resource_ref::<Atom>().unwrap();
        for i in 0..2 {
            for d in 0..3 {
                let g = ga.pos[i][d] as f64;
                let c = ca.pos[i][d] as f64;
                assert!(
                    (g - c).abs() < 1e-3,
                    "atom {i} pos[{d}]: gpu={g} cpu={c} (diff {})",
                    (g - c).abs()
                );
            }
        }
    }
}
