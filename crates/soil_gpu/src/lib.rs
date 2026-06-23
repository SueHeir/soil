//! GPU compute kernels for SOIL via `wgpu` (Metal on macOS, Vulkan/DX12 elsewhere).
//!
//! Milestone 2 of the GPU port: an end-to-end, single-precision (f32) velocity
//! Verlet integrator running on the GPU, proving the host <-> device path. Apple
//! GPUs have no f64, so the device path is f32 regardless of the host build's
//! [`soil_core::Real`]/[`soil_core::Accum`] precision; values are cast to f32 on
//! upload.
//!
//! This is the standalone, validated kernel. Wiring it into the per-timestep
//! schedule with resident buffers (so it actually accelerates a run) is the next
//! milestone — see the crate README / plan.
//!
//! # Example
//! ```no_run
//! # use soil_core::Atom;
//! # use soil_gpu::{GpuContext, VerletGpu};
//! let ctx = GpuContext::new().expect("no GPU adapter");
//! let mut atoms = Atom::new();
//! atoms.dt = 1e-3;
//! atoms.push_test_atom(0, [0.0; 3], 0.5, 1.0);
//! atoms.nlocal = 1;
//! let mut gpu = VerletGpu::new(ctx, &atoms);
//! gpu.run_constant_force_steps(1000); // f=const -> exact VV trajectory
//! gpu.download_into(&mut atoms);
//! ```

use bytemuck::{Pod, Zeroable};
use soil_core::Atom;

pub mod plugin;
pub use plugin::{GpuVerlet, VelocityVerletGpuPlugin};

pub mod cell_list;
pub use cell_list::{CellList, Grid};

pub mod coherence;
pub use coherence::DualBuffer;

pub mod neighbor_slots;
pub use neighbor_slots::{NeighborSlots, SLOTS_WGSL};

pub mod boundary;
pub use boundary::{Boundary, Plane, BOUNDARY_WGSL};

pub mod gpu_state;
pub use gpu_state::{GpuForce, GpuState};

/// A wgpu device + queue. Created once and shared by GPU kernels.
///
/// `Clone` is cheap — wgpu `Device`/`Queue` are reference-counted handles — which
/// lets the plugin re-create sized buffers when the local atom count changes.
#[derive(Clone)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Human-readable adapter name (backend + GPU), for logging.
    pub adapter_info: String,
}

impl GpuContext {
    /// Acquire a GPU device. Returns `None` when no adapter is available
    /// (e.g. headless CI without a software fallback) so callers can skip.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        Self::from_adapter(adapter)
    }

    /// One-GPU-per-rank binding for MPI scale-out (step 3): rank `local_rank` binds
    /// adapter `local_rank % num_adapters`. On a single-GPU machine every rank gets
    /// the one device (a no-op, so this is safe everywhere); on a multi-GPU node it
    /// spreads ranks across devices. Falls back to the default adapter if adapter
    /// enumeration yields none.
    pub fn new_for_rank(local_rank: usize) -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let adapter = if adapters.is_empty() {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok()?
        } else {
            let idx = local_rank % adapters.len();
            adapters.into_iter().nth(idx)?
        };
        Self::from_adapter(adapter)
    }

    fn from_adapter(adapter: wgpu::Adapter) -> Option<Self> {
        let info = adapter.get_info();
        let adapter_info = format!("{} ({:?})", info.name, info.backend);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("soil_gpu device"),
            required_features: wgpu::Features::empty(),
            // Request the adapter's full limits — the resident substrate binds many
            // storage buffers (cell list + state + per-neighbor slots), above the
            // conservative default of 8.
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        Some(Self { device, queue, adapter_info })
    }
}

/// Uniform params block. Padded to 16 bytes for uniform-buffer alignment.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    dt: f32,
    n: u32,
    _pad: [u32; 2],
}

/// GPU-resident velocity Verlet integrator for a fixed atom count.
///
/// Buffers stay resident on the device across [`step_initial`](Self::step_initial)
/// / [`step_final`](Self::step_final) calls; upload once, dispatch many times,
/// download when host state is needed.
pub struct VerletGpu {
    ctx: GpuContext,
    n: usize,
    pos: wgpu::Buffer,
    vel: wgpu::Buffer,
    force: wgpu::Buffer,
    inv_mass: wgpu::Buffer,
    params: wgpu::Buffer,
    /// Mappable readback buffer sized for a 3*N f32 array (pos or vel).
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    initial_pipeline: wgpu::ComputePipeline,
    final_pipeline: wgpu::ComputePipeline,
}

const WORKGROUP_SIZE: u32 = 64;

impl VerletGpu {
    /// Create buffers/pipelines for `atoms` (local atoms only) and upload state.
    pub fn new(ctx: GpuContext, atoms: &Atom) -> Self {
        let n = atoms.nlocal as usize;
        let device = &ctx.device;

        let vec3_bytes = (n.max(1) * 3 * std::mem::size_of::<f32>()) as u64;
        let scalar_bytes = (n.max(1) * std::mem::size_of::<f32>()) as u64;

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let pos = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pos"), size: vec3_bytes, usage: storage, mapped_at_creation: false,
        });
        let vel = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vel"), size: vec3_bytes, usage: storage, mapped_at_creation: false,
        });
        let force = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("force"), size: vec3_bytes, usage: storage, mapped_at_creation: false,
        });
        let inv_mass = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inv_mass"), size: scalar_bytes, usage: storage, mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: vec3_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("verlet"),
            source: wgpu::ShaderSource::Wgsl(include_str!("verlet.wgsl").into()),
        });

        // Bind group layout: pos/vel read-write storage, force/inv_mass read-only
        // storage, params uniform.
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("verlet bgl"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, true),
                storage_entry(3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("verlet bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pos.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: vel.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: force.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: inv_mass.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params.as_entire_binding() },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("verlet pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        let initial_pipeline = make_pipeline("initial");
        let final_pipeline = make_pipeline("final_kick");

        let mut gpu = Self {
            ctx, n, pos, vel, force, inv_mass, params, staging,
            bind_group, initial_pipeline, final_pipeline,
        };
        gpu.upload(atoms);
        gpu
    }

    /// (Re)upload all per-atom state + params from `atoms`. Values are cast to
    /// f32 regardless of the host build's precision.
    pub fn upload(&mut self, atoms: &Atom) {
        assert_eq!(atoms.nlocal as usize, self.n, "atom count changed; recreate VerletGpu");
        let q = &self.ctx.queue;
        q.write_buffer(&self.pos, 0, bytemuck::cast_slice(&flat3(&atoms.pos[..self.n])));
        q.write_buffer(&self.vel, 0, bytemuck::cast_slice(&flat3(&atoms.vel[..self.n])));
        q.write_buffer(&self.force, 0, bytemuck::cast_slice(&flat3(&atoms.force[..self.n])));
        let inv_mass: Vec<f32> = atoms.inv_mass[..self.n].iter().map(|&m| m as f32).collect();
        q.write_buffer(&self.inv_mass, 0, bytemuck::cast_slice(&inv_mass));
        let params = Params { dt: atoms.dt as f32, n: self.n as u32, _pad: [0; 2] };
        q.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
    }

    /// Re-upload only the force buffer (for use after an external force recompute).
    pub fn upload_force(&mut self, atoms: &Atom) {
        self.ctx.queue.write_buffer(&self.force, 0, bytemuck::cast_slice(&flat3(&atoms.force[..self.n])));
    }

    fn dispatch(&self, pipeline: &wgpu::ComputePipeline, label: &str) {
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(label),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = (self.n as u32).div_ceil(WORKGROUP_SIZE).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// Dispatch the initial integration (kick + drift).
    pub fn step_initial(&self) {
        self.dispatch(&self.initial_pipeline, "initial");
    }

    /// Dispatch the final integration (completing kick).
    pub fn step_final(&self) {
        self.dispatch(&self.final_pipeline, "final");
    }

    /// Run `n_steps` full velocity Verlet steps holding the uploaded force
    /// constant (exact for constant force / free particle — used for validation).
    pub fn run_constant_force_steps(&self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step_initial();
            self.step_final();
        }
    }

    /// Read back positions from the device as `[f32; 3]` per atom.
    pub fn download_pos(&self) -> Vec<[f32; 3]> {
        self.read_vec3(&self.pos)
    }

    /// Read back velocities from the device as `[f32; 3]` per atom.
    pub fn download_vel(&self) -> Vec<[f32; 3]> {
        self.read_vec3(&self.vel)
    }

    /// Number of atoms this integrator's buffers are sized for.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns true if sized for zero atoms.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Download only velocities back into `atoms` (the final kick leaves positions
    /// unchanged, so the position copy can be skipped).
    pub fn download_vel_into(&self, atoms: &mut Atom) {
        for (dst, src) in atoms.vel[..self.n].iter_mut().zip(self.download_vel()) {
            *dst = [src[0] as _, src[1] as _, src[2] as _];
        }
    }

    /// Download positions and velocities back into `atoms` (cast to host precision).
    pub fn download_into(&self, atoms: &mut Atom) {
        for (dst, src) in atoms.pos[..self.n].iter_mut().zip(self.download_pos()) {
            *dst = [src[0] as _, src[1] as _, src[2] as _];
        }
        for (dst, src) in atoms.vel[..self.n].iter_mut().zip(self.download_vel()) {
            *dst = [src[0] as _, src[1] as _, src[2] as _];
        }
    }

    fn read_vec3(&self, buffer: &wgpu::Buffer) -> Vec<[f32; 3]> {
        let bytes = (self.n * 3 * std::mem::size_of::<f32>()) as u64;
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
        enc.copy_buffer_to_buffer(buffer, 0, &self.staging, 0, bytes);
        self.ctx.queue.submit(Some(enc.finish()));

        let slice = self.staging.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let out = floats.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        drop(data);
        self.staging.unmap();
        out
    }

    /// The adapter description (backend + GPU name).
    pub fn adapter_info(&self) -> &str {
        &self.ctx.adapter_info
    }
}

/// Flatten `&[[T; 3]]` (T castable to f32) into a tightly packed `Vec<f32>`.
fn flat3<T: Copy + Into<f64>>(v: &[[T; 3]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(v.len() * 3);
    for a in v {
        out.push(a[0].into() as f32);
        out.push(a[1].into() as f32);
        out.push(a[2].into() as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Velocity Verlet is *exact* for constant force, so the GPU f32 kernel must
    // reproduce the analytic trajectory x = x0 + v0 t + ½ a t², v = v0 + a t to
    // within f32 round-off. These tests validate the full host->device->host path
    // on whatever adapter wgpu finds (Metal on macOS).

    /// Free particle (f = 0): velocity is preserved, position drifts linearly.
    #[test]
    fn gpu_free_particle_constant_velocity() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter available; skipping GPU test");
            return;
        };
        let mut atoms = Atom::new();
        atoms.dt = 1e-3;
        atoms.push_test_atom(0, [0.0, 0.0, 0.0], 0.5, 1.0);
        atoms.vel[0] = [1.5, -2.3, 0.7];
        atoms.nlocal = 1;
        atoms.natoms = 1;

        let n = 1000;
        let gpu = VerletGpu::new(ctx, &atoms);
        gpu.run_constant_force_steps(n);

        let pos = gpu.download_pos();
        let vel = gpu.download_vel();
        let t = n as f64 * 1e-3; // = 1.0
        let expect_pos = [1.5 * t, -2.3 * t, 0.7 * t];
        let expect_vel = [1.5, -2.3, 0.7];
        for k in 0..3 {
            assert!((pos[0][k] as f64 - expect_pos[k]).abs() < 1e-4,
                "pos[{k}]: gpu={} expected={}", pos[0][k], expect_pos[k]);
            assert!((vel[0][k] as f64 - expect_vel[k]).abs() < 1e-5,
                "vel[{k}]: gpu={} expected={}", vel[0][k], expect_vel[k]);
        }
    }

    /// Many atoms (> one workgroup of 64) each under a distinct constant force —
    /// exercises parallel dispatch, per-atom indexing, and the bounds guard.
    #[test]
    fn gpu_many_atoms_constant_force_matches_analytic() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter available; skipping GPU test");
            return;
        };
        let natoms = 200usize; // spans 4 workgroups, last one partial
        let dt = 1e-4_f64;
        let mut atoms = Atom::new();
        atoms.dt = dt;
        // Keep coordinates near the origin: f32 loses precision adding the tiny
        // per-step drift to a large absolute coordinate (the large-coordinate
        // problem the residency milestone must handle with a careful frame). This
        // test isolates parallel dispatch / per-atom indexing, not that effect.
        for i in 0..natoms {
            let mass = 1.0 + i as f64 * 0.01;
            atoms.push_test_atom(i as u32, [i as f64 * 0.01, 0.0, 0.0], 0.5, mass);
            atoms.vel[i][0] = (0.1 * i as f64) as _;
            // distinct acceleration a_i -> force = m * a_i
            let a_i = -0.5 - 0.001 * i as f64;
            atoms.force[i][0] = (mass * a_i) as _;
        }
        atoms.nlocal = natoms as u32;
        atoms.natoms = natoms as u64;

        let n_steps = 500;
        let gpu = VerletGpu::new(ctx, &atoms);
        eprintln!("GPU adapter: {}", gpu.adapter_info());
        gpu.run_constant_force_steps(n_steps);
        let pos = gpu.download_pos();
        let vel = gpu.download_vel();

        let t = n_steps as f64 * dt;
        for i in 0..natoms {
            let v0 = 0.1 * i as f64;
            let a_i = -0.5 - 0.001 * i as f64;
            let exp_x = i as f64 * 0.01 + v0 * t + 0.5 * a_i * t * t;
            let exp_v = v0 + a_i * t;
            // f32 round-off accumulates over ~1000 half-kick additions; bound is
            // the f32-tolerance band, not bit-exactness.
            assert!((pos[i][0] as f64 - exp_x).abs() < 1e-3,
                "atom {i} x: gpu={} expected={}", pos[i][0], exp_x);
            assert!((vel[i][0] as f64 - exp_v).abs() < 1e-3,
                "atom {i} v: gpu={} expected={}", vel[i][0], exp_v);
        }
    }

    /// GPU f32 result should track the CPU `soil_verlet`-style f64 reference
    /// within f32 tolerance (the GPU-vs-CPU equivalence the README calls for).
    #[test]
    fn gpu_matches_cpu_reference_within_f32_tol() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter available; skipping GPU test");
            return;
        };
        let dt = 1e-3_f64;
        let mass = 3.0_f64;
        let a = 2.0_f64;
        let force = mass * a;
        let (x0, v0) = (1.0_f64, -0.5_f64);

        let mut atoms = Atom::new();
        atoms.dt = dt;
        atoms.push_test_atom(0, [x0, 0.0, 0.0], 0.5, mass);
        atoms.vel[0][0] = v0 as _;
        atoms.force[0][0] = force as _;
        atoms.nlocal = 1;
        atoms.natoms = 1;

        let n = 2000;
        let gpu = VerletGpu::new(ctx, &atoms);
        gpu.run_constant_force_steps(n);
        let gpu_x = gpu.download_pos()[0][0] as f64;

        // CPU f64 velocity Verlet reference (constant force).
        let (mut x, mut v) = (x0, v0);
        for _ in 0..n {
            v += 0.5 * dt * force / mass;
            x += v * dt;
            v += 0.5 * dt * force / mass;
        }
        assert!((gpu_x - x).abs() < 1e-3, "GPU f32 {gpu_x} vs CPU f64 {x}");
    }
}
