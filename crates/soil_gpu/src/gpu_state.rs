//! Resident GPU simulation state (substrate) — the core of the resident loop.
//!
//! Owns the resident per-atom core buffers (pos/vel/force/inv_mass) and drives a
//! velocity-Verlet step with a generic body force (gravity), entirely on-device.
//! The constitutive force (DEM contact, MD pair, peridynamics bond, walls) is a
//! **Force hook** ([`GpuForce`]) the consumer registers; it runs between the two
//! integration half-steps and accumulates into `force` (i-centric, no atomics).
//! This iteration wires the integration core + gravity + the Force-hook dispatch;
//! the cell-list build (so hooks can find neighbors) is added next.

use bytemuck::{Pod, Zeroable};

use crate::cell_list::{CellList, Grid};
use crate::GpuContext;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    dt: f32,
    gx: f32,
    gy: f32,
    gz: f32,
    // Periodic box: length on each axis (0 = non-periodic axis), low-corner
    // origin, and the Lees–Edwards xy shear (x tilt per y-image + the Δv velocity
    // offset). Drives the on-device PBC + LE remap in `integrate_initial`.
    lx: f32,
    ly: f32,
    lz: f32,
    ox: f32,
    oy: f32,
    oz: f32,
    tilt_xy: f32,
    dv_xy: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}

/// A constitutive force a consumer registers into the resident loop. Its
/// `record` runs between the integration half-steps (after gravity is seeded)
/// and accumulates into the resident `force` buffer (i-centric, no atomics). The
/// implementor owns its pipeline + bind groups, built over `GpuState`'s exposed
/// buffers (pos/vel/force, and later cell-list outputs) plus its own.
pub trait GpuForce {
    fn record(&self, pass: &mut wgpu::ComputePass);
    /// History-neutral force evaluation for the entry prime of `run_steps`:
    /// compute F(x₀) from the *current* contact history WITHOUT advancing it (and
    /// without storing an advanced history), so re-priming at window boundaries
    /// (the residency / MPI-halo model) doesn't double-advance the Mindlin spring.
    /// Default: identical to `record` — correct for stateless hooks.
    fn record_prime(&self, pass: &mut wgpu::ComputePass) {
        self.record(pass);
    }
}

/// An auxiliary integrated degree of freedom: a velocity-like `state` driven by
/// `rate` with per-atom inverse coefficient `inv_coeff`, integrated with the same
/// two half-kicks as velocity but with no position drift (see resident_aux.wgsl).
/// Method-agnostic: granular angular velocity (state=omega, rate=torque,
/// inv_coeff=1/inertia), thermostat DOF, etc. A Force hook owns `rate` (writes it
/// each step); `state` is exposed so the same hook can read it.
struct AuxDof {
    state: wgpu::Buffer,
    rate: wgpu::Buffer,
    inv_coeff: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    staging: wgpu::Buffer,
}

/// Resident core simulation state on the GPU.
#[allow(dead_code)]
pub struct GpuState {
    ctx: GpuContext,
    n: usize,
    /// Owns the resident `pos` buffer (integrated in place) and rebins it each
    /// step so Force hooks can find neighbors via its cell_start/sorted_atoms.
    cell_list: CellList,
    vel: wgpu::Buffer,
    force: wgpu::Buffer,
    inv_mass: wgpu::Buffer,
    params: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    p_seed: wgpu::ComputePipeline,
    p_init: wgpu::ComputePipeline,
    p_final: wgpu::ComputePipeline,
    p_aux: wgpu::ComputePipeline,
    aux_bgl: wgpu::BindGroupLayout,
    aux_dofs: Vec<AuxDof>,
    hooks: Vec<Box<dyn GpuForce>>,
    gravity: [f32; 3],
    dt: f32,
    /// Periodic box state (0 lengths = non-periodic). See [`Self::set_box`].
    box_len: [f32; 3],
    box_origin: [f32; 3],
    tilt_xy: f32,
    dv_xy: f32,
}

const WG: u32 = 64;

impl GpuState {
    pub fn new(ctx: GpuContext, n: usize, total_cells: usize) -> Self {
        // The cell list owns the resident `pos` buffer; integration writes it,
        // the cell-list kernels rebin it, and Force hooks read it — one buffer.
        let cell_list = CellList::new(ctx.clone(), n, total_cells);
        let device = &ctx.device;
        let nz = n.max(1);
        let vec3 = (nz * 3 * 4) as u64;
        let scal = (nz * 4) as u64;
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str, size, usage| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size, usage, mapped_at_creation: false,
        });
        let vel = mk("gs_vel", vec3, rw);
        let force = mk("gs_force", vec3, rw);
        let inv_mass = mk("gs_inv_mass", scal, ro);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gs_params"), size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gs_staging"), size: vec3,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident"),
            source: wgpu::ShaderSource::Wgsl(include_str!("resident.wgsl").into()),
        });
        let st = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gs bgl"),
            entries: &[
                st(0, false), st(1, false), st(2, false), st(3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gs bg"), layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cell_list.pos_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: vel.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: force.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: inv_mass.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params.as_entire_binding() },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gs pl"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
        });
        let mk_pipe = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry), layout: Some(&layout), module: &shader, entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });
        // Auxiliary-DOF half-kick pipeline (separate module; its own group-0
        // layout: aux_state(rw), aux_rate(r), aux_inv_coeff(r), params(uniform)).
        let aux_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident_aux"),
            source: wgpu::ShaderSource::Wgsl(include_str!("resident_aux.wgsl").into()),
        });
        let aux_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("aux bgl"),
            entries: &[
                st(0, false), st(1, true), st(2, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let aux_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("aux pl"), bind_group_layouts: &[Some(&aux_bgl)], immediate_size: 0,
        });
        let p_aux = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("aux_kick"), layout: Some(&aux_layout), module: &aux_shader, entry_point: Some("aux_kick"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        Self {
            p_seed: mk_pipe("seed_gravity"),
            p_init: mk_pipe("integrate_initial"),
            p_final: mk_pipe("integrate_final"),
            p_aux, aux_bgl, aux_dofs: Vec::new(),
            ctx, n, cell_list, vel, force, inv_mass, params, staging, bind_group,
            hooks: Vec::new(), gravity: [0.0; 3], dt: 1.0e-4,
            box_len: [0.0; 3], box_origin: [0.0; 3], tilt_xy: 0.0, dv_xy: 0.0,
        }
    }

    /// Register a constitutive force hook. It runs each step after gravity is
    /// seeded and before the final integration half, accumulating into `force`.
    /// Hooks run in registration order. Build the hook over the buffers exposed
    /// by [`pos_buffer`](Self::pos_buffer) / [`vel_buffer`](Self::vel_buffer) /
    /// [`force_buffer`](Self::force_buffer) so it binds the SAME resident buffers.
    pub fn add_force_hook(&mut self, hook: Box<dyn GpuForce>) {
        self.hooks.push(hook);
    }

    /// Resident buffers a Force hook binds into its own bind group. They are the
    /// flat `array<f32>` (3*i+c) per-atom buffers; `inv_mass` is `array<f32>`.
    /// `cell_start`/`sorted_atoms`/`atom_cell` are the cell-list outputs (rebuilt
    /// each step) a hook walks to find each atom's neighbors.
    pub fn pos_buffer(&self) -> &wgpu::Buffer { self.cell_list.pos_buffer() }
    pub fn vel_buffer(&self) -> &wgpu::Buffer { &self.vel }
    pub fn force_buffer(&self) -> &wgpu::Buffer { &self.force }
    pub fn inv_mass_buffer(&self) -> &wgpu::Buffer { &self.inv_mass }
    pub fn cell_start_buffer(&self) -> &wgpu::Buffer { self.cell_list.cell_start_buffer() }
    pub fn sorted_atoms_buffer(&self) -> &wgpu::Buffer { self.cell_list.sorted_atoms_buffer() }
    pub fn atom_cell_buffer(&self) -> &wgpu::Buffer { self.cell_list.atom_cell_buffer() }
    pub fn n(&self) -> usize { self.n }
    pub fn context(&self) -> &GpuContext { &self.ctx }

    /// Register an auxiliary integrated DOF (e.g. granular rotation). Returns its
    /// index. The consumer sets `inv_coeff` (e.g. 1/inertia) and initial `state`,
    /// and binds `aux_state_buffer`/`aux_rate_buffer` into its Force hook (the
    /// hook overwrites `rate`, the resident loop integrates `state`). Call before
    /// `run_steps`.
    pub fn add_aux_dof(&mut self) -> usize {
        let device = &self.ctx.device;
        let nz = self.n.max(1);
        let vec3 = (nz * 3 * 4) as u64;
        let scal = (nz * 4) as u64;
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str, size, usage| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size, usage, mapped_at_creation: false,
        });
        let state = mk("aux_state", vec3, rw);
        let rate = mk("aux_rate", vec3, rw);
        let inv_coeff = mk("aux_inv_coeff", scal, ro);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aux_staging"), size: vec3,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aux bg"), layout: &self.aux_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rate.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: inv_coeff.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.params.as_entire_binding() },
            ],
        });
        self.aux_dofs.push(AuxDof { state, rate, inv_coeff, bind_group, staging });
        self.aux_dofs.len() - 1
    }

    pub fn aux_state_buffer(&self, i: usize) -> &wgpu::Buffer { &self.aux_dofs[i].state }
    pub fn aux_rate_buffer(&self, i: usize) -> &wgpu::Buffer { &self.aux_dofs[i].rate }
    pub fn aux_inv_coeff_buffer(&self, i: usize) -> &wgpu::Buffer { &self.aux_dofs[i].inv_coeff }

    pub fn set_aux_state(&self, i: usize, state: &[[f32; 3]]) {
        self.ctx.queue.write_buffer(&self.aux_dofs[i].state, 0, bytemuck::cast_slice(&flat3(state)));
    }

    // ── Partial slice writes (GPU-resident MPI halos) ────────────────────────
    // Overwrite a contiguous atom range starting at `start` without touching the
    // rest of the buffer. The resident-MPI path uses these to refresh the ghost
    // slice (`start = nlocal`) each step while the local bulk stays on-device.
    const STRIDE3: u64 = 3 * 4; // [f32; 3] = 12 bytes per atom
    pub fn write_pos_slice(&self, start: usize, pos: &[[f32; 3]]) {
        self.ctx.queue.write_buffer(self.cell_list.pos_buffer(), start as u64 * Self::STRIDE3, bytemuck::cast_slice(&flat3(pos)));
    }
    pub fn write_vel_slice(&self, start: usize, vel: &[[f32; 3]]) {
        self.ctx.queue.write_buffer(&self.vel, start as u64 * Self::STRIDE3, bytemuck::cast_slice(&flat3(vel)));
    }
    pub fn write_aux_slice(&self, i: usize, start: usize, state: &[[f32; 3]]) {
        self.ctx.queue.write_buffer(&self.aux_dofs[i].state, start as u64 * Self::STRIDE3, bytemuck::cast_slice(&flat3(state)));
    }
    pub fn set_aux_inv_coeff(&self, i: usize, inv_coeff: &[f32]) {
        self.ctx.queue.write_buffer(&self.aux_dofs[i].inv_coeff, 0, bytemuck::cast_slice(inv_coeff));
    }
    pub fn download_aux_state(&self, i: usize) -> Vec<[f32; 3]> {
        self.download_into(&self.aux_dofs[i].state, &self.aux_dofs[i].staging)
    }

    /// Upload the initial state. `grid` is the (fixed) cell-list grid, typically
    /// `Grid::from_positions(pos, cutoff)`. Positions live in the owned cell list.
    pub fn set_state(&self, pos: &[[f32; 3]], vel: &[[f32; 3]], inv_mass: &[f32], grid: Grid) {
        assert_eq!(pos.len(), self.n);
        self.cell_list.upload_positions(pos, grid);
        let q = &self.ctx.queue;
        q.write_buffer(&self.vel, 0, bytemuck::cast_slice(&flat3(vel)));
        q.write_buffer(&self.inv_mass, 0, bytemuck::cast_slice(inv_mass));
    }

    /// Update the cell-list grid (origin / bin size) WITHOUT re-uploading the
    /// resident positions. Used by the GPU-resident MPI halo path, which keeps
    /// the local bulk on-device and only writes the ghost slice each step (via
    /// `pos_buffer()`/`vel_buffer()` + `queue.write_buffer` at the ghost offset),
    /// then re-bins under the refreshed grid in the next `run_steps`.
    pub fn set_grid(&self, grid: Grid) {
        self.cell_list.set_grid(grid);
    }

    pub fn set_params(&mut self, dt: f32, gravity: [f32; 3]) {
        self.dt = dt;
        self.gravity = gravity;
        self.write_params();
    }

    /// Enable on-device periodic boundaries + Lees–Edwards shear for the resident
    /// loop. `lengths` is the box size per axis (0 = that axis is non-periodic);
    /// `origin` is the low corner; `tilt_xy` is the LE x-shift per y-image and
    /// `dv_xy` the corresponding velocity offset Δv = γ̇·Lᵧ. Atoms are wrapped into
    /// the box each step (y-crossings get x,vx remapped by the tilt/Δv). For a
    /// continuously-sheared run, advance `tilt_xy` (box-flip-wrapped) and re-call.
    pub fn set_box(&mut self, lengths: [f32; 3], origin: [f32; 3], tilt_xy: f32, dv_xy: f32) {
        self.box_len = lengths;
        self.box_origin = origin;
        self.tilt_xy = tilt_xy;
        self.dv_xy = dv_xy;
        self.write_params();
    }

    fn write_params(&self) {
        let p = Params {
            n: self.n as u32, dt: self.dt,
            gx: self.gravity[0], gy: self.gravity[1], gz: self.gravity[2],
            lx: self.box_len[0], ly: self.box_len[1], lz: self.box_len[2],
            ox: self.box_origin[0], oy: self.box_origin[1], oz: self.box_origin[2],
            tilt_xy: self.tilt_xy, dv_xy: self.dv_xy,
            _p0: 0.0, _p1: 0.0, _p2: 0.0,
        };
        self.ctx.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
    }

    /// Run `steps` resident velocity-Verlet steps (one submit, no per-step
    /// transfer). VV needs the force at the entry positions, F(x₀), before the
    /// first half-kick, so the force is primed once (seed + hooks) before the
    /// loop. Then per step:
    ///   1. `integrate_initial` — half kick with F(x), then drift x
    ///   2. `seed_gravity` — reset force to the body force m*g at the new x
    ///   3. registered Force hooks — accumulate constitutive force into `force`
    ///   4. `integrate_final` — half kick with the new force
    /// The force left by step 2–3 serves both step 4's kick and the *next*
    /// iteration's step-1 kick (one force eval per step). Each resident kernel
    /// re-binds group 0 since hooks rebind it for their own dispatch. (The
    /// cell-list build will slot just before step 2.)
    pub fn run_steps(&self, steps: usize) {
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gs steps") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gs steps"), timestamp_writes: None });
            let g = (self.n as u32).div_ceil(WG).max(1);
            // Prime F(x₀): build the cell list at the entry positions, then the
            // force (seed + hooks), before the first kick.
            self.cell_list.record(&mut pass);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.p_seed);
            pass.dispatch_workgroups(g, 1, 1);
            // History-neutral prime: evaluate F(x₀) without advancing contact
            // history, so chopping a run into windows (residency / MPI halos)
            // doesn't double-advance the Mindlin spring at each boundary.
            for hook in &self.hooks {
                hook.record_prime(&mut pass);
            }
            self.record_step_loop(&mut pass, g, steps);
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// Continue a resident run for `steps` more steps WITHOUT re-priming the
    /// force at entry — it trusts the resident `force` buffer left valid by the
    /// previous window's last step. This is the correct way to stitch residency
    /// windows (the sync points for neighbor rebuild / MPI halo exchange / I-O):
    /// re-priming (as `run_steps` does at entry) would re-evaluate the
    /// velocity-dependent contact damping at the *full* end-of-step velocity
    /// instead of the *mid-step* (half-kick) velocity the integrator actually
    /// used, deterministically diverging the trajectory at every boundary. Call
    /// `run_steps` once to establish F(x₀), then `run_steps_continue` thereafter;
    /// stitched windows then reproduce one uninterrupted `run_steps` bit-for-bit.
    /// (Positions/velocities/history must be unchanged on the host between calls;
    /// if the host mutated them, prime again with `run_steps`.)
    pub fn run_steps_continue(&self, steps: usize) {
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gs steps cont") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gs steps cont"), timestamp_writes: None });
            let g = (self.n as u32).div_ceil(WG).max(1);
            self.record_step_loop(&mut pass, g, steps);
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    /// The K-step velocity-Verlet loop body shared by `run_steps` and
    /// `run_steps_continue`. Assumes the force buffer already holds F at the
    /// current positions (set by the caller's prime, or the previous window).
    fn record_step_loop(&self, pass: &mut wgpu::ComputePass, g: u32, steps: usize) {
        for _ in 0..steps {
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.p_init);
            pass.dispatch_workgroups(g, 1, 1); // kick+drift AND re-seed force=m*g
            // First aux half-kick (uses the previous step's rate).
            self.aux_kick(pass, g);
            // Rebin at the drifted positions so hooks see current neighbors.
            self.cell_list.record(pass);
            // Force hooks accumulate the constitutive force onto the m*g seed.
            for hook in &self.hooks {
                hook.record(pass);
            }
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.p_final);
            pass.dispatch_workgroups(g, 1, 1);
            // Second aux half-kick (uses the freshly computed rate).
            self.aux_kick(pass, g);
        }
    }

    /// Evaluate the force once at the current positions WITHOUT integrating:
    /// build the cell list, seed the body force, run the Force hooks. For
    /// one-shot force/torque inspection (testing / cross-validation). Afterward
    /// `force_buffer` holds body + hook force, and each aux DOF's `rate` holds
    /// whatever its hook wrote (e.g. torque). Set gravity to zero first to isolate
    /// the constitutive force.
    pub fn eval_force_once(&self) {
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gs eval") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gs eval"), timestamp_writes: None });
            let g = (self.n as u32).div_ceil(WG).max(1);
            self.cell_list.record(&mut pass);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.p_seed);
            pass.dispatch_workgroups(g, 1, 1);
            for hook in &self.hooks {
                hook.record(&mut pass);
            }
        }
        self.ctx.queue.submit(Some(enc.finish()));
    }

    pub fn download_aux_rate(&self, i: usize) -> Vec<[f32; 3]> {
        self.download_into(&self.aux_dofs[i].rate, &self.aux_dofs[i].staging)
    }

    /// Record an aux-DOF half-kick for every registered DOF (no-op if none).
    /// Uses the aux module's own group-0 layout, so callers re-bind the resident
    /// group 0 afterward (the loop does, before the next resident kernel).
    fn aux_kick(&self, pass: &mut wgpu::ComputePass, g: u32) {
        if self.aux_dofs.is_empty() { return; }
        pass.set_pipeline(&self.p_aux);
        for aux in &self.aux_dofs {
            pass.set_bind_group(0, &aux.bind_group, &[]);
            pass.dispatch_workgroups(g, 1, 1);
        }
    }

    /// Block until all submitted GPU work has completed (for timing/sync).
    pub fn wait(&self) {
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }

    pub fn download_pos(&self) -> Vec<[f32; 3]> { self.download(self.cell_list.pos_buffer()) }
    pub fn download_vel(&self) -> Vec<[f32; 3]> { self.download(&self.vel) }
    pub fn download_force(&self) -> Vec<[f32; 3]> { self.download(&self.force) }

    fn download(&self, buf: &wgpu::Buffer) -> Vec<[f32; 3]> {
        self.download_into(buf, &self.staging)
    }

    fn download_into(&self, buf: &wgpu::Buffer, staging: &wgpu::Buffer) -> Vec<[f32; 3]> {
        let bytes = (self.n * 3 * 4) as u64;
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gs dl") });
        enc.copy_buffer_to_buffer(buf, 0, staging, 0, bytes);
        self.ctx.queue.submit(Some(enc.finish()));
        let slice = staging.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range();
        let f: &[f32] = bytemuck::cast_slice(&data);
        let out = f.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        drop(data);
        staging.unmap();
        out
    }
}

fn flat3(v: &[[f32; 3]]) -> Vec<f32> {
    let mut o = Vec::with_capacity(v.len() * 3);
    for a in v { o.extend_from_slice(a); }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_state_freefall_matches_analytic() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let p0 = [[0.0f32, 0.0, 10.0]];
        let grid = Grid::from_positions(&p0, 1.0);
        let mut gs = GpuState::new(ctx, 1, grid.total_cells);
        let dt = 1.0e-3f32;
        let g = [0.0f32, 0.0, -9.81];
        gs.set_params(dt, g);
        gs.set_state(&p0, &[[1.0, 0.0, 2.0]], &[1.0], grid);

        let steps = 500;
        gs.run_steps(steps);
        let p = gs.download_pos()[0];
        let v = gs.download_vel()[0];
        let t = steps as f32 * dt; // 0.5 s

        // VV is exact for constant force: x = x0 + v0 t + 0.5 a t^2.
        let ex = [0.0 + 1.0 * t, 0.0, 10.0 + 2.0 * t + 0.5 * g[2] * t * t];
        let ev = [1.0, 0.0, 2.0 + g[2] * t];
        for c in 0..3 {
            assert!((p[c] - ex[c]).abs() < 1e-3, "pos[{c}]: {} vs {}", p[c], ex[c]);
            assert!((v[c] - ev[c]).abs() < 1e-3, "vel[{c}]: {} vs {}", v[c], ev[c]);
        }
        eprintln!("GpuState free-fall: pos={p:?} vel={v:?} (analytic match)");
    }

    #[test]
    fn resident_periodic_wrap() {
        let Some(ctx) = GpuContext::new() else { eprintln!("no GPU; skipping"); return; };
        let p0 = [[0.9f32, 0.5, 0.5]];
        let grid = Grid::from_positions(&p0, 1.0);
        let mut gs = GpuState::new(ctx, 1, grid.total_cells);
        let dt = 1.0e-2f32;
        gs.set_params(dt, [0.0; 3]); // no gravity
        gs.set_box([1.0, 1.0, 1.0], [0.0; 3], 0.0, 0.0); // unit periodic box
        gs.set_state(&p0, &[[1.0, 0.0, 0.0]], &[1.0], grid); // streaming +x
        gs.run_steps(20); // x: 0.9 + 1.0*0.01*20 = 1.1 -> wraps to 0.1
        let p = gs.download_pos()[0];
        assert!((p[0] - 0.1).abs() < 1e-4, "x did not wrap: {p:?}");
        assert!(p[0] >= 0.0 && p[0] < 1.0, "x out of box: {p:?}");
        eprintln!("resident periodic wrap: pos={p:?}");
    }

    #[test]
    fn resident_lees_edwards_y_cross() {
        let Some(ctx) = GpuContext::new() else { eprintln!("no GPU; skipping"); return; };
        let p0 = [[0.5f32, 0.95, 0.5]];
        let grid = Grid::from_positions(&p0, 1.0);
        let mut gs = GpuState::new(ctx, 1, grid.total_cells);
        let dt = 1.0e-2f32;
        gs.set_params(dt, [0.0; 3]);
        // Only y periodic (lx=lz=0); LE tilt 0.3, Δv 2.0 per y-image.
        gs.set_box([0.0, 1.0, 0.0], [0.0; 3], 0.3, 2.0);
        gs.set_state(&p0, &[[0.0, 1.0, 0.0]], &[1.0], grid); // streaming +y
        gs.run_steps(10); // y: 0.95 + 0.1 = 1.05 -> crosses once
        let p = gs.download_pos()[0];
        let v = gs.download_vel()[0];
        // One y-image crossing: y wrapped into [0,1), vx got the LE offset -Δv.
        assert!(p[1] >= 0.0 && p[1] < 1.0, "y not wrapped: {p:?}");
        assert!((v[0] - (-2.0)).abs() < 1e-4, "LE Δv not applied: vx={}", v[0]);
        assert!((v[1] - 1.0).abs() < 1e-4, "vy changed: {v:?}");
        eprintln!("resident Lees-Edwards y-cross: pos={p:?} vel={v:?}");
    }

    // A minimal Force hook: accumulates a constant force into every atom's slot.
    // Stands in for a real constitutive law (DEM contact, MD pair) to validate
    // that a registered hook runs each step and its force reaches integration.
    const DUMMY_WGSL: &str = r#"
struct HP { n: u32, fx: f32, fy: f32, fz: f32 };
@group(0) @binding(0) var<storage, read_write> force: array<f32>;
@group(0) @binding(1) var<uniform> hp: HP;
@compute @workgroup_size(64)
fn add_const(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= hp.n) { return; }
    let b = 3u * i;
    force[b]      = force[b]      + hp.fx;
    force[b + 1u] = force[b + 1u] + hp.fy;
    force[b + 2u] = force[b + 2u] + hp.fz;
}
"#;

    struct DummyForce {
        n: u32,
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
    }

    impl DummyForce {
        fn new(gs: &GpuState, f: [f32; 3]) -> Self {
            let device = &gs.context().device;
            #[repr(C)]
            #[derive(Clone, Copy, Pod, Zeroable)]
            struct HP { n: u32, fx: f32, fy: f32, fz: f32 }
            let hp = HP { n: gs.n() as u32, fx: f[0], fy: f[1], fz: f[2] };
            let params = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dummy hp"), size: std::mem::size_of::<HP>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
            });
            gs.context().queue.write_buffer(&params, 0, bytemuck::bytes_of(&hp));
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("dummy"), source: wgpu::ShaderSource::Wgsl(DUMMY_WGSL.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dummy bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                        count: None,
                    },
                ],
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dummy bg"), layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gs.force_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
                ],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dummy pl"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("dummy add_const"), layout: Some(&layout), module: &shader, entry_point: Some("add_const"),
                compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
            });
            DummyForce { n: gs.n() as u32, pipeline, bind_group }
        }
    }

    impl GpuForce for DummyForce {
        fn record(&self, pass: &mut wgpu::ComputePass) {
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.pipeline);
            pass.dispatch_workgroups(self.n.div_ceil(64).max(1), 1, 1);
        }
    }

    #[test]
    fn force_hook_reaches_integration() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let p0 = [[0.0f32, 0.0, 0.0]];
        let grid = Grid::from_positions(&p0, 1.0);
        let mut gs = GpuState::new(ctx, 1, grid.total_cells);
        let dt = 1.0e-3f32;
        // No gravity: motion is driven purely by the hook's constant force.
        gs.set_params(dt, [0.0, 0.0, 0.0]);
        gs.set_state(&p0, &[[0.0, 0.0, 0.0]], &[1.0], grid); // m = 1
        let a = [0.3f32, -0.7, 1.1]; // constant force == acceleration (m=1)
        let hook = DummyForce::new(&gs, a);
        gs.add_force_hook(Box::new(hook));

        let steps = 400;
        gs.run_steps(steps);
        let p = gs.download_pos()[0];
        let v = gs.download_vel()[0];
        let t = steps as f32 * dt; // 0.4 s

        // VV is exact for constant force: x = 0.5 a t^2, v = a t.
        for c in 0..3 {
            let ex = 0.5 * a[c] * t * t;
            let ev = a[c] * t;
            assert!((p[c] - ex).abs() < 1e-3, "pos[{c}]: {} vs {}", p[c], ex);
            assert!((v[c] - ev).abs() < 1e-3, "vel[{c}]: {} vs {}", v[c], ev);
        }
        eprintln!("Force hook: pos={p:?} vel={v:?} (constant-force analytic match)");
    }

    // A Force hook that reads the cell-list outputs (atom_cell → cell_start →
    // sorted_atoms) to find each atom's neighbors and applies a linear pair
    // repulsion. i-centric (each atom accumulates only its own force, no atomics)
    // — the exact binding pattern dirt's Hertz/Mindlin force will use. NOTE: it
    // only walks the atom's *home* cell, not the ±1 stencil, so it's correct only
    // when interacting pairs share a cell (the test forces that with a coarse
    // grid). The full stencil walk is the consumer kernel's job.
    const PAIR_WGSL: &str = r#"
struct PP { n: u32, k: f32, cutoff: f32, _p: f32 };
@group(0) @binding(0) var<storage, read>       pos: array<f32>;
@group(0) @binding(1) var<storage, read_write> force: array<f32>;
@group(0) @binding(2) var<storage, read>       cell_start: array<u32>;
@group(0) @binding(3) var<storage, read>       sorted_atoms: array<u32>;
@group(0) @binding(4) var<storage, read>       atom_cell: array<u32>;
@group(0) @binding(5) var<uniform>             pp: PP;
@compute @workgroup_size(64)
fn pair_repel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pp.n) { return; }
    let bi = 3u * i;
    let xi = vec3<f32>(pos[bi], pos[bi + 1u], pos[bi + 2u]);
    let c = atom_cell[i];
    var fi = vec3<f32>(0.0, 0.0, 0.0);
    let s = cell_start[c];
    let e = cell_start[c + 1u];
    for (var m = s; m < e; m = m + 1u) {
        let j = sorted_atoms[m];
        if (j == i) { continue; }
        let bj = 3u * j;
        let d = xi - vec3<f32>(pos[bj], pos[bj + 1u], pos[bj + 2u]);
        let r = length(d);
        if (r < pp.cutoff && r > 1e-6) {
            fi = fi + pp.k * (pp.cutoff - r) * (d / r);
        }
    }
    force[bi]      = force[bi]      + fi.x;
    force[bi + 1u] = force[bi + 1u] + fi.y;
    force[bi + 2u] = force[bi + 2u] + fi.z;
}
"#;

    struct PairRepel {
        n: u32,
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
    }

    impl PairRepel {
        fn new(gs: &GpuState, k: f32, cutoff: f32) -> Self {
            let device = &gs.context().device;
            #[repr(C)]
            #[derive(Clone, Copy, Pod, Zeroable)]
            struct PP { n: u32, k: f32, cutoff: f32, _p: f32 }
            let pp = PP { n: gs.n() as u32, k, cutoff, _p: 0.0 };
            let params = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pair pp"), size: std::mem::size_of::<PP>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
            });
            gs.context().queue.write_buffer(&params, 0, bytemuck::bytes_of(&pp));
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pair"), source: wgpu::ShaderSource::Wgsl(PAIR_WGSL.into()),
            });
            let st = |binding, read_only| wgpu::BindGroupLayoutEntry {
                binding, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pair bgl"),
                entries: &[
                    st(0, true), st(1, false), st(2, true), st(3, true), st(4, true),
                    wgpu::BindGroupLayoutEntry {
                        binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                        count: None,
                    },
                ],
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pair bg"), layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gs.pos_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: gs.force_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: gs.cell_start_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: gs.sorted_atoms_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: gs.atom_cell_buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: params.as_entire_binding() },
                ],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pair pl"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("pair pair_repel"), layout: Some(&layout), module: &shader, entry_point: Some("pair_repel"),
                compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
            });
            PairRepel { n: gs.n() as u32, pipeline, bind_group }
        }
    }

    impl GpuForce for PairRepel {
        fn record(&self, pass: &mut wgpu::ComputePass) {
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.pipeline);
            pass.dispatch_workgroups(self.n.div_ceil(64).max(1), 1, 1);
        }
    }

    // A hook that overwrites an aux DOF's `rate` with a constant each step (zero
    // force). Stands in for a torque-producing contact law to validate that the
    // resident loop integrates the aux `state` from the hook-supplied rate.
    const AUX_RATE_WGSL: &str = r#"
struct RP { n: u32, fx: f32, fy: f32, fz: f32 };
@group(0) @binding(0) var<storage, read_write> rate: array<f32>;
@group(0) @binding(1) var<uniform> rp: RP;
@compute @workgroup_size(64)
fn set_rate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= rp.n) { return; }
    let b = 3u * i;
    rate[b] = rp.fx; rate[b + 1u] = rp.fy; rate[b + 2u] = rp.fz;
}
"#;

    struct AuxRateConst { n: u32, pipeline: wgpu::ComputePipeline, bind_group: wgpu::BindGroup }

    impl AuxRateConst {
        fn new(gs: &GpuState, aux_idx: usize, rate: [f32; 3]) -> Self {
            let device = &gs.context().device;
            #[repr(C)]
            #[derive(Clone, Copy, Pod, Zeroable)]
            struct RP { n: u32, fx: f32, fy: f32, fz: f32 }
            let rp = RP { n: gs.n() as u32, fx: rate[0], fy: rate[1], fz: rate[2] };
            let params = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("auxrate rp"), size: std::mem::size_of::<RP>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
            });
            gs.context().queue.write_buffer(&params, 0, bytemuck::bytes_of(&rp));
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("auxrate"), source: wgpu::ShaderSource::Wgsl(AUX_RATE_WGSL.into()),
            });
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("auxrate bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                        count: None,
                    },
                ],
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("auxrate bg"), layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gs.aux_rate_buffer(aux_idx).as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
                ],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("auxrate pl"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("auxrate set_rate"), layout: Some(&layout), module: &shader, entry_point: Some("set_rate"),
                compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
            });
            AuxRateConst { n: gs.n() as u32, pipeline, bind_group }
        }
    }

    impl GpuForce for AuxRateConst {
        fn record(&self, pass: &mut wgpu::ComputePass) {
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.pipeline);
            pass.dispatch_workgroups(self.n.div_ceil(64).max(1), 1, 1);
        }
    }

    #[test]
    fn aux_dof_integrates_from_hook_rate() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let p0 = [[0.0f32, 0.0, 0.0]];
        let grid = Grid::from_positions(&p0, 1.0);
        let mut gs = GpuState::new(ctx, 1, grid.total_cells);
        let dt = 1.0e-3f32;
        gs.set_params(dt, [0.0, 0.0, 0.0]);
        gs.set_state(&p0, &[[0.0; 3]], &[1.0], grid);

        // Register an aux DOF (like granular rotation): inv_coeff = 1, state = 0.
        let aux = gs.add_aux_dof();
        gs.set_aux_inv_coeff(aux, &[1.0]);
        gs.set_aux_state(aux, &[[0.0, 0.0, 0.0]]);
        let rate = [0.5f32, -0.3, 0.8];
        gs.add_force_hook(Box::new(AuxRateConst::new(&gs, aux, rate)));

        let steps = 400;
        gs.run_steps(steps);
        let s = gs.download_aux_state(aux)[0];
        let t = steps as f32 * dt; // 0.4 s

        // d(state)/dt = inv_coeff * rate = rate (constant) → state = rate * t.
        for c in 0..3 {
            let ex = rate[c] * t;
            assert!((s[c] - ex).abs() < 1e-4, "aux_state[{c}]: {} vs {}", s[c], ex);
        }
        eprintln!("aux DOF: state={s:?} (linear integration of hook rate)");
    }

    #[test]
    fn cell_list_hook_pair_repulsion_is_symmetric() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        // Two atoms on the x-axis, 0.6 apart. A coarse grid (bin = 5) puts both in
        // the same cell so the home-cell-only hook sees the pair.
        let p0 = [[-0.3f32, 0.0, 0.0], [0.3, 0.0, 0.0]];
        let grid = Grid::from_positions(&p0, 5.0);
        let mut gs = GpuState::new(ctx, 2, grid.total_cells);
        gs.set_params(1.0e-3, [0.0, 0.0, 0.0]); // no gravity
        gs.set_state(&p0, &[[0.0; 3]; 2], &[1.0, 1.0], grid);
        gs.add_force_hook(Box::new(PairRepel::new(&gs, 50.0, 1.0))); // cutoff 1.0 > 0.6

        gs.run_steps(300);
        let p = gs.download_pos();
        let v = gs.download_vel();

        // Repulsion pushed them apart, symmetric about the origin (equal masses,
        // i-centric mirror forces), and momentum is conserved (COM at rest).
        assert!(p[0][0] < -0.3 && p[1][0] > 0.3, "did not separate: {p:?}");
        assert!((p[0][0] + p[1][0]).abs() < 1e-4, "asymmetric x: {p:?}");
        assert!((v[0][0] + v[1][0]).abs() < 1e-4, "momentum not conserved: {v:?}");
        // Motion stays on the x-axis.
        for a in 0..2 {
            assert!(p[a][1].abs() < 1e-5 && p[a][2].abs() < 1e-5, "off-axis: {p:?}");
        }
        eprintln!("cell-list hook pair-repel: x=[{}, {}] (symmetric separation)", p[0][0], p[1][0]);
    }
}
