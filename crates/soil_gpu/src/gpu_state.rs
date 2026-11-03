//! Resident GPU simulation state (substrate), the core of the resident loop.
//!
//! Owns the per-atom columns (pos/vel/force/inv_mass, flat xyz floats) and drives a velocity
//! Verlet step with a generic body force, entirely on the device. The constitutive force (DEM
//! contact, MD pair, peridynamics bond, walls) is a [`GpuForce`] hook the consumer registers;
//! it runs between the two half-steps and accumulates into `force` (i-centric, no atomics).

use cubecl::prelude::*;
use grass_gpu::cubecl;
use grass_gpu::{DualBuffer, GpuDevice, GpuReal};

use crate::cell_list::{CellList, DeviceColumn, Grid};

const WG: u32 = 64;

// Layout of the float parameter column.
const DT: usize = 0;
const GX: usize = 1;
const LX: usize = 4;
const OX: usize = 7;
const TILT_XY: usize = 10;
const DV_XY: usize = 11;

/// Sets the force to the body force `m g`, ready for hooks to add to.
#[cube(launch_unchecked)]
fn seed_gravity<F: Float>(sizes: &[u32], params: &[F], inv_mass: &[F], force: &mut [F]) {
    let i = ABSOLUTE_POS;
    if (i as u32) < sizes[0] {
        let im = inv_mass[i];
        let mut m = F::new(0.0f32);
        if im > F::new(0.0f32) {
            m = F::new(1.0f32) / im;
        }
        force[3 * i] = m * params[GX];
        force[3 * i + 1] = m * params[GX + 1];
        force[3 * i + 2] = m * params[GX + 2];
    }
}

/// First half: half kick with the current force, drift, wrap periodic axes (`l == 0` means
/// not periodic), then re-seed the force for this step. A y-image crossing shifts x by the
/// tilt and vx by the shear velocity, which is the Lees-Edwards boundary.
#[cube(launch_unchecked)]
fn integrate_initial<F: Float>(
    sizes: &[u32],
    params: &[F],
    inv_mass: &[F],
    pos: &mut [F],
    vel: &mut [F],
    force: &mut [F],
) {
    let i = ABSOLUTE_POS;
    if (i as u32) < sizes[0] {
        let zero = F::new(0.0f32);
        let im = inv_mass[i];
        let dt = params[DT];
        let h = F::new(0.5f32) * dt * im;
        let mut vx = vel[3 * i] + h * force[3 * i];
        let vy = vel[3 * i + 1] + h * force[3 * i + 1];
        let vz = vel[3 * i + 2] + h * force[3 * i + 2];
        let mut px = pos[3 * i] + vx * dt;
        let mut py = pos[3 * i + 1] + vy * dt;
        let mut pz = pos[3 * i + 2] + vz * dt;
        let (lx, ly, lz) = (params[LX], params[LX + 1], params[LX + 2]);
        if lz > zero {
            pz -= lz * F::floor((pz - params[OX + 2]) / lz);
        }
        if ly > zero {
            let image = F::floor((py - params[OX + 1]) / ly);
            py -= ly * image;
            px -= params[TILT_XY] * image;
            vx -= params[DV_XY] * image;
        }
        if lx > zero {
            px -= lx * F::floor((px - params[OX]) / lx);
        }
        vel[3 * i] = vx;
        vel[3 * i + 1] = vy;
        vel[3 * i + 2] = vz;
        pos[3 * i] = px;
        pos[3 * i + 1] = py;
        pos[3 * i + 2] = pz;
        // Atoms with no inverse mass (clump sub-spheres) feel no body force.
        let mut m = zero;
        if im > zero {
            m = F::new(1.0f32) / im;
        }
        force[3 * i] = m * params[GX];
        force[3 * i + 1] = m * params[GX + 1];
        force[3 * i + 2] = m * params[GX + 2];
    }
}

/// Second half: the completing velocity kick, with the force from this step.
#[cube(launch_unchecked)]
fn integrate_final<F: Float>(
    sizes: &[u32],
    params: &[F],
    inv_mass: &[F],
    force: &[F],
    vel: &mut [F],
) {
    let i = ABSOLUTE_POS;
    if (i as u32) < sizes[0] {
        let h = F::new(0.5f32) * params[DT] * inv_mass[i];
        vel[3 * i] += h * force[3 * i];
        vel[3 * i + 1] += h * force[3 * i + 1];
        vel[3 * i + 2] += h * force[3 * i + 2];
    }
}

/// A half kick for an auxiliary degree of freedom: the same as velocity, with no drift.
#[cube(launch_unchecked)]
fn aux_kick<F: Float>(sizes: &[u32], params: &[F], rate: &[F], inv_coeff: &[F], state: &mut [F]) {
    let i = ABSOLUTE_POS;
    if (i as u32) < sizes[0] {
        let h = F::new(0.5f32) * params[DT] * inv_coeff[i];
        state[3 * i] += h * rate[3 * i];
        state[3 * i + 1] += h * rate[3 * i + 1];
        state[3 * i + 2] += h * rate[3 * i + 2];
    }
}

/// An auxiliary integrated degree of freedom: a velocity-like `state` driven by `rate` with
/// per-atom inverse coefficient `inv_coeff`. Granular rotation maps state to angular
/// velocity, rate to torque, and the coefficient to 1/inertia.
pub struct AuxDof {
    /// The integrated value, flat xyz.
    pub state: DualBuffer<GpuReal>,
    /// What drives it, which a hook writes each step.
    pub rate: DualBuffer<GpuReal>,
    /// Per-atom inverse coefficient.
    pub inv_coeff: DualBuffer<GpuReal>,
}

/// What a hook's kernels can read and write during a step. Current-copy bookkeeping for these
/// is the loop's job, so hooks take arguments with `arg()`.
pub struct StepColumns<'a> {
    /// The device.
    pub gpu: &'a GpuDevice,
    /// Atoms the arrays cover.
    pub n: usize,
    /// Atoms that need a force, which is fewer when the rest are ghosts.
    pub n_force: usize,
    /// Positions, flat xyz.
    pub pos: &'a DualBuffer<GpuReal>,
    /// Velocities, flat xyz.
    pub vel: &'a DualBuffer<GpuReal>,
    /// Forces, flat xyz. Hooks add to this.
    pub force: &'a DualBuffer<GpuReal>,
    /// Inverse masses.
    pub inv_mass: &'a DualBuffer<GpuReal>,
    /// The neighbor build, current for these positions.
    pub cells: &'a CellList,
    /// Auxiliary degrees of freedom.
    pub aux: &'a [AuxDof],
}

/// A constitutive force a consumer registers into the resident loop. Runs between the half
/// steps and adds into `force` (i-centric, no atomics).
pub trait GpuForce {
    /// Launches this step's kernels.
    fn launch(&mut self, columns: &StepColumns<'_>);
    /// A history-neutral evaluation for the start of a run: the force at the current
    /// positions without advancing contact history, so starting a window afresh does not
    /// double-advance a spring. Default: the same as `launch`, right for stateless hooks.
    fn launch_prime(&mut self, columns: &StepColumns<'_>) {
        self.launch(columns);
    }
    /// The atom set or grid changed: the first `n_all` atoms are binned on `grid`, and this
    /// hook computes force for the first `n_force` of them.
    fn set_active(&mut self, n_all: usize, n_force: usize, grid: &Grid) {
        let _ = (n_all, n_force, grid);
    }
}

/// Resident core simulation state on the GPU.
pub struct GpuState {
    gpu: GpuDevice,
    n: usize,
    n_force: usize,
    grid: Grid,
    cell_list: CellList,
    pos: DualBuffer<GpuReal>,
    vel: DualBuffer<GpuReal>,
    force: DualBuffer<GpuReal>,
    inv_mass: DualBuffer<GpuReal>,
    params: [GpuReal; 12],
    params_column: DeviceColumn,
    sizes: DeviceColumn,
    aux_dofs: Vec<AuxDof>,
    hooks: Vec<Box<dyn GpuForce>>,
}

fn uploaded<T: CubeElement>(gpu: &GpuDevice, data: &[T]) -> DeviceColumn {
    DeviceColumn::from_slice(gpu, data)
}

impl GpuState {
    /// State for `n` atoms binned on `grid`.
    pub fn new(gpu: GpuDevice, n: usize, grid: Grid) -> Self {
        let cell_list = CellList::new(gpu.clone(), n, grid);
        let zeros = |len| DualBuffer::filled(&gpu, len, 0.0 as GpuReal);
        let mut params = [0.0; 12];
        params[DT] = 1.0e-4;
        Self {
            n,
            n_force: n,
            grid,
            cell_list,
            pos: zeros(3 * n),
            vel: zeros(3 * n),
            force: zeros(3 * n),
            inv_mass: zeros(n),
            params,
            params_column: uploaded(&gpu, &params),
            sizes: uploaded(&gpu, &[n as u32]),
            aux_dofs: Vec::new(),
            hooks: Vec::new(),
            gpu,
        }
    }

    /// Register a constitutive force hook. Hooks run in registration order.
    pub fn add_force_hook(&mut self, hook: Box<dyn GpuForce>) {
        self.hooks.push(hook);
    }

    /// Atoms the arrays cover.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The device.
    pub fn device(&self) -> &GpuDevice {
        &self.gpu
    }

    /// Register an auxiliary integrated degree of freedom, returning its index.
    pub fn add_aux_dof(&mut self) -> usize {
        let zeros = |len| DualBuffer::filled(&self.gpu, len, 0.0 as GpuReal);
        let aux = AuxDof {
            state: zeros(3 * self.n),
            rate: zeros(3 * self.n),
            inv_coeff: zeros(self.n),
        };
        self.aux_dofs.push(aux);
        self.aux_dofs.len() - 1
    }

    /// Replace an auxiliary state from the CPU, uploaded by the next launch.
    pub fn set_aux_state(&mut self, i: usize, state: &[GpuReal]) {
        self.aux_dofs[i]
            .state
            .host_overwrite()
            .copy_from_slice(state);
    }

    /// Replace an auxiliary inverse coefficient from the CPU.
    pub fn set_aux_inv_coeff(&mut self, i: usize, inv_coeff: &[GpuReal]) {
        self.aux_dofs[i]
            .inv_coeff
            .host_overwrite()
            .copy_from_slice(inv_coeff);
    }

    /// Replace positions from the CPU, uploaded by the next launch.
    pub fn write_pos(&mut self, pos: &[GpuReal]) {
        self.pos.host_overwrite().copy_from_slice(pos);
    }

    /// Replace velocities from the CPU, uploaded by the next launch.
    pub fn write_vel(&mut self, vel: &[GpuReal]) {
        self.vel.host_overwrite().copy_from_slice(vel);
    }

    /// Replace inverse masses from the CPU, uploaded by the next launch.
    pub fn write_inv_mass(&mut self, inv_mass: &[GpuReal]) {
        self.inv_mass.host_overwrite().copy_from_slice(inv_mass);
    }

    /// Set the state from the CPU.
    pub fn set_state(&mut self, pos: &[GpuReal], vel: &[GpuReal], inv_mass: &[GpuReal]) {
        self.write_pos(pos);
        self.write_vel(vel);
        self.write_inv_mass(inv_mass);
    }

    /// Change the atom set and grid, for a host that re-uploads every column afterward.
    /// `n_force` is how many atoms hooks compute force for.
    pub fn set_active(&mut self, n_all: usize, n_force: usize, grid: Grid) {
        self.n = n_all;
        self.n_force = n_force;
        self.grid = grid;
        self.cell_list.set_active(n_all, grid);
        self.sizes = uploaded(&self.gpu, &[n_all as u32]);
        for column in [&mut self.pos, &mut self.vel, &mut self.force] {
            column.resize_overwrite(3 * n_all, 0.0);
        }
        self.inv_mass.resize_overwrite(n_all, 0.0);
        for aux in &mut self.aux_dofs {
            aux.state.resize_overwrite(3 * n_all, 0.0);
            aux.rate.resize_overwrite(3 * n_all, 0.0);
            aux.inv_coeff.resize_overwrite(n_all, 0.0);
        }
        for hook in &mut self.hooks {
            hook.set_active(n_all, n_force, &grid);
        }
    }

    /// Change the grid only, leaving every column as it is.
    pub fn set_grid(&mut self, grid: Grid) {
        self.grid = grid;
        self.cell_list.set_grid(grid);
        let (n_all, n_force) = (self.n, self.n_force);
        for hook in &mut self.hooks {
            hook.set_active(n_all, n_force, &grid);
        }
    }

    /// The time step and body acceleration.
    pub fn set_params(&mut self, dt: GpuReal, gravity: [GpuReal; 3]) {
        self.params[DT] = dt;
        self.params[GX..GX + 3].copy_from_slice(&gravity);
        self.params_column = uploaded(&self.gpu, &self.params);
    }

    /// On-device periodic boundaries and Lees-Edwards shear. `lengths` is the box size per
    /// axis (0 for non-periodic), `origin` its low corner, `tilt_xy` the x shift per y image,
    /// `dv_xy` the matching velocity offset.
    pub fn set_box(
        &mut self,
        lengths: [GpuReal; 3],
        origin: [GpuReal; 3],
        tilt_xy: GpuReal,
        dv_xy: GpuReal,
    ) {
        self.params[LX..LX + 3].copy_from_slice(&lengths);
        self.params[OX..OX + 3].copy_from_slice(&origin);
        self.params[TILT_XY] = tilt_xy;
        self.params[DV_XY] = dv_xy;
        self.params_column = uploaded(&self.gpu, &self.params);
    }

    /// Run `steps` velocity Verlet steps with no transfer between them. The first half kick
    /// needs the force at the entry positions, so it is primed once before the loop, without
    /// advancing history. The force from each step serves both that step's final kick and the
    /// next step's first kick, so there is one force evaluation per step.
    pub fn run_steps(&mut self, steps: usize) {
        self.before_kernels();
        self.before_integrating();
        self.seed_and_bin();
        self.each_hook(|hook, columns| hook.launch_prime(columns));
        self.step_loop(steps);
    }

    /// Continue for `steps` more without re-priming, trusting the force left by the previous
    /// window. Re-priming would evaluate velocity-dependent damping at the end-of-step
    /// velocity, not the half-kick velocity the integrator used, so windows would no longer
    /// stitch together. If the host changed positions or velocities, use
    /// [`run_steps`](Self::run_steps).
    pub fn run_steps_continue(&mut self, steps: usize) {
        self.before_kernels();
        self.before_integrating();
        self.step_loop(steps);
    }

    /// Evaluate the force once at the current positions without integrating.
    pub fn eval_force_once(&mut self) {
        self.before_kernels();
        // Every force and rate is about to be replaced, so nothing need go up first.
        self.force.gpu_overwrite();
        for aux in &mut self.aux_dofs {
            aux.rate.gpu_overwrite();
        }
        self.seed_and_bin();
        self.each_hook(|hook, columns| hook.launch(columns));
    }

    /// Uploads whatever the CPU changed, and marks every column the kernels write as newer
    /// on the GPU.
    fn before_kernels(&mut self) {
        self.inv_mass.gpu_read();
        self.pos.gpu_write();
        self.vel.gpu_write();
        for aux in &mut self.aux_dofs {
            aux.inv_coeff.gpu_read();
            aux.state.gpu_write();
        }
    }

    /// The step loop reads forces and rates from the previous step, so those stay valid.
    fn before_integrating(&mut self) {
        self.force.gpu_write();
        for aux in &mut self.aux_dofs {
            aux.rate.gpu_write();
        }
    }

    fn groups(&self) -> CubeCount {
        CubeCount::Static((self.n as u32).div_ceil(WG).max(1), 1, 1)
    }

    fn seed_and_bin(&self) {
        self.cell_list.launch(self.pos.arg());
        // SAFETY: bounded by `sizes`, which matches these columns.
        unsafe {
            seed_gravity::launch_unchecked::<GpuReal>(
                &self.gpu.client,
                self.groups(),
                CubeDim::new_1d(WG),
                self.sizes.arg(),
                self.params_column.arg(),
                self.inv_mass.arg(),
                self.force.arg(),
            );
        }
    }

    fn aux_kicks(&self) {
        for aux in &self.aux_dofs {
            // SAFETY: bounded by `sizes`, which matches these columns.
            unsafe {
                aux_kick::launch_unchecked::<GpuReal>(
                    &self.gpu.client,
                    self.groups(),
                    CubeDim::new_1d(WG),
                    self.sizes.arg(),
                    self.params_column.arg(),
                    aux.rate.arg(),
                    aux.inv_coeff.arg(),
                    aux.state.arg(),
                );
            }
        }
    }

    fn each_hook(&mut self, mut run: impl FnMut(&mut Box<dyn GpuForce>, &StepColumns<'_>)) {
        let mut hooks = std::mem::take(&mut self.hooks);
        let columns = StepColumns {
            gpu: &self.gpu,
            n: self.n,
            n_force: self.n_force,
            pos: &self.pos,
            vel: &self.vel,
            force: &self.force,
            inv_mass: &self.inv_mass,
            cells: &self.cell_list,
            aux: &self.aux_dofs,
        };
        for hook in &mut hooks {
            run(hook, &columns);
        }
        self.hooks = hooks;
    }

    fn step_loop(&mut self, steps: usize) {
        let dim = CubeDim::new_1d(WG);
        for _ in 0..steps {
            // SAFETY: bounded by `sizes`, which matches these columns.
            unsafe {
                integrate_initial::launch_unchecked::<GpuReal>(
                    &self.gpu.client,
                    self.groups(),
                    dim,
                    self.sizes.arg(),
                    self.params_column.arg(),
                    self.inv_mass.arg(),
                    self.pos.arg(),
                    self.vel.arg(),
                    self.force.arg(),
                );
            }
            self.aux_kicks();
            self.cell_list.launch(self.pos.arg());
            self.each_hook(|hook, columns| hook.launch(columns));
            // SAFETY: as above.
            unsafe {
                integrate_final::launch_unchecked::<GpuReal>(
                    &self.gpu.client,
                    self.groups(),
                    dim,
                    self.sizes.arg(),
                    self.params_column.arg(),
                    self.inv_mass.arg(),
                    self.force.arg(),
                    self.vel.arg(),
                );
            }
            self.aux_kicks();
        }
    }

    /// The position column as a kernel argument, for a kernel outside the step loop.
    pub fn pos_arg(&mut self) -> BufferArg {
        self.pos.gpu_read()
    }

    /// Blocks until all launched work has finished.
    pub fn wait(&self) {
        self.gpu.wait();
    }

    /// Brings positions, velocities, and auxiliary states down together, with one wait.
    pub fn sync_state_to_host(&mut self) {
        let mut columns: Vec<&mut DualBuffer<GpuReal>> = vec![&mut self.pos, &mut self.vel];
        columns.extend(self.aux_dofs.iter_mut().map(|aux| &mut aux.state));
        DualBuffer::sync_all_to_host(&mut columns);
    }

    /// Brings forces and one auxiliary rate down together, with one wait.
    pub fn sync_force_and_rate(&mut self, aux: usize) {
        DualBuffer::sync_all_to_host(&mut [&mut self.force, &mut self.aux_dofs[aux].rate]);
    }

    /// Positions on the CPU, flat xyz, downloaded only if the GPU copy is newer.
    pub fn pos(&mut self) -> &[GpuReal] {
        self.pos.host()
    }
    /// Velocities on the CPU, flat xyz, downloaded only if the GPU copy is newer.
    pub fn vel(&mut self) -> &[GpuReal] {
        self.vel.host()
    }
    /// Forces on the CPU, flat xyz, downloaded only if the GPU copy is newer.
    pub fn force(&mut self) -> &[GpuReal] {
        self.force.host()
    }
    /// An auxiliary state on the CPU, flat xyz, downloaded only if the GPU copy is newer.
    pub fn aux_state(&mut self, i: usize) -> &[GpuReal] {
        self.aux_dofs[i].state.host()
    }
    /// An auxiliary rate on the CPU, flat xyz, downloaded only if the GPU copy is newer.
    pub fn aux_rate(&mut self, i: usize) -> &[GpuReal] {
        self.aux_dofs[i].rate.host()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: GpuReal = 2.0e-3;

    fn gpu() -> Option<GpuDevice> {
        let device = GpuDevice::default_device().ok();
        if device.is_none() {
            eprintln!("no GPU adapter; skipping");
        }
        device
    }

    #[test]
    fn freefall_matches_analytic() {
        let Some(gpu) = gpu() else { return };
        let p0 = [0.0, 0.0, 10.0];
        let grid = Grid::from_positions(&p0, 1.0);
        let mut state = GpuState::new(gpu, 1, grid);
        let (dt, g) = (1.0e-3, -9.81);
        state.set_params(dt, [0.0, 0.0, g]);
        state.set_state(&p0, &[1.0, 0.0, 2.0], &[1.0]);
        state.run_steps(500);
        let t = 500.0 * dt;
        let z = 10.0 + 2.0 * t + 0.5 * g * t * t;
        let p = state.pos().to_vec();
        assert!(
            (p[0] - t).abs() < TOL && (p[2] - z).abs() < TOL,
            "{p:?} vs z {z}"
        );
        let v = state.vel().to_vec();
        assert!((v[2] - (2.0 + g * t)).abs() < TOL, "{v:?}");
    }

    #[test]
    fn periodic_wrap_and_lees_edwards_crossing() {
        let Some(gpu) = gpu() else { return };
        let p0 = [0.5, 0.95, 0.5];
        let grid = Grid::for_box([0.0; 3], [1.0; 3], [true; 3], &p0, 0.2, 1000);
        let mut state = GpuState::new(gpu, 1, grid);
        state.set_params(1.0e-2, [0.0; 3]);
        state.set_box([1.0; 3], [0.0; 3], 0.25, 0.5);
        state.set_state(&p0, &[0.0, 1.0, 0.0], &[1.0]);
        state.run_steps(10);
        // Crossed y once: y wraps, x shifts by the tilt, vx by the shear velocity.
        let p = state.pos().to_vec();
        assert!((p[1] - 0.05).abs() < TOL, "{p:?}");
        assert!((p[0] - 0.25).abs() < 0.03, "{p:?}");
        let v = state.vel().to_vec();
        assert!((v[0] + 0.5).abs() < TOL, "{v:?}");
    }

    /// Pushes atoms apart when within a cutoff, using the cell list.
    #[cube(launch_unchecked)]
    fn repel<F: Float>(
        sizes: &[u32],
        pos: &[F],
        cell_start: &[u32],
        sorted_atoms: &[u32],
        force: &mut [F],
        rate: &mut [F],
    ) {
        let i = ABSOLUTE_POS;
        if (i as u32) < sizes[0] {
            let total = sizes[1] as usize;
            let mut c = 0;
            while c < total {
                let mut m = cell_start[c];
                while m < cell_start[c + 1] {
                    let j = sorted_atoms[m as usize] as usize;
                    if j != i {
                        let dx = pos[3 * j] - pos[3 * i];
                        if F::abs(dx) < F::new(1.0f32) {
                            force[3 * i] -= F::new(50.0f32) * dx;
                            rate[3 * i + 2] = F::new(2.0f32);
                        }
                    }
                    m += 1u32;
                }
                c += 1;
            }
        }
    }

    struct Repel {
        sizes: DeviceColumn,
    }
    impl GpuForce for Repel {
        fn launch(&mut self, c: &StepColumns<'_>) {
            // SAFETY: bounded by `sizes`.
            unsafe {
                repel::launch_unchecked::<GpuReal>(
                    &c.gpu.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(64),
                    self.sizes.arg(),
                    c.pos.arg(),
                    c.cells.cell_start(),
                    c.cells.sorted_atoms(),
                    c.force.arg(),
                    c.aux[0].rate.arg(),
                );
            }
        }
    }

    #[test]
    fn hook_force_reaches_the_integrator_and_aux_state() {
        let Some(gpu) = gpu() else { return };
        let p0 = [-0.3, 0.0, 0.0, 0.3, 0.0, 0.0];
        let grid = Grid::from_positions(&p0, 2.0);
        let mut state = GpuState::new(gpu.clone(), 2, grid);
        state.set_params(1.0e-3, [0.0; 3]);
        state.set_state(&p0, &[0.0; 6], &[1.0, 1.0]);
        let aux = state.add_aux_dof();
        state.set_aux_inv_coeff(aux, &[1.0, 1.0]);
        state.set_aux_state(aux, &[0.0; 6]);
        let sizes = DeviceColumn::from_slice(&gpu, &[2u32, grid.total_cells as u32]);
        state.add_force_hook(Box::new(Repel { sizes }));
        state.run_steps(300);

        let p = state.pos().to_vec();
        let v = state.vel().to_vec();
        assert!(p[0] < -0.3 && p[3] > 0.3, "did not separate: {p:?}");
        assert!((p[0] + p[3]).abs() < 1e-4, "asymmetric: {p:?}");
        assert!((v[0] + v[3]).abs() < 1e-4, "momentum: {v:?}");
        // Rate 2 for 300 steps of 1e-3 integrates to 0.6.
        let spin = state.aux_state(aux)[2];
        assert!((spin - 0.6).abs() < TOL, "{spin}");
    }
}
