// Resident core kernels (substrate): generic body force + velocity-Verlet
// integration over the resident pos/vel/force/inv_mass buffers. No physics
// constitutive law — that's a Force hook the consumer registers between the two
// integration half-steps.

struct Params {
    n: u32,
    dt: f32,
    gx: f32,
    gy: f32,
    gz: f32,
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
};

@group(0) @binding(0) var<storage, read_write> pos: array<f32>;
@group(0) @binding(1) var<storage, read_write> vel: array<f32>;
@group(0) @binding(2) var<storage, read_write> force: array<f32>;
@group(0) @binding(3) var<storage, read>       inv_mass: array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

// Seed the force accumulator with the body force m*g (i-centric overwrite; the
// Force hook then accumulates contact/wall contributions with +=).
@compute @workgroup_size(64)
fn seed_gravity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let b = 3u * i;
    let im = inv_mass[i];
    let m = select(0.0, 1.0 / im, im > 0.0);
    force[b]      = m * params.gx;
    force[b + 1u] = m * params.gy;
    force[b + 2u] = m * params.gz;
}

// First half: half-step velocity kick (using current force) + position drift,
// then re-seed the force accumulator with the body force m*g for this step. The
// kick consumes the previous step's full force before the overwrite, so this
// fuses the gravity seed into the integrate pass (one fewer O(N) dispatch than a
// separate seed_gravity kernel). Force hooks then accumulate with += as before.
@compute @workgroup_size(64)
fn integrate_initial(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let b = 3u * i;
    let im = inv_mass[i];
    let hdtm = 0.5 * params.dt * im;
    var vx = vel[b]      + hdtm * force[b];
    let vy = vel[b + 1u] + hdtm * force[b + 1u];
    let vz = vel[b + 2u] + hdtm * force[b + 2u];
    var px = pos[b]      + vx * params.dt;
    var py = pos[b + 1u] + vy * params.dt;
    var pz = pos[b + 2u] + vz * params.dt;
    // On-device PBC + Lees–Edwards remap (l == 0 → axis non-periodic, skipped).
    // A y-image crossing shifts x by the tilt and vx by Δv (the LE boundary).
    if (params.lz > 0.0) { pz = pz - params.lz * floor((pz - params.oz) / params.lz); }
    if (params.ly > 0.0) {
        let img = floor((py - params.oy) / params.ly);
        py = py - params.ly * img;
        px = px - params.tilt_xy * img;
        vx = vx - params.dv_xy * img;
    }
    if (params.lx > 0.0) { px = px - params.lx * floor((px - params.ox) / params.lx); }
    vel[b]      = vx;
    vel[b + 1u] = vy;
    vel[b + 2u] = vz;
    pos[b]      = px;
    pos[b + 1u] = py;
    pos[b + 2u] = pz;
    // Seed next force = m*g (clump sub-spheres with inv_mass==0 -> no body force).
    let m = select(0.0, 1.0 / im, im > 0.0);
    force[b]      = m * params.gx;
    force[b + 1u] = m * params.gy;
    force[b + 2u] = m * params.gz;
}

// Second half: completing velocity kick using the post-force force.
@compute @workgroup_size(64)
fn integrate_final(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let b = 3u * i;
    let hdtm = 0.5 * params.dt * inv_mass[i];
    vel[b]      = vel[b]      + hdtm * force[b];
    vel[b + 1u] = vel[b + 1u] + hdtm * force[b + 1u];
    vel[b + 2u] = vel[b + 2u] + hdtm * force[b + 2u];
}
