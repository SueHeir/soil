// Velocity Verlet integration — single-precision (f32) compute kernels.
//
// Apple GPUs (Metal/WGSL) have no f64, so these kernels are intrinsically f32.
// Positions, velocities and forces are stored as flat `array<f32>` of length
// 3*N (component j of atom i lives at index 3*i + j), which matches the tightly
// packed `Vec<[f32; 3]>` host layout and sidesteps WGSL's vec3 alignment rules.

struct Params {
    dt: f32,
    n: u32,
};

@group(0) @binding(0) var<storage, read_write> pos: array<f32>;
@group(0) @binding(1) var<storage, read_write> vel: array<f32>;
@group(0) @binding(2) var<storage, read>       force: array<f32>;
@group(0) @binding(3) var<storage, read>       inv_mass: array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

// First half of velocity Verlet: half-step velocity kick + full-step position
// drift. Mirrors soil_verlet::initial_integration.
@compute @workgroup_size(64)
fn initial(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let b = 3u * i;
    let hdtm = 0.5 * params.dt * inv_mass[i];

    let vx = vel[b]      + hdtm * force[b];
    let vy = vel[b + 1u] + hdtm * force[b + 1u];
    let vz = vel[b + 2u] + hdtm * force[b + 2u];
    vel[b]      = vx;
    vel[b + 1u] = vy;
    vel[b + 2u] = vz;

    pos[b]      = pos[b]      + vx * params.dt;
    pos[b + 1u] = pos[b + 1u] + vy * params.dt;
    pos[b + 2u] = pos[b + 2u] + vz * params.dt;
}

// Second half of velocity Verlet: completing velocity kick using the new
// forces. Mirrors soil_verlet::final_integration. ("final" is avoided as it is
// a WGSL reserved keyword.)
@compute @workgroup_size(64)
fn final_kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let b = 3u * i;
    let hdtm = 0.5 * params.dt * inv_mass[i];

    vel[b]      = vel[b]      + hdtm * force[b];
    vel[b + 1u] = vel[b + 1u] + hdtm * force[b + 1u];
    vel[b + 2u] = vel[b + 2u] + hdtm * force[b + 2u];
}
