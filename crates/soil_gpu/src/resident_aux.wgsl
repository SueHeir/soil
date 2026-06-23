// Auxiliary integrated DOF half-kick (substrate; method-agnostic).
//
// A velocity-like quantity `aux_state` driven by `aux_rate` with per-atom inverse
// coefficient `aux_inv_coeff`, integrated with the SAME half-kick as velocity but
// WITHOUT a position drift:  state += 0.5*dt*inv_coeff*rate.  Run once after the
// first integration half and once after the second, bracketing the force/rate
// evaluation — exactly the velocity-Verlet treatment of velocity. The consumer's
// Force hook owns `aux_rate` (overwrites it each step, i-centric). Generic: a
// granular code maps state=angular velocity, rate=torque, inv_coeff=1/inertia; a
// thermostat or other extra DOF maps its own triple. Soil stays method-agnostic.

struct Params {
    n: u32,
    dt: f32,
    gx: f32,
    gy: f32,
    gz: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
};

@group(0) @binding(0) var<storage, read_write> aux_state: array<f32>;
@group(0) @binding(1) var<storage, read>       aux_rate: array<f32>;
@group(0) @binding(2) var<storage, read>       aux_inv_coeff: array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn aux_kick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let b = 3u * i;
    let h = 0.5 * params.dt * aux_inv_coeff[i];
    aux_state[b]      = aux_state[b]      + h * aux_rate[b];
    aux_state[b + 1u] = aux_state[b + 1u] + h * aux_rate[b + 1u];
    aux_state[b + 2u] = aux_state[b + 2u] + h * aux_rate[b + 2u];
}
