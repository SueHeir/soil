//! Minimal Barnes-Hut gravity demonstrator on soil's ORBIT spatial tree.
//!
//! This is intentionally an example, not substrate API: gravity, masses,
//! timestepping, and validation belong to the consumer. `soil_core` supplies only
//! the agnostic [`Octree`] hierarchy and monopole aggregation.
//!
//! Validation uses the analytic two-body circular Kepler orbit with `G = 1`,
//! total mass `M = 1`, separation `a = 1`, and period `T = 2*pi`. A conservative
//! opening angle (`theta = 0.5`) keeps the two bodies in the direct near field and
//! must conserve period and energy. The negative control (`theta = 4.0`) accepts
//! the root monopole for each target; that aggregate includes the target's own
//! mass and produces the expected energy drift.
//!
//! Run: `cargo run --release --example orbit_nbody_gravity`

use soil_core::Octree;

const G: f64 = 1.0;
const PERIOD: f64 = std::f64::consts::TAU;

#[derive(Clone, Copy, Debug)]
struct Body {
    pos: [f64; 3],
    vel: [f64; 3],
    mass: f64,
}

#[derive(Clone, Copy, Debug)]
struct RunResult {
    measured_period: f64,
    period_rel_err: f64,
    max_energy_rel_drift: f64,
    final_energy_rel_drift: f64,
}

fn circular_two_body() -> Vec<Body> {
    vec![
        Body {
            pos: [0.5, 0.0, 0.0],
            vel: [0.0, 0.5, 0.0],
            mass: 0.5,
        },
        Body {
            pos: [-0.5, 0.0, 0.0],
            vel: [0.0, -0.5, 0.0],
            mass: 0.5,
        },
    ]
}

fn positions(bodies: &[Body]) -> Vec<[f64; 3]> {
    bodies.iter().map(|b| b.pos).collect()
}

fn masses(bodies: &[Body]) -> Vec<f64> {
    bodies.iter().map(|b| b.mass).collect()
}

fn dist_vec(a: [f64; 3], b: [f64; 3]) -> ([f64; 3], f64) {
    let r = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let d = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    (r, d)
}

fn accelerations(bodies: &[Body], theta: f64) -> Vec<[f64; 3]> {
    let pos = positions(bodies);
    let mass = masses(bodies);
    let tree = Octree::build(&pos, 1, 32);
    let monopoles = tree.monopoles(&pos, &mass);
    let mut acc = vec![[0.0; 3]; bodies.len()];
    let (mut near, mut far) = (Vec::new(), Vec::new());

    for i in 0..bodies.len() {
        tree.far_field_partition(pos[i], theta, &mut near, &mut far);
        for &j in &near {
            let j = j as usize;
            if j == i {
                continue;
            }
            let (r, d) = dist_vec(pos[i], pos[j]);
            let scale = G * mass[j] / (d * d * d);
            for k in 0..3 {
                acc[i][k] += scale * r[k];
            }
        }
        for &node in &far {
            let m = monopoles[node as usize];
            let (r, d) = dist_vec(pos[i], m.center);
            let scale = G * m.total_weight / (d * d * d);
            for k in 0..3 {
                acc[i][k] += scale * r[k];
            }
        }
    }

    acc
}

fn physical_energy(bodies: &[Body]) -> f64 {
    let kinetic: f64 = bodies
        .iter()
        .map(|b| 0.5 * b.mass * (b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1] + b.vel[2] * b.vel[2]))
        .sum();
    let mut potential = 0.0;
    for i in 0..bodies.len() {
        for j in i + 1..bodies.len() {
            let (_, d) = dist_vec(bodies[i].pos, bodies[j].pos);
            potential -= G * bodies[i].mass * bodies[j].mass / d;
        }
    }
    kinetic + potential
}

fn integrate(theta: f64, dt: f64, periods: f64) -> RunResult {
    let mut bodies = circular_two_body();
    let initial_energy = physical_energy(&bodies);
    let steps = (periods * PERIOD / dt).round() as usize;
    let mut acc = accelerations(&bodies, theta);
    let mut max_energy_rel_drift = 0.0f64;
    let mut crossings = Vec::new();
    let mut prev_y = bodies[0].pos[1];
    let mut prev_x = bodies[0].pos[0];
    let mut prev_t = 0.0;

    for step in 1..=steps {
        for (body, a) in bodies.iter_mut().zip(&acc) {
            for k in 0..3 {
                body.vel[k] += 0.5 * dt * a[k];
                body.pos[k] += dt * body.vel[k];
            }
        }

        acc = accelerations(&bodies, theta);

        for (body, a) in bodies.iter_mut().zip(&acc) {
            for k in 0..3 {
                body.vel[k] += 0.5 * dt * a[k];
            }
        }

        let t = step as f64 * dt;
        let y = bodies[0].pos[1];
        let x = bodies[0].pos[0];
        if prev_y < 0.0 && y >= 0.0 && x > 0.0 {
            let frac = -prev_y / (y - prev_y);
            let cross_t = prev_t + frac * (t - prev_t);
            let cross_x = prev_x + frac * (x - prev_x);
            if cross_x > 0.0 {
                crossings.push(cross_t);
            }
        }
        prev_y = y;
        prev_x = x;
        prev_t = t;

        let drift = ((physical_energy(&bodies) - initial_energy) / initial_energy).abs();
        max_energy_rel_drift = max_energy_rel_drift.max(drift);
    }

    let measured_period = if crossings.len() >= 2 {
        crossings.windows(2).map(|w| w[1] - w[0]).sum::<f64>() / (crossings.len() - 1) as f64
    } else if crossings.len() == 1 {
        crossings[0]
    } else {
        f64::INFINITY
    };
    let final_energy_rel_drift =
        ((physical_energy(&bodies) - initial_energy) / initial_energy).abs();

    RunResult {
        measured_period,
        period_rel_err: ((measured_period - PERIOD) / PERIOD).abs(),
        max_energy_rel_drift,
        final_energy_rel_drift,
    }
}

fn main() {
    println!("== soil ORBIT Barnes-Hut gravity demonstrator ==");
    println!("analytic two-body circular orbit: period = 2*pi = {PERIOD:.12}\n");

    let dt = PERIOD / 4000.0;
    let good = integrate(0.5, dt, 4.0);
    let bad = integrate(4.0, dt, 1.0);

    println!("[1] Barnes-Hut on Octree, conservative theta = 0.5");
    println!(
        "    measured period        = {:.12} (rel err {:.3e})",
        good.measured_period, good.period_rel_err
    );
    println!(
        "    max energy rel drift   = {:.3e}; final drift = {:.3e}",
        good.max_energy_rel_drift, good.final_energy_rel_drift
    );

    println!("\n[2] negative control, theta = 4.0 (root monopole accepted)");
    println!(
        "    measured period        = {:.12} (rel err {:.3e})",
        bad.measured_period, bad.period_rel_err
    );
    println!(
        "    max energy rel drift   = {:.3e}; final drift = {:.3e}",
        bad.max_energy_rel_drift, bad.final_energy_rel_drift
    );

    let period_ok = good.period_rel_err < 1.0e-3;
    let energy_ok = good.max_energy_rel_drift < 1.0e-6;
    let negative_ok = bad.max_energy_rel_drift > 1.0e-2;

    println!(
        "\nchecks: period_ok={} energy_ok={} negative_control_ok={}",
        period_ok, energy_ok, negative_ok
    );

    if period_ok && energy_ok && negative_ok {
        println!("ALL CHECKS PASSED");
    } else {
        eprintln!("CHECK FAILED");
        std::process::exit(1);
    }
}
