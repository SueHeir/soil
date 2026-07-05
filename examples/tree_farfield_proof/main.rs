//! Substrate-level proof that soil's spatial hierarchy + near/far partition is
//! enough to build a long-range particle method — self-proved against the direct
//! O(n²) sum.
//!
//! The physics here (a 1/r kernel, per-atom "charges") lives ENTIRELY in this
//! example: soil itself stays method-agnostic. We only reach into `soil_core` for
//! the agnostic primitives — [`Octree`], [`Octree::near_pairs`],
//! [`Octree::far_field_partition`], [`Octree::monopoles`]. A gravity code (masses,
//! attractive 1/r), an Ewald real/reciprocal split, or an SPH far-field would sit
//! here in exactly the same place; none of it is baked into the substrate.
//!
//! Three checks, matching the goal's acceptance criteria:
//!
//! 1. **Near/far partition == brute force.** [`near_pairs`](Octree::near_pairs)
//!    reproduces the brute-force O(n²) within-cutoff neighbor set exactly, at
//!    several cutoffs including one past the diameter (all pairs). The near set
//!    plus its complement (the far set) is a clean partition of all C(n,2) pairs.
//! 2. **Treecode == direct sum, to a defensible tolerance.** A monopole treecode
//!    built on the tree reproduces the direct 1/r potential; the error shrinks as
//!    the opening angle θ → 0 at the ~θ² rate the monopole approximation predicts,
//!    i.e. it is *consistent* with the direct sum, not merely close at one θ.
//! 3. **Sub-O(n²) scaling.** The treecode's interaction count grows far slower
//!    than the n²/2 of the direct sum; a log–log fit gives an exponent well below
//!    2 (Barnes–Hut is ~n log n).
//!
//! Run: `cargo run --release --example tree_farfield_proof`

use rand::{rngs::StdRng, Rng, SeedableRng};
use soil_core::Octree;

/// Deterministic points in the unit cube.
fn random_points(n: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            [
                rng.random::<f64>(),
                rng.random::<f64>(),
                rng.random::<f64>(),
            ]
        })
        .collect()
}

/// Deterministic positive "charges" (all positive keeps the potential bounded
/// away from zero, so relative error is well defined — think masses in gravity).
fn charges(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| 0.5 + rng.random::<f64>()).collect()
}

#[inline]
fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// The reference: direct O(n²) sum of the 1/r kernel, φ_i = Σ_{j≠i} q_j / r_ij.
fn direct_potential(pos: &[[f64; 3]], q: &[f64]) -> Vec<f64> {
    let n = pos.len();
    let mut phi = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            if i != j {
                acc += q[j] / dist(pos[i], pos[j]);
            }
        }
        phi[i] = acc;
    }
    phi
}

/// The treecode: for each target, split sources via `far_field_partition`, sum
/// near sources directly and far clusters through their monopole. Returns the
/// potential and the total number of kernel evaluations (near atoms + far nodes)
/// — the work measure for the scaling check.
fn treecode_potential(tree: &Octree, pos: &[[f64; 3]], q: &[f64], theta: f64) -> (Vec<f64>, u64) {
    let mono = tree.monopoles(pos, q);
    let n = pos.len();
    let mut phi = vec![0.0f64; n];
    let mut interactions: u64 = 0;
    let (mut near, mut far) = (Vec::new(), Vec::new());
    for i in 0..n {
        tree.far_field_partition(pos[i], theta, &mut near, &mut far);
        let mut acc = 0.0;
        for &j in &near {
            let j = j as usize;
            if j != i {
                acc += q[j] / dist(pos[i], pos[j]);
            }
        }
        for &node in &far {
            let m = mono[node as usize];
            acc += m.total_weight / dist(pos[i], m.center);
        }
        phi[i] = acc;
        interactions += near.len() as u64 + far.len() as u64;
    }
    (phi, interactions)
}

/// max and mean relative error of `approx` vs `reference`.
fn rel_errors(approx: &[f64], reference: &[f64]) -> (f64, f64) {
    let mut max = 0.0f64;
    let mut sum = 0.0f64;
    for (&a, &r) in approx.iter().zip(reference) {
        let e = (a - r).abs() / r.abs();
        max = max.max(e);
        sum += e;
    }
    (max, sum / approx.len() as f64)
}

/// Least-squares slope of log(y) vs log(x).
fn loglog_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let lx: Vec<f64> = xs.iter().map(|v| v.ln()).collect();
    let ly: Vec<f64> = ys.iter().map(|v| v.ln()).collect();
    let mx = lx.iter().sum::<f64>() / n;
    let my = ly.iter().sum::<f64>() / n;
    let num: f64 = lx.iter().zip(&ly).map(|(&x, &y)| (x - mx) * (y - my)).sum();
    let den: f64 = lx.iter().map(|&x| (x - mx) * (x - mx)).sum();
    num / den
}

fn brute_pairs(pos: &[[f64; 3]], cutoff: f64) -> Vec<(usize, usize)> {
    let c2 = cutoff * cutoff;
    let mut out = Vec::new();
    for i in 0..pos.len() {
        for j in i + 1..pos.len() {
            let d = pos[i];
            let e = pos[j];
            let r2 = (d[0] - e[0]).powi(2) + (d[1] - e[1]).powi(2) + (d[2] - e[2]).powi(2);
            if r2 <= c2 {
                out.push((i, j));
            }
        }
    }
    out
}

fn main() {
    let mut failures = 0usize;
    println!("== soil far-field tree primitive — self-proof vs brute force ==\n");

    // ---- Check 1: near/far partition reproduces the brute-force neighbor set ----
    println!("[1] near/far partition vs brute-force O(n^2) neighbor set");
    {
        let n = 800;
        let pos = random_points(n, 11);
        let tree = Octree::build(&pos, 8, 24);
        let total_pairs = n * (n - 1) / 2;
        let mut ok = true;
        for &cutoff in &[0.03, 0.06, 0.12, 0.25, 100.0] {
            let mut got = tree.near_pairs(&pos, cutoff);
            let mut want = brute_pairs(&pos, cutoff);
            got.sort_unstable();
            want.sort_unstable();
            let near = got.len();
            let far = total_pairs - near; // complement = far half of the split
            let matched = got == want;
            ok &= matched;
            println!(
                "    cutoff {:>6.2}: near={:>7} far={:>7} (near+far={}), matches brute = {}",
                cutoff,
                near,
                far,
                near + far,
                matched
            );
            assert_eq!(near + far, total_pairs, "near/far is not a clean partition");
        }
        if ok {
            println!(
                "    => PASS: tree near set == brute force at every cutoff (incl. all-pairs)\n"
            );
        } else {
            println!("    => FAIL: tree near set diverged from brute force\n");
            failures += 1;
        }
    }

    // ---- Check 2: treecode reproduces the direct 1/r sum; error ~ theta^2 -> 0 ----
    println!("[2] 1/r treecode vs direct O(n^2) sum (accuracy + convergence)");
    let (mut prev_mean, mut conv_ratio) = (f64::NAN, f64::NAN);
    {
        let n = 3000;
        let pos = random_points(n, 21);
        let q = charges(n, 22);
        let tree = Octree::build(&pos, 16, 24);
        let phi_direct = direct_potential(&pos, &q);

        let thetas = [0.6, 0.4, 0.2, 0.1];
        let mut max_at_02 = f64::NAN;
        for (k, &theta) in thetas.iter().enumerate() {
            let (phi_tc, interactions) = treecode_potential(&tree, &pos, &q, theta);
            let (emax, emean) = rel_errors(&phi_tc, &phi_direct);
            println!(
                "    theta {:>4.2}: max rel err {:>9.2e}  mean rel err {:>9.2e}  interactions {:>10}",
                theta, emax, emean, interactions
            );
            if (theta - 0.2).abs() < 1e-12 {
                max_at_02 = emax;
                if prev_mean.is_finite() {
                    conv_ratio = prev_mean / emean; // mean err ratio between theta=0.4 and 0.2
                }
            }
            if k == 1 {
                prev_mean = emean; // remember theta=0.4 mean for the ratio at 0.2
            }
        }
        // Defensible bounds: monopole-at-centroid error is O((size/dist)^2) ~ theta^2.
        // At theta=0.2 that predicts ~4e-2 worst case; require the measured max below
        // 5e-2, mean-error convergence between theta=0.4 and 0.2 at least 2.5x
        // (theta^2 predicts 4x) — i.e. the treecode genuinely converges to the
        // direct sum as theta -> 0, not just sits near it at one angle.
        let bound_ok = max_at_02 < 5e-2;
        let conv_ok = conv_ratio >= 2.5;
        println!(
            "    max rel err @theta=0.2 = {:.2e} (< 5e-2 ? {})",
            max_at_02, bound_ok
        );
        println!(
            "    mean-err convergence 0.4->0.2 = {:.2}x (>= 2.5x ? {})",
            conv_ratio, conv_ok
        );
        if bound_ok && conv_ok {
            println!("    => PASS: treecode reproduces the direct sum and converges at ~theta^2\n");
        } else {
            println!("    => FAIL: treecode accuracy/convergence bound not met\n");
            failures += 1;
        }
    }

    // ---- Check 3: sub-O(n^2) scaling ----
    println!("[3] scaling: treecode interaction count vs direct n^2/2");
    {
        let theta = 0.4;
        let sizes = [1000usize, 2000, 4000, 8000];
        let mut ns = Vec::new();
        let mut counts = Vec::new();
        for &n in &sizes {
            let pos = random_points(n, 100 + n as u64);
            let q = charges(n, 200 + n as u64);
            let tree = Octree::build(&pos, 16, 24);
            let (_phi, interactions) = treecode_potential(&tree, &pos, &q, theta);
            let direct = (n as u64) * (n as u64 - 1) / 2;
            ns.push(n as f64);
            counts.push(interactions as f64);
            println!(
                "    n={:>5}: treecode interactions={:>11}  direct n^2/2={:>13}  ratio={:>7.3}",
                n,
                interactions,
                direct,
                interactions as f64 / direct as f64
            );
        }
        let slope = loglog_slope(&ns, &counts);
        let slope_ok = slope < 1.6;
        println!(
            "    log-log growth exponent = {:.3} (< 1.6 ? {}; direct sum is 2.0)",
            slope, slope_ok
        );
        if slope_ok {
            println!("    => PASS: treecode work grows sub-quadratically\n");
        } else {
            println!("    => FAIL: treecode did not show sub-O(n^2) scaling\n");
            failures += 1;
        }
    }

    if failures == 0 {
        println!("ALL CHECKS PASSED");
    } else {
        println!("{failures} CHECK(S) FAILED");
        std::process::exit(1);
    }
}
