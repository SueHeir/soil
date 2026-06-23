//! vec3 — minimal 3-vector helpers over the engine's native `[f64; 3]` storage.
//!
//! These exist to make contact/integration kernels readable WITHOUT changing the
//! SoA data layout (`Vec<[f64; 3]>`) that neighbor traversal, MPI packing, and GPU
//! upload depend on. Every fn is `#[inline]`, `Copy`-only, and allocation-free, so
//! it lowers to the same code as the equivalent hand-written component math.
//!
//! Deliberately NOT a newtype with operator overloads: a `struct Vec3([f64; 3])`
//! would pull callers back toward array-of-structs thinking and reopen `Pod`/
//! alignment questions at the GPU/MPI boundary. Free functions on the bare array
//! stay neutral to the storage layout.
//!
//! NOTE (precision): on the `gpu` branch this becomes `type Vec3 = [Real; 3]` with
//! the math carried in `Accum`. On `master` storage is plain `f64`, so this module
//! is f64 and trivially swappable once the `Real`/`Accum` abstraction lands here.

/// A 3-vector in the engine's native storage form.
pub type Vec3 = [f64; 3];

#[inline(always)]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline(always)]
pub fn neg(a: Vec3) -> Vec3 {
    [-a[0], -a[1], -a[2]]
}

#[inline(always)]
pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// `a + s * b` — fused scale-add (e.g. the integrator's `v += (dt/m) * f` step).
#[inline(always)]
pub fn add_scaled(a: Vec3, b: Vec3, s: f64) -> Vec3 {
    [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]]
}

#[inline(always)]
pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
pub fn norm_sq(a: Vec3) -> f64 {
    dot(a, a)
}

#[inline(always)]
pub fn norm(a: Vec3) -> f64 {
    norm_sq(a).sqrt()
}

/// Unit vector plus its magnitude, in one pass. Returns `([0, 0, 0], 0.0)` for a
/// zero vector so callers can branch on the magnitude instead of propagating NaN.
#[inline(always)]
pub fn normalize(a: Vec3) -> (Vec3, f64) {
    let n = norm(a);
    if n > 0.0 {
        (scale(a, 1.0 / n), n)
    } else {
        ([0.0; 3], 0.0)
    }
}

// --- SoA scatter helpers: accumulate into a `force[i]` / `torque[i]` slot ---

#[inline(always)]
pub fn add_assign(dst: &mut Vec3, a: Vec3) {
    dst[0] += a[0];
    dst[1] += a[1];
    dst[2] += a[2];
}

#[inline(always)]
pub fn sub_assign(dst: &mut Vec3, a: Vec3) {
    dst[0] -= a[0];
    dst[1] -= a[1];
    dst[2] -= a[2];
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[inline]
    fn close(a: Vec3, b: Vec3) -> bool {
        (0..3).all(|k| (a[k] - b[k]).abs() < EPS)
    }

    #[test]
    fn cross_is_right_handed() {
        let x = [1.0, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        assert!(close(cross(x, y), [0.0, 0.0, 1.0]));
        assert!(close(cross(y, x), [0.0, 0.0, -1.0]));
    }

    #[test]
    fn cross_is_perpendicular_to_inputs() {
        let a = [0.3, -1.7, 2.1];
        let b = [4.0, 0.5, -2.2];
        let c = cross(a, b);
        assert!(dot(c, a).abs() < EPS);
        assert!(dot(c, b).abs() < EPS);
    }

    #[test]
    fn dot_and_norm_agree() {
        let a = [3.0, 4.0, 0.0];
        assert!((norm(a) - 5.0).abs() < EPS);
        assert!((norm_sq(a) - 25.0).abs() < EPS);
    }

    #[test]
    fn add_scaled_matches_manual() {
        let v = [1.0, 2.0, 3.0];
        let f = [10.0, 20.0, 30.0];
        let s = 0.5;
        assert!(close(add_scaled(v, f, s), [6.0, 12.0, 18.0]));
    }

    #[test]
    fn normalize_handles_zero() {
        let (u, n) = normalize([0.0, 0.0, 0.0]);
        assert_eq!(n, 0.0);
        assert!(close(u, [0.0, 0.0, 0.0]));
    }

    #[test]
    fn normalize_unit_length() {
        let (u, n) = normalize([0.0, 3.0, 4.0]);
        assert!((n - 5.0).abs() < EPS);
        assert!((norm(u) - 1.0).abs() < EPS);
    }

    #[test]
    fn scatter_helpers_accumulate() {
        let mut f: Vec3 = [1.0, 1.0, 1.0];
        add_assign(&mut f, [2.0, 0.0, -1.0]);
        sub_assign(&mut f, [0.5, 0.5, 0.5]);
        assert!(close(f, [2.5, 0.5, -0.5]));
    }
}
