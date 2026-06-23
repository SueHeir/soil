//! Planar boundary geometry (substrate; no physics).
//!
//! A boundary is a set of planes (point + outward unit normal); particles sit on
//! the +normal side. This is generic geometry — DEM walls, MD/peridynamics
//! boundaries all reuse it. The *response* to overlap (Hertz repulsion, friction,
//! reflection, …) is the consumer's job; this module only provides the geometry,
//! a packed device representation, and a signed-distance helper.

use bytemuck::{Pod, Zeroable};

/// A plane: a point on it and the outward unit normal (particles on +normal side).
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub point: [f32; 3],
    pub normal: [f32; 3],
}

impl Plane {
    pub fn new(point: [f32; 3], normal: [f32; 3]) -> Self {
        // normalize defensively
        let n = normal;
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(f32::MIN_POSITIVE);
        Plane { point, normal: [n[0] / len, n[1] / len, n[2] / len] }
    }

    /// Signed distance of `pos` from the plane (+ on the normal side).
    pub fn signed_distance(&self, pos: [f32; 3]) -> f32 {
        (pos[0] - self.point[0]) * self.normal[0]
            + (pos[1] - self.point[1]) * self.normal[1]
            + (pos[2] - self.point[2]) * self.normal[2]
    }

    /// Overlap of a sphere of radius `r` centred at `pos` (>0 when penetrating).
    pub fn overlap(&self, pos: [f32; 3], r: f32) -> f32 {
        r - self.signed_distance(pos)
    }
}

/// Device-layout plane (std140-friendly: vec4 point, vec4 normal = 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuPlane {
    pub point: [f32; 4],
    pub normal: [f32; 4],
}

/// A set of planar boundaries; packs to a device buffer of [`GpuPlane`].
#[derive(Clone, Debug, Default)]
pub struct Boundary {
    pub planes: Vec<Plane>,
}

impl Boundary {
    pub fn new() -> Self { Self { planes: Vec::new() } }
    pub fn push(&mut self, plane: Plane) { self.planes.push(plane); }
    pub fn len(&self) -> usize { self.planes.len() }
    pub fn is_empty(&self) -> bool { self.planes.is_empty() }

    /// Pack to the device representation (one [`GpuPlane`] per plane).
    pub fn to_gpu(&self) -> Vec<GpuPlane> {
        self.planes
            .iter()
            .map(|p| GpuPlane {
                point: [p.point[0], p.point[1], p.point[2], 0.0],
                normal: [p.normal[0], p.normal[1], p.normal[2], 0.0],
            })
            .collect()
    }
}

/// WGSL helpers a consumer concatenates into its wall/boundary kernel. The
/// consumer declares its own `array<BoundaryPlane>` binding + a count and loops,
/// calling `boundary_signed_distance` / `boundary_overlap`; it then applies its
/// own response (Hertz, friction, reflect, …).
pub const BOUNDARY_WGSL: &str = r#"
struct BoundaryPlane { point: vec4<f32>, normal: vec4<f32> };

fn boundary_signed_distance(pl: BoundaryPlane, pos: vec3<f32>) -> f32 {
    return dot(pos - pl.point.xyz, pl.normal.xyz);
}
fn boundary_overlap(pl: BoundaryPlane, pos: vec3<f32>, r: f32) -> f32 {
    return r - boundary_signed_distance(pl, pos);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_signed_distance_and_overlap() {
        // Floor at z=0, normal +z.
        let floor = Plane::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!((floor.signed_distance([1.0, 2.0, 0.5]) - 0.5).abs() < 1e-6);
        assert!((floor.signed_distance([0.0, 0.0, -0.3]) + 0.3).abs() < 1e-6);
        // sphere r=0.5 centred at z=0.2 penetrates the floor by 0.3.
        assert!((floor.overlap([0.0, 0.0, 0.2], 0.5) - 0.3).abs() < 1e-6);
        // resting just touching: overlap ~0.
        assert!(floor.overlap([0.0, 0.0, 0.5], 0.5).abs() < 1e-6);

        // Normalization: a non-unit normal still gives true distance.
        let tilted = Plane::new([0.0, 0.0, 0.0], [0.0, 0.0, 2.0]);
        assert!((tilted.signed_distance([0.0, 0.0, 1.0]) - 1.0).abs() < 1e-6);

        // Device packing round-trips the components.
        let mut b = Boundary::new();
        b.push(floor);
        let g = b.to_gpu();
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].normal[2], 1.0);
    }
}
