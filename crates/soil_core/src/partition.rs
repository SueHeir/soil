//! Read-only ownership queries for Cartesian particle decompositions.

/// Global directory mapping a point to its SOIL role-rank owner.
///
/// This mirrors SOIL's orthogonal Cartesian decomposition without performing
/// communication. Coupling packages use it to address migrating particle
/// records; SOIL remains the owner of the spatial policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticlePartitionDirectory {
    bounds_low: [f64; 3],
    bounds_high: [f64; 3],
    decomposition: [i32; 3],
}

impl ParticlePartitionDirectory {
    /// Construct a directory for global bounds and `[px, py, pz]` ranks.
    pub fn new(
        bounds_low: [f64; 3],
        bounds_high: [f64; 3],
        decomposition: [i32; 3],
    ) -> Result<Self, &'static str> {
        if decomposition.iter().any(|&n| n <= 0) {
            return Err("particle decomposition entries must be positive");
        }
        if (0..3).any(|axis| bounds_high[axis] <= bounds_low[axis]) {
            return Err("particle partition bounds must have positive extent");
        }
        Ok(Self {
            bounds_low,
            bounds_high,
            decomposition,
        })
    }

    /// Return the role-local rank owning `point`, or `None` outside the domain.
    /// Internal boundaries are half-open; the global upper boundary belongs to
    /// the final partition on that axis.
    pub fn owner_rank(&self, point: [f64; 3]) -> Option<i32> {
        let mut cell = [0_i32; 3];
        for axis in 0..3 {
            if point[axis] < self.bounds_low[axis] || point[axis] > self.bounds_high[axis] {
                return None;
            }
            let fraction = (point[axis] - self.bounds_low[axis])
                / (self.bounds_high[axis] - self.bounds_low[axis]);
            cell[axis] = ((fraction * self.decomposition[axis] as f64).floor() as i32)
                .min(self.decomposition[axis] - 1);
        }
        Some((cell[0] * self.decomposition[1] + cell[1]) * self.decomposition[2] + cell[2])
    }

    /// Processor-grid shape represented by this directory.
    pub fn decomposition(&self) -> [i32; 3] {
        self.decomposition
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_full_three_dimensional_decomposition() {
        let directory = ParticlePartitionDirectory::new([0.0; 3], [1.0; 3], [2, 2, 2]).unwrap();
        assert_eq!(directory.owner_rank([0.1, 0.1, 0.1]), Some(0));
        assert_eq!(directory.owner_rank([0.1, 0.6, 0.6]), Some(3));
        assert_eq!(directory.owner_rank([0.6, 0.1, 0.6]), Some(5));
        assert_eq!(directory.owner_rank([1.0, 1.0, 1.0]), Some(7));
        assert_eq!(directory.owner_rank([1.01, 0.5, 0.5]), None);
    }
}
