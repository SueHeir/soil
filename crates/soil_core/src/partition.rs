//! Read-only ownership queries for Cartesian particle decompositions.

use crate::Domain;

/// Global directory mapping a point to its SOIL role-rank owner.
///
/// This mirrors SOIL's orthogonal Cartesian decomposition without performing
/// communication. Coupling packages use it (through [`grass_mpi::OwnerMap`])
/// to address particle records; SOIL remains the owner of the spatial policy.
///
/// Build it with [`from_domain`](Self::from_domain) to follow SOIL's live,
/// load-balanced cuts. [`new`](Self::new) describes even cuts only.
#[derive(Debug, Clone, PartialEq)]
pub struct ParticlePartitionDirectory {
    bounds_low: [f64; 3],
    bounds_high: [f64; 3],
    decomposition: [i32; 3],
    /// Cut planes per axis in fractional coordinates, `p + 1` values on `[0, 1]`.
    lamda_planes: [Vec<f64>; 3],
}

impl ParticlePartitionDirectory {
    /// Construct a directory for global bounds and `[px, py, pz]` ranks with
    /// even cuts.
    pub fn new(
        bounds_low: [f64; 3],
        bounds_high: [f64; 3],
        decomposition: [i32; 3],
    ) -> Result<Self, &'static str> {
        if decomposition.iter().any(|&n| n <= 0) {
            return Err("particle decomposition entries must be positive");
        }
        let planes = std::array::from_fn(|d| crate::domain::uniform_lamda_planes(decomposition[d]));
        Self::with_planes(bounds_low, bounds_high, planes)
    }

    /// Construct a directory from explicit cut planes per axis, in fractional
    /// coordinates (`p + 1` strictly increasing values from `0.0` to `1.0`).
    // Negated comparisons on purpose: they also reject NaN.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn with_planes(
        bounds_low: [f64; 3],
        bounds_high: [f64; 3],
        lamda_planes: [Vec<f64>; 3],
    ) -> Result<Self, &'static str> {
        if (0..3).any(|axis| !(bounds_high[axis] > bounds_low[axis])) {
            return Err("particle partition bounds must have positive extent");
        }
        for p in &lamda_planes {
            if p.len() < 2
                || p[0] != 0.0
                || p[p.len() - 1] != 1.0
                || p.windows(2).any(|w| !(w[1] > w[0]))
            {
                return Err("partition planes must increase strictly from 0.0 to 1.0");
            }
        }
        Ok(Self {
            bounds_low,
            bounds_high,
            decomposition: std::array::from_fn(|d| lamda_planes[d].len() as i32 - 1),
            lamda_planes,
        })
    }

    /// The directory for SOIL's current cuts, including any load balancing.
    /// Needs no communication: every rank already holds every rank's planes.
    /// Orthogonal boxes only; returns an error for a triclinic box.
    pub fn from_domain(domain: &Domain) -> Result<Self, &'static str> {
        if domain.triclinic {
            return Err("particle partition directory supports orthogonal boxes only");
        }
        Self::with_planes(
            domain.boundaries_low,
            domain.boundaries_high,
            domain.lamda_planes.clone(),
        )
    }

    /// Return the role-local rank owning `point`, or `None` outside the domain.
    /// Internal boundaries are half-open; the global upper boundary belongs to
    /// the final partition on that axis.
    pub fn owner_rank(&self, point: [f64; 3]) -> Option<i32> {
        let mut cell = [0_i32; 3];
        for axis in 0..3 {
            if !(point[axis] >= self.bounds_low[axis] && point[axis] <= self.bounds_high[axis]) {
                return None;
            }
            let lamda = (point[axis] - self.bounds_low[axis])
                / (self.bounds_high[axis] - self.bounds_low[axis]);
            let planes = &self.lamda_planes[axis];
            let inner = &planes[1..planes.len() - 1];
            cell[axis] = inner.partition_point(|&plane| plane <= lamda) as i32;
        }
        Some((cell[0] * self.decomposition[1] + cell[1]) * self.decomposition[2] + cell[2])
    }

    /// Processor-grid shape represented by this directory.
    pub fn decomposition(&self) -> [i32; 3] {
        self.decomposition
    }

    /// Cut planes along `axis` in fractional coordinates.
    pub fn lamda_planes(&self, axis: usize) -> &[f64] {
        &self.lamda_planes[axis]
    }

    /// A fingerprint of the cuts and bounds: equal on every rank for the same
    /// decomposition, and different once SOIL re-cuts or the box moves.
    pub fn version(&self) -> u64 {
        // FNV-1a over the bit patterns, so it is identical on every rank.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |x: f64| {
            for b in x.to_bits().to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        };
        for axis in 0..3 {
            eat(self.bounds_low[axis]);
            eat(self.bounds_high[axis]);
            self.lamda_planes[axis].iter().for_each(|&p| eat(p));
        }
        h
    }
}

impl grass_mpi::OwnerMap for ParticlePartitionDirectory {
    fn owner_rank(&self, point: [f64; 3]) -> Option<i32> {
        ParticlePartitionDirectory::owner_rank(self, point)
    }

    fn version(&self) -> u64 {
        ParticlePartitionDirectory::version(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{rebalance_shift, Atom};
    use grass_mpi::{CommBackend, OwnerMap, SingleProcessComm};

    #[test]
    fn maps_full_three_dimensional_decomposition() {
        let directory = ParticlePartitionDirectory::new([0.0; 3], [1.0; 3], [2, 2, 2]).unwrap();
        assert_eq!(directory.owner_rank([0.1, 0.1, 0.1]), Some(0));
        assert_eq!(directory.owner_rank([0.1, 0.6, 0.6]), Some(3));
        assert_eq!(directory.owner_rank([0.6, 0.1, 0.6]), Some(5));
        assert_eq!(directory.owner_rank([1.0, 1.0, 1.0]), Some(7));
        assert_eq!(directory.owner_rank([1.01, 0.5, 0.5]), None);
        assert_eq!(directory.owner_rank([f64::NAN, 0.5, 0.5]), None);
    }

    #[test]
    fn even_cuts_match_the_previous_closed_form() {
        // The old directory computed floor(fraction * p); the plane search
        // must give the same rank everywhere, including on plane values.
        let d =
            ParticlePartitionDirectory::new([-2.0, 0.0, 1.0], [3.0, 1.0, 4.0], [3, 1, 5]).unwrap();
        for i in 0..=200 {
            let t = i as f64 / 200.0;
            let p = [-2.0 + 5.0 * t, 0.5, 1.0 + 3.0 * t];
            let old = |axis: usize, lo: f64, hi: f64, n: i32| {
                (((p[axis] - lo) / (hi - lo) * n as f64).floor() as i32).min(n - 1)
            };
            let want = old(0, -2.0, 3.0, 3) * 5 + old(2, 1.0, 4.0, 5);
            assert_eq!(d.owner_rank(p), Some(want), "point {p:?}");
        }
    }

    #[test]
    fn uneven_planes_route_to_the_balanced_owner() {
        let d = ParticlePartitionDirectory::with_planes(
            [0.0; 3],
            [10.0; 3],
            [vec![0.0, 0.2, 0.25, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]],
        )
        .unwrap();
        assert_eq!(d.decomposition(), [3, 1, 1]);
        assert_eq!(d.owner_rank([1.0, 5.0, 5.0]), Some(0));
        assert_eq!(d.owner_rank([2.0, 5.0, 5.0]), Some(1));
        assert_eq!(d.owner_rank([2.4, 5.0, 5.0]), Some(1));
        assert_eq!(d.owner_rank([2.5, 5.0, 5.0]), Some(2));
        assert!(ParticlePartitionDirectory::with_planes(
            [0.0; 3],
            [1.0; 3],
            [vec![0.0, 0.5, 0.5, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]],
        )
        .is_err());
    }

    /// The x = 0 rank of a 4-rank x split, run serially: the all-reduce adds
    /// the plane values ranks 1..3 would contribute, so `rebalance_shift` runs
    /// its real plane update.
    struct FirstOfFourInX(SingleProcessComm);
    impl CommBackend for FirstOfFourInX {
        fn rank(&self) -> i32 {
            0
        }
        fn size(&self) -> i32 {
            4
        }
        fn processor_decomposition(&self) -> [i32; 3] {
            [4, 1, 1]
        }
        fn processor_position(&self) -> [i32; 3] {
            [0, 0, 0]
        }
        fn set_processor_grid(&mut self, _: [i32; 3], _: [i32; 3]) {}
        fn all_reduce_sum_f64(&self, v: f64) -> f64 {
            v
        }
        fn all_reduce_min_f64(&self, v: f64) -> f64 {
            v
        }
        fn all_reduce_sum_f64_slice(&self, buf: &mut [f64]) {
            // The buffer ends with the 5 x-planes; this rank wrote plane 0.
            let n = buf.len();
            for r in 1..=4 {
                buf[n - 5 + r] += r as f64 / 4.0;
            }
        }
        fn barrier(&self) {}
        fn send_f64(&self, d: i32, b: &[f64]) {
            self.0.send_f64(d, b)
        }
        fn recv_f64(&self, s: i32) -> Vec<f64> {
            self.0.recv_f64(s)
        }
        fn recv_f64_any(&self) -> Vec<f64> {
            self.0.recv_f64_any()
        }
        fn sendrecv_f64(&self, d: i32, b: &[f64], s: i32) -> Vec<f64> {
            self.0.sendrecv_f64(d, b, s)
        }
        fn sendrecv_f64_into(&self, d: i32, b: &[f64], s: i32, r: &mut [f64]) {
            self.0.sendrecv_f64_into(d, b, s, r)
        }
        fn sendrecv_batch_f64_into(&self, ops: &mut [grass_mpi::SendRecvOp<'_>]) {
            self.0.sendrecv_batch_f64_into(ops)
        }
    }

    #[test]
    fn directory_follows_load_balanced_cuts() {
        let comm = FirstOfFourInX(SingleProcessComm::new());
        let mut domain = Domain::new();
        domain.boundaries_high = [8.0, 1.0, 1.0];
        domain.boundary_type = [crate::BoundaryType::Fixed; 3];
        domain.lamda_planes[0] = crate::domain::uniform_lamda_planes(4);
        domain.sub_lamda_high[0] = 0.25;
        domain.sub_domain_high[0] = 2.0;
        domain.ghost_cutoff = 0.01;
        domain.update_derived();
        let before = ParticlePartitionDirectory::from_domain(&domain).unwrap();
        assert_eq!(before.owner_rank([1.9, 0.5, 0.5]), Some(0));

        // Every atom sits near x = 1: the balanced cuts crowd toward it.
        let mut atoms = Atom::new();
        for i in 0..400 {
            atoms.push_test_atom(i, [0.5 + 0.0025 * i as f64, 0.5, 0.5], 0.01, 1.0);
        }
        atoms.nlocal = 400;
        atoms.natoms = 400;
        let moved = rebalance_shift(&atoms, &mut domain, &comm, 64, [1, 1, 1]);
        assert!(moved, "rebalance must move planes");

        let after = ParticlePartitionDirectory::from_domain(&domain).unwrap();
        assert_ne!(before.version(), after.version());
        assert_eq!(OwnerMap::version(&after), after.version());
        // The directory agrees with this rank's own new slab.
        let x = domain.sub_domain_high[0] - 1e-9;
        assert_eq!(after.owner_rank([x, 0.5, 0.5]), Some(0));
        assert_eq!(
            after.owner_rank([domain.sub_domain_high[0], 0.5, 0.5]),
            Some(1)
        );
        assert_eq!(after.lamda_planes(0)[1], domain.sub_lamda_high[0]);
    }

    #[test]
    fn fixed_cuts_set_exact_slabs_and_route_like_the_cuts() {
        let cuts = crate::FixedCuts {
            planes: [vec![0.0, 0.8, 2.0], vec![-1.0, 0.0, 1.0], vec![0.5, 1.5]],
        };
        assert_eq!(cuts.decomposition(), [2, 2, 1]);
        cuts.validate([0.0, -1.0, 0.5], [2.0, 1.0, 1.5], 4).unwrap();
        assert!(cuts.validate([0.0, -1.0, 0.5], [2.0, 1.0, 1.5], 6).is_err());
        assert!(cuts.validate([0.9, -1.0, 0.5], [2.0, 1.0, 1.5], 4).is_err());

        let mut domain = Domain::new();
        domain.boundaries_low = [0.0, -1.0, 0.5];
        domain.boundaries_high = [2.0, 1.0, 1.5];
        domain.update_derived();
        crate::apply_fixed_cuts(&mut domain, &cuts, [1, 0, 0]);
        assert_eq!(domain.sub_domain_low, [0.8, -1.0, 0.5]);
        assert_eq!(domain.sub_domain_high, [2.0, 0.0, 1.5]);
        let d = ParticlePartitionDirectory::from_domain(&domain).unwrap();
        assert_eq!(d.decomposition(), [2, 2, 1]);
        assert_eq!(d.owner_rank([0.79, 0.5, 1.0]), Some(1));
        assert_eq!(d.owner_rank([0.81, -0.5, 1.0]), Some(2));
    }
}
