//! Hierarchical spatial decomposition (octree) and a near/far interaction
//! partition, for building **long-range** particle methods on the soil substrate.
//!
//! soil already ships a short-range primitive (the binned [`Neighbor`](crate::Neighbor)
//! list). Long-range methods — gravitational N-body, electrostatics (Ewald/PPPM),
//! implicit MD, global constraints — instead need a *spatial hierarchy* and a way
//! to split each particle's interactions into a **near** part (handled directly,
//! pairwise) and a **far** part (approximated by an aggregate of a whole cluster).
//! This module provides exactly that hierarchy and split, and nothing more.
//!
//! # What stays out (agnosticism contract)
//!
//! This is a *substrate* primitive: it is deliberately blind to the particle
//! method. It knows only positions (`[f64; 3]`) and — for the optional monopole
//! helper — caller-supplied non-negative *weights* (mass, charge magnitude,
//! volume, SPH weight, …). It contains **no force law, no potential, no kernel**.
//! A 1/r gravity/Coulomb kernel, an Ewald real/reciprocal split, a specific pair
//! potential — all of those are physics and belong in a tier *above* soil. The
//! litmus holds: an [`Octree`] + [`Octree::far_field_partition`] serves an MD,
//! SPH, peridynamics, or gravity code identically, because none of that physics
//! appears here.
//!
//! The proof that this primitive is sufficient to build a real far-field method
//! lives — kernel and all — in `examples/tree_farfield_proof.rs`, which stacks a
//! generic 1/r treecode on top and checks it against the direct O(n²) sum.
//!
//! # The two pieces
//!
//! - [`Octree::near_pairs`] — a cutoff neighbor query driven by the tree. With a
//!   cutoff at least as large as the point-set diameter it returns the *complete*
//!   brute-force O(n²) pair set; with a smaller cutoff it returns exactly the
//!   pairs within it. This is the "near" half of a near/far split (near = within
//!   cutoff, far = beyond), and it is verified pair-for-pair against brute force.
//! - [`Octree::far_field_partition`] — for one target point, a Barnes–Hut-style
//!   walk that partitions *every* source into either a `near` atom (returned by
//!   index, to be summed directly) or a `far` tree node (returned by node id, to
//!   be approximated by that node's aggregate). Together with the optional
//!   [`Octree::monopoles`] helper this is all a treecode needs.

/// A tree node's cubic cell and the atoms it contains.
///
/// The tree subdivides a root cube into octants; every node is therefore an
/// axis-aligned cube of side `2 * half_width` centred at `center`. `radius` is
/// the distance from the centre to a corner (`half_width * √3`) — a conservative
/// bound on how far any contained atom can be from the centre, used by the
/// multipole-acceptance test in [`Octree::far_field_partition`].
///
/// The atoms in a node occupy the contiguous slice
/// `ordered_atoms()[start .. start + count]` (see [`Octree::node_atoms`]). Because
/// children are laid out before their parent in construction order, an internal
/// node's slice is the concatenation of its children's slices — so *any* node,
/// internal or leaf, exposes its full atom membership as one slice.
#[derive(Clone, Debug)]
pub struct OctreeNode {
    /// Geometric centre of the cubic cell.
    pub center: [f64; 3],
    /// Half the cube's side length.
    pub half_width: f64,
    /// Distance from `center` to a cube corner (`half_width * √3`); the cell's
    /// bounding radius for the acceptance criterion.
    pub radius: f64,
    /// Start index into [`Octree::ordered_atoms`] of this node's atoms.
    pub start: u32,
    /// Number of atoms in (the whole subtree rooted at) this node.
    pub count: u32,
    /// Child node ids, one per octant; `-1` where an octant is empty. Octant bit
    /// layout: `bit0 = x>cx`, `bit1 = y>cy`, `bit2 = z>cz`.
    pub children: [i32; 8],
    /// True when this node was not subdivided (holds atoms directly).
    pub is_leaf: bool,
}

impl OctreeNode {
    /// Squared minimum distance between this node's cube and `other`'s cube
    /// (0 if they touch or overlap). Used to prune the dual-tree cutoff query.
    #[inline]
    fn box_min_dist_sq(&self, other: &OctreeNode) -> f64 {
        let mut s = 0.0;
        for d in 0..3 {
            let gap =
                (self.center[d] - other.center[d]).abs() - (self.half_width + other.half_width);
            if gap > 0.0 {
                s += gap * gap;
            }
        }
        s
    }
}

/// A node's monopole aggregate: total weight and the weight-averaged centre.
///
/// Produced by [`Octree::monopoles`]. This is pure geometric aggregation
/// (`Σ wᵢ` and `Σ wᵢ xᵢ / Σ wᵢ`) — the caller decides what the weight *means*
/// (mass for gravity, |charge| for electrostatics, volume for SPH). Expanding a
/// far cluster about its weighted centroid makes the monopole approximation exact
/// to first order (the dipole moment vanishes there), so the leading error is the
/// quadrupole term, `O((cell size / distance)²)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Monopole {
    /// Sum of the caller's weights over the node's atoms.
    pub total_weight: f64,
    /// Weight-averaged position of the node's atoms (falls back to the node's
    /// geometric centre if the total weight is zero).
    pub center: [f64; 3],
}

/// An octree over a set of points, plus the near/far interaction partition built
/// on it. See the [module docs](self) for the agnosticism contract.
///
/// Build with [`Octree::build`] (auto-sized root cube) or
/// [`Octree::build_in_box`] (explicit root cube). The point coordinates are *not*
/// retained — pass the same `positions` slice to the query methods.
pub struct Octree {
    nodes: Vec<OctreeNode>,
    ordered: Vec<u32>,
    root: u32,
    leaf_capacity: usize,
    max_depth: usize,
}

impl Octree {
    /// Build an octree over `positions`, sizing the root cube to tightly enclose
    /// them. `leaf_capacity` caps how many atoms a leaf holds before it splits;
    /// `max_depth` caps the tree height (a guard against unsplittable coincident
    /// points). See [`build_in_box`](Self::build_in_box) for the parameter
    /// meanings; this just derives the enclosing cube first.
    pub fn build(positions: &[[f64; 3]], leaf_capacity: usize, max_depth: usize) -> Octree {
        let (center, half) = enclosing_cube(positions);
        Octree::build_in_box(positions, center, half, leaf_capacity, max_depth)
    }

    /// Build an octree whose root cube is centred at `center` with half-side
    /// `half_width`. All `positions` must lie inside that cube.
    ///
    /// `leaf_capacity` (clamped to ≥ 1) is the maximum atom count in a leaf before
    /// it subdivides; `max_depth` (clamped to ≥ 1) bounds recursion so that a
    /// cluster of (near-)coincident points can't recurse forever — such a cluster
    /// simply lands in one over-capacity leaf at `max_depth`.
    pub fn build_in_box(
        positions: &[[f64; 3]],
        center: [f64; 3],
        half_width: f64,
        leaf_capacity: usize,
        max_depth: usize,
    ) -> Octree {
        let leaf_capacity = leaf_capacity.max(1);
        let max_depth = max_depth.max(1);
        let mut tree = Octree {
            nodes: Vec::new(),
            ordered: Vec::with_capacity(positions.len()),
            root: 0,
            leaf_capacity,
            max_depth,
        };
        // A degenerate (empty) point set yields a single empty leaf so callers can
        // still traverse a valid root.
        let half = if half_width > 0.0 { half_width } else { 1.0 };
        let indices: Vec<u32> = (0..positions.len() as u32).collect();
        tree.root = tree.build_node(positions, center, half, indices, 0);
        tree
    }

    /// Recursively build a node from the atom `indices` assigned to it; returns
    /// the new node's id. Children are pushed before the parent, so the parent's
    /// atom slice is the contiguous concatenation of its children's slices.
    fn build_node(
        &mut self,
        positions: &[[f64; 3]],
        center: [f64; 3],
        half: f64,
        indices: Vec<u32>,
        depth: usize,
    ) -> u32 {
        let start = self.ordered.len() as u32;

        // Leaf: at/under capacity, at the depth cap, or numerically unsplittable.
        if indices.len() <= self.leaf_capacity
            || depth >= self.max_depth
            || half <= f64::MIN_POSITIVE
        {
            let count = indices.len() as u32;
            self.ordered.extend_from_slice(&indices);
            let id = self.nodes.len() as u32;
            self.nodes.push(OctreeNode {
                center,
                half_width: half,
                radius: half * SQRT_3,
                start,
                count,
                children: [-1; 8],
                is_leaf: true,
            });
            return id;
        }

        // Internal: bucket atoms into the 8 octants of this cube.
        let mut buckets: [Vec<u32>; 8] = Default::default();
        for &idx in &indices {
            let p = positions[idx as usize];
            let octant = (usize::from(p[0] > center[0]))
                | (usize::from(p[1] > center[1]) << 1)
                | (usize::from(p[2] > center[2]) << 2);
            buckets[octant].push(idx);
        }

        let child_half = half * 0.5;
        let mut children = [-1i32; 8];
        for (octant, bucket) in buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let cc = [
                center[0]
                    + if octant & 1 != 0 {
                        child_half
                    } else {
                        -child_half
                    },
                center[1]
                    + if octant & 2 != 0 {
                        child_half
                    } else {
                        -child_half
                    },
                center[2]
                    + if octant & 4 != 0 {
                        child_half
                    } else {
                        -child_half
                    },
            ];
            children[octant] = self.build_node(positions, cc, child_half, bucket, depth + 1) as i32;
        }

        // `count` spans every descendant: the ordered slice grew from `start` to
        // its current length as the children were appended, contiguously.
        let count = self.ordered.len() as u32 - start;
        let id = self.nodes.len() as u32;
        self.nodes.push(OctreeNode {
            center,
            half_width: half,
            radius: half * SQRT_3,
            start,
            count,
            children,
            is_leaf: false,
        });
        id
    }

    /// The root node's id.
    #[inline]
    pub fn root(&self) -> usize {
        self.root as usize
    }

    /// All nodes, root last (children precede their parents).
    #[inline]
    pub fn nodes(&self) -> &[OctreeNode] {
        &self.nodes
    }

    /// A single node by id.
    #[inline]
    pub fn node(&self, id: usize) -> &OctreeNode {
        &self.nodes[id]
    }

    /// Atom indices grouped so every node's atoms form a contiguous slice.
    #[inline]
    pub fn ordered_atoms(&self) -> &[u32] {
        &self.ordered
    }

    /// The atom indices belonging to node `id` (works for internal nodes too).
    #[inline]
    pub fn node_atoms(&self, id: usize) -> &[u32] {
        let n = &self.nodes[id];
        &self.ordered[n.start as usize..(n.start + n.count) as usize]
    }

    /// All unordered pairs `(i, j)` with `i < j` whose points are within `cutoff`
    /// of each other, found via a dual-tree traversal (node pairs whose cubes are
    /// more than `cutoff` apart are pruned wholesale).
    ///
    /// This is the tree-driven **near** set of a near/far split. With
    /// `cutoff ≥ diameter` of the point set it is exactly the brute-force O(n²)
    /// neighbor set; the complement (pairs beyond `cutoff`) is the far set.
    ///
    /// `positions` must be the slice the tree was built from. Returned pairs are
    /// each emitted once; order is unspecified.
    pub fn near_pairs(&self, positions: &[[f64; 3]], cutoff: f64) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let cutoff_sq = cutoff * cutoff;
        if !self.nodes.is_empty() {
            self.pairs_within(self.root as usize, positions, cutoff_sq, &mut out);
        }
        out
    }

    /// Emit within-cutoff pairs inside one node (its atoms among themselves, then
    /// across every distinct pair of its children).
    fn pairs_within(
        &self,
        id: usize,
        positions: &[[f64; 3]],
        cutoff_sq: f64,
        out: &mut Vec<(usize, usize)>,
    ) {
        let node = &self.nodes[id];
        if node.is_leaf {
            let atoms = self.node_atoms(id);
            for a in 0..atoms.len() {
                let i = atoms[a] as usize;
                for &jb in &atoms[a + 1..] {
                    let j = jb as usize;
                    if dist_sq(positions[i], positions[j]) <= cutoff_sq {
                        out.push(order(i, j));
                    }
                }
            }
            return;
        }
        let kids: Vec<usize> = node
            .children
            .iter()
            .filter(|&&c| c >= 0)
            .map(|&c| c as usize)
            .collect();
        for (x, &cx) in kids.iter().enumerate() {
            self.pairs_within(cx, positions, cutoff_sq, out);
            for &cy in &kids[x + 1..] {
                self.pairs_across(cx, cy, positions, cutoff_sq, out);
            }
        }
    }

    /// Emit within-cutoff pairs with one endpoint under node `a` and the other
    /// under the disjoint node `b`, pruning when the cubes are too far apart.
    fn pairs_across(
        &self,
        a: usize,
        b: usize,
        positions: &[[f64; 3]],
        cutoff_sq: f64,
        out: &mut Vec<(usize, usize)>,
    ) {
        let na = &self.nodes[a];
        let nb = &self.nodes[b];
        if na.box_min_dist_sq(nb) > cutoff_sq {
            return;
        }
        match (na.is_leaf, nb.is_leaf) {
            (true, true) => {
                for &ia in self.node_atoms(a) {
                    let i = ia as usize;
                    for &jb in self.node_atoms(b) {
                        let j = jb as usize;
                        if dist_sq(positions[i], positions[j]) <= cutoff_sq {
                            out.push(order(i, j));
                        }
                    }
                }
            }
            // Descend the internal side (the larger cube when both are internal),
            // shrinking cubes fastest so the distance prune bites sooner.
            (false, true) => {
                self.for_children(a, |c| self.pairs_across(c, b, positions, cutoff_sq, out))
            }
            (true, false) => {
                self.for_children(b, |c| self.pairs_across(a, c, positions, cutoff_sq, out))
            }
            (false, false) => {
                if na.half_width >= nb.half_width {
                    self.for_children(a, |c| self.pairs_across(c, b, positions, cutoff_sq, out));
                } else {
                    self.for_children(b, |c| self.pairs_across(a, c, positions, cutoff_sq, out));
                }
            }
        }
    }

    /// Run `f` on each existing child id of node `id`.
    #[inline]
    fn for_children(&self, id: usize, mut f: impl FnMut(usize)) {
        for &c in &self.nodes[id].children {
            if c >= 0 {
                f(c as usize);
            }
        }
    }

    /// Partition every source relative to one `target` point into a **near** part
    /// (individual atom indices, to be summed directly) and a **far** part (tree
    /// node ids, each to be approximated by that node's aggregate).
    ///
    /// This is a Barnes–Hut walk from the root using the multipole-acceptance
    /// criterion `cell_size < theta · distance` (with `cell_size = 2 · half_width`
    /// the cube side and `distance` from `target` to the cube centre). A node that
    /// passes is emitted to `far`; a leaf that fails contributes its atoms to
    /// `near`; an internal node that fails is opened. Smaller `theta` accepts
    /// fewer/further nodes → more accurate, more work; `theta → 0` degenerates to
    /// an all-near direct sum. For `theta < 1` an accepted node is provably
    /// separated from `target` (distance `> cell_size`), so `target`'s own cell is
    /// never approximated.
    ///
    /// `near` and `far` are cleared first. Every source atom is represented
    /// exactly once across the two outputs (the target itself, if part of the set,
    /// lands in `near`; the caller skips the self term).
    pub fn far_field_partition(
        &self,
        target: [f64; 3],
        theta: f64,
        near: &mut Vec<u32>,
        far: &mut Vec<u32>,
    ) {
        near.clear();
        far.clear();
        if self.nodes.is_empty() {
            return;
        }
        let theta_sq = theta * theta;
        let mut stack = vec![self.root as usize];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id];
            if node.count == 0 {
                continue;
            }
            let dx = target[0] - node.center[0];
            let dy = target[1] - node.center[1];
            let dz = target[2] - node.center[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let size = 2.0 * node.half_width;
            // Accept as far when size < theta * distance, i.e. size² < theta² · d².
            if size * size < theta_sq * dist_sq {
                far.push(id as u32);
            } else if node.is_leaf {
                near.extend_from_slice(self.node_atoms(id));
            } else {
                for &c in &node.children {
                    if c >= 0 {
                        stack.push(c as usize);
                    }
                }
            }
        }
    }

    /// Compute each node's [`Monopole`] (total weight + weight-averaged centre)
    /// from caller-supplied per-atom `weights`, indexed to match `positions`.
    ///
    /// Returned vector is indexed by node id (same order as [`nodes`](Self::nodes)).
    /// This is the aggregate a treecode multiplies into the far half of
    /// [`far_field_partition`](Self::far_field_partition). Weights are treated as
    /// opaque scalars; for a signed quantity (e.g. charge) pass magnitudes for
    /// the acceptance geometry and carry the sign in the caller's kernel, or
    /// expand about the geometric centre. Cost is O(#nodes) via a single
    /// post-order sweep (children precede parents in node order).
    pub fn monopoles(&self, positions: &[[f64; 3]], weights: &[f64]) -> Vec<Monopole> {
        // Accumulate Σw and Σw·x per node; leaves sum their atoms, internals sum
        // their children. Node ids increase from leaves to root, so one forward
        // pass suffices.
        let mut sum_w = vec![0.0f64; self.nodes.len()];
        let mut sum_wx = vec![[0.0f64; 3]; self.nodes.len()];
        for (id, node) in self.nodes.iter().enumerate() {
            if node.is_leaf {
                for &a in self.node_atoms(id) {
                    let w = weights[a as usize];
                    let p = positions[a as usize];
                    sum_w[id] += w;
                    sum_wx[id][0] += w * p[0];
                    sum_wx[id][1] += w * p[1];
                    sum_wx[id][2] += w * p[2];
                }
            } else {
                for &c in &node.children {
                    if c >= 0 {
                        let c = c as usize;
                        sum_w[id] += sum_w[c];
                        sum_wx[id][0] += sum_wx[c][0];
                        sum_wx[id][1] += sum_wx[c][1];
                        sum_wx[id][2] += sum_wx[c][2];
                    }
                }
            }
        }
        self.nodes
            .iter()
            .enumerate()
            .map(|(id, node)| {
                let w = sum_w[id];
                let center = if w.abs() > f64::MIN_POSITIVE {
                    [sum_wx[id][0] / w, sum_wx[id][1] / w, sum_wx[id][2] / w]
                } else {
                    node.center
                };
                Monopole {
                    total_weight: w,
                    center,
                }
            })
            .collect()
    }
}

const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// The cube (centre, half-side) that tightly encloses `positions`. A little
/// padding keeps points that sit exactly on a face strictly inside the root.
fn enclosing_cube(positions: &[[f64; 3]]) -> ([f64; 3], f64) {
    if positions.is_empty() {
        return ([0.0; 3], 1.0);
    }
    let mut lo = positions[0];
    let mut hi = positions[0];
    for p in positions {
        for d in 0..3 {
            if p[d] < lo[d] {
                lo[d] = p[d];
            }
            if p[d] > hi[d] {
                hi[d] = p[d];
            }
        }
    }
    let center = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    let mut half = 0.0f64;
    for d in 0..3 {
        half = half.max((hi[d] - lo[d]) * 0.5);
    }
    // Pad so boundary points fall strictly inside (and avoid a zero-size cube).
    half = half * (1.0 + 1e-9) + 1e-12;
    (center, half)
}

#[inline]
fn dist_sq(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[inline]
fn order(i: usize, j: usize) -> (usize, usize) {
    if i < j {
        (i, j)
    } else {
        (j, i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

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

    /// Brute-force reference for the within-cutoff pair set.
    fn brute_pairs(positions: &[[f64; 3]], cutoff: f64) -> Vec<(usize, usize)> {
        let c2 = cutoff * cutoff;
        let mut out = Vec::new();
        for i in 0..positions.len() {
            for j in i + 1..positions.len() {
                if dist_sq(positions[i], positions[j]) <= c2 {
                    out.push((i, j));
                }
            }
        }
        out
    }

    fn sorted(mut v: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
        v.sort_unstable();
        v
    }

    #[test]
    fn every_atom_lands_in_exactly_one_leaf() {
        let pts = random_points(500, 1);
        let tree = Octree::build(&pts, 8, 20);
        // ordered_atoms is a permutation of 0..n
        let mut seen = vec![false; pts.len()];
        for &a in tree.ordered_atoms() {
            assert!(!seen[a as usize], "atom {a} appears twice");
            seen[a as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "some atom missing from the tree");
        // Root spans all atoms.
        assert_eq!(tree.node(tree.root()).count as usize, pts.len());
    }

    #[test]
    fn near_pairs_matches_brute_force_partial_cutoff() {
        let pts = random_points(400, 2);
        let tree = Octree::build(&pts, 6, 20);
        for &cutoff in &[0.05, 0.1, 0.2, 0.37] {
            let got = sorted(tree.near_pairs(&pts, cutoff));
            let want = sorted(brute_pairs(&pts, cutoff));
            assert_eq!(got, want, "near_pairs != brute force at cutoff {cutoff}");
        }
    }

    #[test]
    fn near_pairs_reproduces_full_all_pairs_set() {
        let pts = random_points(300, 3);
        let tree = Octree::build(&pts, 5, 20);
        // Cutoff beyond the diameter → every pair is "near".
        let got = tree.near_pairs(&pts, 100.0);
        assert_eq!(got.len(), pts.len() * (pts.len() - 1) / 2);
        assert_eq!(sorted(got), sorted(brute_pairs(&pts, 100.0)));
    }

    #[test]
    fn far_field_partition_covers_every_source_once() {
        let pts = random_points(600, 4);
        let tree = Octree::build(&pts, 8, 20);
        let (mut near, mut far) = (Vec::new(), Vec::new());
        for &theta in &[0.2, 0.5, 0.8] {
            for i in 0..pts.len() {
                tree.far_field_partition(pts[i], theta, &mut near, &mut far);
                let far_atoms: usize = far
                    .iter()
                    .map(|&n| tree.node(n as usize).count as usize)
                    .sum();
                assert_eq!(
                    near.len() + far_atoms,
                    pts.len(),
                    "partition dropped/duplicated sources (theta {theta}, target {i})"
                );
                // Accepted far nodes must be genuinely separated from the target.
                for &n in &far {
                    let node = tree.node(n as usize);
                    let d = dist_sq(pts[i], node.center).sqrt();
                    assert!(
                        2.0 * node.half_width < theta * d + 1e-12,
                        "far node violates MAC"
                    );
                }
            }
        }
    }

    #[test]
    fn monopoles_conserve_weight_and_centroid_at_root() {
        let pts = random_points(500, 5);
        let mut rng = StdRng::seed_from_u64(99);
        let weights: Vec<f64> = (0..pts.len()).map(|_| rng.random::<f64>() + 0.1).collect();
        let tree = Octree::build(&pts, 8, 20);
        let mono = tree.monopoles(&pts, &weights);
        let root = &mono[tree.root()];
        let total_w: f64 = weights.iter().sum();
        let mut cx = [0.0; 3];
        for (p, &w) in pts.iter().zip(&weights) {
            for d in 0..3 {
                cx[d] += w * p[d];
            }
        }
        assert!((root.total_weight - total_w).abs() < 1e-9 * total_w);
        for (&rc, &c) in root.center.iter().zip(&cx) {
            assert!((rc - c / total_w).abs() < 1e-9);
        }
    }
}
