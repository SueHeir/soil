# Octree Near/Far Partition

`soil_core::tree::Octree` is SOIL's spatial hierarchy for long-range particle
methods. It is the tree and partitioning surface only: positions go in, atom or
tree-node indices come out, and a physics tier decides what to do with those
indices.

The public Rust API is rustdoc-commented in `crates/soil_core/src/tree.rs`.
Because `soil_core` is not published on docs.rs, the mdBook links below point to
the checked reference sections rather than to broken public rustdoc URLs:

- [`Octree`](#what-soil-provides)
- [`OctreeNode`](#far-field-partition)
- [`Monopole`](#monopoles)

SOIL also re-exports these at the crate root as `soil_core::Octree`,
`soil_core::OctreeNode`, and `soil_core::Monopole`.

## What SOIL provides

<a id="build"></a>
Use [`Octree::build`](#build)
when you want SOIL to choose a root cube that encloses the input points:

```rust,ignore
use soil_core::Octree;

let positions: Vec<[f64; 3]> = load_positions();
let tree = Octree::build(&positions, 16, 24);
```

<a id="build_in_box"></a>
Use [`Octree::build_in_box`](#build_in_box)
when the caller already owns the enclosing box:

```rust,ignore
let tree = Octree::build_in_box(
    &positions,
    [0.0, 0.0, 0.0], // root cube center
    10.0,            // root half-side
    16,              // leaf_capacity
    24,              // max_depth
);
```

Both constructors store the tree topology and an atom ordering. They do not copy
or own the particle coordinates; pass the same `positions` slice back to query
methods.

## Near Pairs

<a id="near_pairs"></a>
[`Octree::near_pairs`](#near_pairs)
returns every unordered pair `(i, j)` with `i < j` whose points are within a
cutoff:

```rust,ignore
let near = tree.near_pairs(&positions, cutoff);
for (i, j) in near {
    // Sum the consumer's direct, pairwise near-field contribution here.
}
```

This is a tree-driven cutoff query. With a cutoff at least as large as the point
set diameter it returns the complete brute-force pair set; with a smaller cutoff
it returns exactly the pairs inside that radius. The complement is the far part
for algorithms that split direct near work from approximated long-range work.

For short-range local physics inside a decomposed SOIL simulation, the ordinary
`Neighbor` list remains the timestep primitive because it is integrated with
ghost exchange and rebuild policy. `near_pairs` is the standalone tree query used
when a long-range method is assembling its own near/far split.

## Far-Field Partition

<a id="far_field_partition"></a>
[`Octree::far_field_partition`](#far_field_partition)
walks the tree for one target point and partitions every source into exactly one
of two outputs:

- `near`: atom indices to handle directly.
- `far`: tree node ids to approximate as aggregates.

```rust,ignore
let mut near = Vec::new();
let mut far = Vec::new();

tree.far_field_partition(target_position, theta, &mut near, &mut far);

for &j in &near {
    // Direct consumer-side term for source atom j.
}

for &node_id in &far {
    // Consumer-side aggregate term for this accepted tree node.
}
```

The walk uses the Barnes-Hut-style opening test `cell_size < theta * distance`,
where `cell_size` is the node cube side and `distance` is measured from the
target to the node center. A node that passes is emitted to `far`; a leaf that
does not pass contributes its atoms to `near`; an internal node that does not
pass is opened. The method clears the output vectors first.

Every source atom is represented once across `near` and `far`. If the target is
one of the source atoms, it lands in `near`; the consumer skips the self term in
its own kernel.

## Monopoles

<a id="monopoles_method"></a>
[`Octree::monopoles`](#monopoles_method)
computes a [`Monopole`](#monopoles)
for every tree node from caller-supplied per-atom weights:

```rust,ignore
let weights: Vec<f64> = load_non_negative_weights();
let monopoles = tree.monopoles(&positions, &weights);

for &node_id in &far {
    let aggregate = monopoles[node_id as usize];
    // aggregate.total_weight and aggregate.center feed the consumer's expansion.
}
```

The aggregate is only `sum(w_i)` and the weight-averaged center. SOIL does not
assign a physical meaning to the weight. It may be mass, charge magnitude,
particle volume, an SPH weight, or another non-negative quantity chosen by the
consumer.

## API Link Targets

These mdBook anchors mirror the public `soil_core::tree` rustdoc names and are
kept here so hosted docs do not depend on an unpublished docs.rs crate.

| API | Reference |
|---|---|
| `Octree` | [What SOIL provides](#what-soil-provides) |
| `OctreeNode` | [Far-field partition node ids](#far-field-partition) |
| `Monopole` | [Monopoles](#monopoles) |
| `Octree::build` | [build](#build) |
| `Octree::build_in_box` | [build_in_box](#build_in_box) |
| `Octree::near_pairs` | [near_pairs](#near_pairs) |
| `Octree::far_field_partition` | [far_field_partition](#far_field_partition) |
| `Octree::monopoles` | [monopoles](#monopoles_method) |

## Choosing The Parameters

`leaf_capacity` controls how many atoms a leaf may hold before it splits. Smaller
values create more nodes and shorter direct loops; larger values create fewer
nodes and more direct work inside each leaf. A practical starting range is
`8..32`; use the method's own accuracy and timing checks to tune it.

`max_depth` is a recursion guard. It prevents coincident or nearly coincident
points from subdividing forever; an over-capacity cluster at the depth cap stays
in one leaf. Values around `20..32` are usually enough for normalized or
well-scaled coordinates. Increase it only when the point distribution and root
box warrant more spatial resolution.

`theta` is the opening angle for `far_field_partition`. Smaller `theta` opens
more nodes, puts more sources in `near`, and makes a consumer's approximation
closer to its direct sum. Larger `theta` accepts more far nodes and does less
work, but the approximation quality is entirely the consumer method's
responsibility. `theta -> 0` degenerates toward an all-near direct evaluation.

## The Method-Agnostic Contract

The octree API is a substrate primitive. It knows positions, tree cells, atom
indices, and optional caller-supplied scalar weights. It does not know:

- a force law or potential,
- a gravity or electrostatics constant,
- an SPH, MD, peridynamics, DEM, or vortex method,
- a multipole expansion beyond the geometric monopole helper,
- a timestepper or energy definition.

Those belong in physics crates or examples above SOIL. The same `Octree` surface
can serve gravity, electrostatics, molecular dynamics, SPH, peridynamics, or a
global constraint solver because none of those methods appear in the substrate
API.

## Validation Examples

[`examples/tree_farfield_proof`](https://github.com/SueHeir/soil/tree/main/examples/tree_farfield_proof)
is the substrate-facing check for the seam. It builds a consumer-side `1/r`
treecode on top of `Octree`, then checks three things against brute-force
references:

- `near_pairs` matches the brute-force within-cutoff pair set, including the
  all-pairs case.
- A monopole treecode assembled from `far_field_partition` and `monopoles`
  converges toward the direct `O(n^2)` sum as `theta` decreases.
- The accepted near atoms plus far nodes reduce the measured interaction count
  below quadratic growth for the checked point sets.

[`examples/orbit_nbody_gravity`](https://github.com/SueHeir/soil/tree/main/examples/orbit_nbody_gravity)
is a consumer example. It keeps gravity-specific state, acceleration evaluation,
time integration, energy, and the analytic Kepler validation in the example, and
uses SOIL only for the agnostic tree partition and monopole aggregation. That is
the placement pattern to follow: physics kernels consume the tree above SOIL; the
substrate docs describe the API contract, not a force law.
