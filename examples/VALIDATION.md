# SOIL Validation Examples

Validation examples live under top-level `examples/`, matching the DIRT layout.
Crate directories contain library code only.

## overlap_forward_validate

MPI validation that overlapped forward communication matches the baseline path
to the FP floor on 1, 2, and 4 ranks while doing measured interior work before
boundary work.

![overlap forward validation](overlap_forward_validate/plots/overlap_forward_validate.png)

## tree_farfield_proof

Octree near/far partition and far-field proof against brute-force references:
exact near pairs, convergent `1/r` treecode accuracy, and sub-quadratic
interaction-count scaling.

![tree far-field proof](tree_farfield_proof/plots/tree_farfield_proof.png)
