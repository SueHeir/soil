# SOIL Validation Examples

Validation examples live under top-level `examples/`, matching the DIRT layout.
Crate directories contain library code only.

## atomdata_row_contract

Exact row-count contract validation for late extension registration and a
malformed manual multi-column store. The accepted store must have three rows;
the malformed store's hidden two-row column must be rejected.

![AtomData row contract](atomdata_row_contract/plots/atomdata_row_contract.png)

## particle_store_facade

Executable validation and measured representative communication path for the
new synchronized `ParticleStore` facade.  It checks transactional local
construction, exact core-plus-extension parity for a framed multi-record
migration, and all-or-nothing rollback for a trailing-payload failure.

![ParticleStore facade communication validation](particle_store_facade/plots/particle_store_facade.png)

## particleswith_queries

Generic `ParticlesWith` typed extension-query validation: tuple order and
mutability, optional data, preparation errors, scheduler resource access, and
a zero-allocation hot-loop guard.

![ParticlesWith query validation](particleswith_queries/plots/particleswith_queries.png)

## virial_thermo_crossrank

MPI validation that `VirialStress` pair-force contributions reach the
`soil_print` thermo columns through a global reduction. The same declarative
pair-force tensor is checked against the analytic result on 1, 2, and 4 ranks.

![virial thermo cross-rank validation](virial_thermo_crossrank/plots/virial_thermo_crossrank.png)

## overlap_forward_validate

MPI validation that overlapped forward communication matches the baseline path
to the FP floor on 1, 2, and 4 ranks while doing measured interior work before
boundary work. The same sweep compares opt-in f32 forward ghost position/velocity
communication against the default f64 path.

![overlap forward validation](overlap_forward_validate/plots/overlap_forward_validate.png)

![f32 forward drift](overlap_forward_validate/plots/f32_forward_drift.png)

## exchange_multihop_gate

MPI release-build validation that a required two-rank atom migration stops with
a hard, actionable exchange error instead of leaving a local atom outside its
owner subdomain.

![exchange multihop gate](exchange_multihop_gate/plots/exchange_multihop_gate.png)

## polydisperse_neighbor_rebuild

Mixed-radius neighbor-list validation against an independent brute-force pair
set on 1, 2, and 4 MPI ranks. It checks that polydisperse SOIL neighbor lists
keep every valid pair while allowing per-atom rebuild budgets.

![polydisperse neighbor rebuild](polydisperse_neighbor_rebuild/plots/polydisperse_neighbor_rebuild.png)

## neighbor_comm_profile

Performance harness that records timing, allocation, and communication counters
for SOIL border exchange, forward/reverse communication, atom sorting, and
neighbor-list build phases on checked-in serial and MPI layouts.

![neighbor/comm profile](neighbor_comm_profile/plots/neighbor_comm_profile.png)

## neighbor_hot_path

Validation and microbenchmark for the neighbor distance kernel. Checked-in
uniform and polydisperse layouts compare SIMD-requested neighbor pairs against
the scalar pair set, then report per-build timing.

![neighbor distance kernel validation](neighbor_hot_path/plots/neighbor_hot_path.png)

## tree_farfield_proof

Octree near/far partition and far-field proof against brute-force references:
exact near pairs, convergent `1/r` treecode accuracy, and sub-quadratic
interaction-count scaling.

![tree far-field proof](tree_farfield_proof/plots/tree_farfield_proof.png)

## orbit_nbody_gravity

ORBIT octree gravity demonstration checked against the analytic two-body
Kepler period, a max-energy-drift tolerance, and a theta `4.0` negative
control.

![ORBIT N-body gravity validation](orbit_nbody_gravity/plots/orbit_nbody_gravity.png)

## dsmc_thermal_relaxation

Homogeneous Maxwell-molecule DSMC thermal-relaxation check against the BGK
`exp(-tau)` anisotropy decay reference, with the stochastic final-ratio and
fitted-rate gates shown in the committed figure.

![DSMC thermal relaxation](dsmc_thermal_relaxation/plots/dsmc_thermal_relaxation.png)
