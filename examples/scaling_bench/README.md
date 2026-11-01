# scaling_bench

Parallel-scaling benchmark for the SOIL substrate. A dense, frictionless
granular gas (spring-dashpot normal contact) runs through the stock plugins
(`AtomPlugin`, `DomainPlugin`, `NeighborPlugin`, `CommunicationPlugin`,
`VelocityVerletPlugin`) with a DEM-shaped `AtomData` extension whose columns
mirror DIRT's `DemAtom` (forward `omega` and `body_id`, reverse/zeroed
`torque`, owner-only orientation and inertia). Each phase of the step is
bracketed by timer systems, and rank 0 prints a LAMMPS-style min/avg/max
breakdown plus decomposition-independent checksums (global KE and spin) so a
multi-rank run can be checked against the serial one.

```sh
cargo build --release --example scaling_bench
# strong scaling: fixed 32^3 lattice
mpiexec -n 4 target/release/examples/scaling_bench --global 32 --grid 2x2x1 --steps 500
# weak scaling: 20^3 particles per rank
mpiexec -n 4 target/release/examples/scaling_bench --per-rank 20 --grid 2x2x1 --steps 500
```

Options: `--global N|NxMxK`, `--per-rank N|NxMxK`, `--grid PxQxR`, `--steps`,
`--warmup`, `--spacing` (lattice pitch in diameters, default 1.02), `--vmax`
(initial speed, default 0.5), `--skin` (neighbor `skin_fraction`, default
1.15), `--seed`. Initial velocities depend only on the global tag, so every
decomposition starts from the same state.

## Before/after: particle-store and hot-path fixes

Measured on a 4-core cloud VM (OpenMPI, shared memory, `--bind-to core`),
`lto = "fat"`, `codegen-units = 1`, 500 steps after 20 warm-up steps. "Before"
is `main` at `3c274b6`; "after" is the branch that introduced this example.
Loop time excludes setup.

| case | ranks | atoms | setup before | setup after | loop before | loop after | speedup |
|---|---|---|---|---|---|---|---|
| strong | 1 | 32,768 | 55.0 s | 0.007 s | 1.686 s | 1.119 s | 1.51x |
| strong | 2 | 32,768 | 8.0 s | 0.006 s | 1.173 s | 0.675 s | 1.74x |
| strong | 4 | 32,768 | 1.18 s | 0.002 s | 0.554 s | 0.344 s | 1.61x |
| weak | 1 | 8,000 | 1.08 s | 0.003 s | 0.362 s | 0.258 s | 1.40x |
| weak | 2 | 16,000 | 1.11 s | 0.003 s | 0.429 s | 0.314 s | 1.37x |
| weak | 4 | 32,000 | 1.37 s | 0.004 s | 0.573 s | 0.327 s | 1.75x |

Parallel efficiency at 4 ranks: strong 76% -> 81%, weak 63% -> 79%.

Per-phase, max over ranks, 32,768 atoms (500 steps):

| phase | 1 rank before | 1 rank after | 4 ranks before | 4 ranks after |
|---|---|---|---|---|
| borders (5 rebuilds) | 0.228 s | 0.018 s | 0.037 s | 0.009 s |
| exchange | 0.0002 s | 0.0003 s | 0.0185 s | 0.0006 s |
| forward comm | 0.162 s | 0.122 s | 0.089 s | 0.058 s |
| reverse comm | 0.097 s | 0.061 s | 0.115 s | 0.083 s |
| rebuild decision | 0.267 s | 0.100 s | 0.114 s | 0.042 s |

Setup was quadratic before (every `push_default_local` deep-copied the whole
store), so it grows as N^2: 32k atoms took 55 s on one rank, and a 1M-atom
insertion would take hours.
