# particleswith_queries

This validation driver exercises the generic SOIL `ParticlesWith` extension
query API: ordered tuple views, mutation through `Write`, absent optional
columns, preparation-time missing-column and duplicate-access errors, and the
scheduler's read/write declaration. A separate allocator guard runs one
thousand `(Write<Position>, Read<Velocity>)` fetches and requires zero heap
allocations.

Run from the repository root:

```sh
uv run --with matplotlib python examples/particleswith_queries/sweep.py
```

`uv` supplies the plotting interpreter and `matplotlib`; the driver itself
selects SOIL's non-MPI, precision-double Cargo configuration, so it does not
depend on a system MPI toolchain.

![ParticlesWith acceptance checks](plots/particleswith_queries.png)

PASS means all five runtime query gates passed and the measured hot-loop
allocation count matched the declared zero-allocation requirement.
