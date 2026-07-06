# neighbor_comm_profile

Reproducible SOIL substrate profiling harness for the communication and
neighbor-list hot path. It runs a deterministic particle lattice through the
normal SOIL systems and records timing, allocation, and communication counters
for:

- `sort_atoms_by_bin`
- `borders`
- `forward_comm`
- `reverse_comm`
- neighbor-list build (`bin_neighbor_list`)

Run the checked-in serial and MPI layouts from the repository root:

```sh
cargo build --release --example neighbor_comm_profile
target/release/examples/neighbor_comm_profile examples/neighbor_comm_profile/config_serial.toml
mpiexec -n 2 target/release/examples/neighbor_comm_profile examples/neighbor_comm_profile/config_mpi_x2.toml
```

Or run both layouts and regenerate the CSV/plot:

```sh
python3 examples/neighbor_comm_profile/sweep.py
```

The executable prints one `PROFILE` line per rank and phase. Compare changes by
running the same config before and after a branch and diffing the generated
`data/profile_summary.csv`; the stable keys are `layout`, `rank`, and `phase`.
Use `mean_ns`, `p2p_messages`, `p2p_f64`, `collectives`, `alloc_calls`, and
`alloc_bytes` as the first-pass counters before tuning individual loops.

![neighbor/comm profile](plots/neighbor_comm_profile.png)

The plot shows measured mean time by phase for the checked-in serial and
2-rank MPI layouts; the CSV records the raw counters used for comparison.
