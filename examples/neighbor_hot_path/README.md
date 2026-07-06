# neighbor_hot_path

Neighbor-list distance-kernel validation and microbenchmark. The example runs
the scalar neighbor builder and the SIMD-requested builder on the same
declarative uniform and polydisperse cases from `config.toml`, fails on any
pair-set mismatch, and records the per-build timing.

Run from the repository root:

```sh
python3 examples/neighbor_hot_path/sweep.py
```

![neighbor distance kernel validation](plots/neighbor_hot_path.png)

PASS means the SIMD-requested path produced exactly the same neighbor pairs as
the scalar path for both layouts; the bars show the measured build timing.
