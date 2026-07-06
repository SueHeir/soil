# exchange_multihop_gate

This MPI validation checks SOIL's release-build behavior when an atom would need
more than one neighbor hop during exchange. Four ranks split x into quarters; rank
0 deliberately owns one atom at `x=0.625`, which belongs to rank 2. Because the
current exchange implementation is single-hop, the run must stop with an
actionable error before the atom can be mis-binned.

```sh
python3 examples/exchange_multihop_gate/sweep.py
```

![exchange multihop gate](plots/exchange_multihop_gate.png)

PASS means the release MPI run exits nonzero and reports `SOIL exchange left`,
confirming that a required multi-hop migration cannot silently continue.
