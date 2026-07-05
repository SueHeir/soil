# overlap_forward_validate

MPI validation that `soil_core::forward_comm_overlap` produces the same final
state as the non-overlapped forward communication path while doing real interior
work before the boundary force. The case is substrate-level and particle-method
agnostic: a Lennard-Jones gas is only the local force used to expose the
communication behavior.

Run the full validation sweep from the repository root:

```sh
python3 examples/overlap_forward_validate/sweep.py
```

![overlap forward validation](plots/overlap_forward_validate.png)

The plot shows measured max state difference between overlap and baseline
against the `1e-12` tolerance, plus the measured interior/boundary work split.
PASS means every rank layout matches baseline to the FP floor and overlap mode
has non-empty ordered interior and boundary work.
