# virial_thermo_crossrank

SOIL-level validation that `VirialStress` pair-force accumulation reaches the
`soil_print` thermo columns through the global MPI reduction. The input TOML
declares two pair contributions; the analytic tensor is the direct sum
`r_ij outer F_i` over those rows.

The sweep runs the same case on 1, 2, and 4 ranks. On multi-rank layouts the
declared pair contributions are split across ranks before `soil_print` reduces
`virial_xx`, `virial_yy`, `virial_zz`, `virial_xy`, `virial_xz`, and
`virial_yz`.

```bash
python3 examples/virial_thermo_crossrank/sweep.py
```

![virial thermo cross-rank validation](plots/virial_thermo_crossrank.png)

PASS: measured thermo columns on 1, 2, and 4 MPI ranks match the analytic tensor
and the 1-rank reference within `1e-12`.
