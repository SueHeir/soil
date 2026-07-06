# dsmc_thermal_relaxation

DSMC thermal-relaxation validation case. SOIL supplies particle state and cell
binning; the Maxwell-molecule collision step and BGK exponential comparison stay
inside this example.

Run from the repository root:

```sh
python3 examples/dsmc_thermal_relaxation/sweep.py
```

![DSMC thermal relaxation](plots/dsmc_thermal_relaxation.png)

The plot compares the measured thermal anisotropy decay with the BGK
`exp(-tau)` reference, shows the +/-20% final-ratio pass band, and marks the
fitted decay-rate target `1.0 +/- 0.08`; the current sweep passes both gates.
