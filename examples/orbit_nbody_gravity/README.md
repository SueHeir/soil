# orbit_nbody_gravity

N-body gravity demonstration built on SOIL octree primitives. Gravity-specific
state, timestepping, and validation remain in the example so `soil_core` stays
particle-method agnostic.

Run the validation sweep from the repository root:

```sh
python3 examples/orbit_nbody_gravity/sweep.py
```

![ORBIT N-body gravity validation](plots/orbit_nbody_gravity.png)

The plot compares the measured two-body period with the analytic Kepler
reference `2*pi` and the `1e-3` relative period gate. It also shows the
measured max relative energy drift against the `1e-6` pass gate and the
theta `4.0` negative-control drift threshold; the current sweep passes all
three checks.
