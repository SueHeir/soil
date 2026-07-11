# particle_store_facade

This executable checks the `ParticleStore` communication boundary rather than
the older extension-registration contract.  It creates generic SoA particles
through transactional default insertion, packs a variable-width multi-record
migrant stream, and receives it through the framed batch API.  Core tags and
positions plus an independently registered extension must retain exact order.
A valid stream with a trailing field must fail as one batch and restore the
original core and extension rows.  The final case records elapsed time for the
same facade receive path over the configured number of rows.

Run from the repository root:

```sh
uv run --with matplotlib python examples/particle_store_facade/sweep.py
```

![ParticleStore framed communication validation](plots/particle_store_facade.png)

PASS means the facade's actual framed communication path preserved all rows in
the multi-record case, rejected the malformed batch without partial mutation,
and completed the measured representative receive workload.  This is generic
storage validation: it uses no force model or method-specific particle data.
