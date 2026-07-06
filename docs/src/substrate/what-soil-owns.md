# What SOIL Owns

SOIL owns one resource, `Atom`, holding the fields every particle method needs
regardless of physics:

| field | meaning |
|---|---|
| `tag`, `atom_type`, `origin_index`, `is_ghost` | identity / bookkeeping |
| `pos`, `vel`, `force` (`Vec<[f64; 3]>`) | Newtonian state |
| `mass`, `inv_mass`, `cutoff_radius`, `image` | integration + PBC + neighboring |
| `natoms`, `nlocal`, `nghost`, `ntypes`, `dt` | counts + clock |

On top of that base atom, the substrate is responsible for:

- **Domain decomposition** — splitting the simulation box across MPI ranks.
- **Ghost (halo) communication** — replicating border atoms onto neighboring
  ranks so local force computations see their neighbors.
- **Atom migration** — moving an atom (and all its physics columns) to a new rank
  when it crosses a subdomain boundary.
- **Neighbor-list construction** — binning and building the pair lists physics
  iterates over.
- **Method-agnostic topology and exclusions** — storing per-atom bond adjacency
  and 1-2/1-3 pair-exclusion queries through
  [`BondStore` / `BondPlugin`](https://github.com/SueHeir/soil/blob/main/crates/soil_core/src/bond.rs).
- **Generic virial accumulation** — providing the
  [`VirialStress` / `VirialStressPlugin`](https://github.com/SueHeir/soil/blob/main/crates/soil_core/src/virial.rs)
  resource that force plugins can zero, fill, and publish through output.

It knows nothing about contact force laws, bond force laws, damage evolution, or
any specific physics. A downstream physics tier decides what a bond means, which
coefficients it uses, and how forces or damage are computed; SOIL only keeps the
topology/exclusion data in step with atom migration, ghosts, and restarts.

Likewise, `VirialStress` is an accumulation and output contract, not a stress
model. Force plugins opt in, add pair contributions using SOIL's sign
convention, and let `soil_print` expose the six tensor components.

## The crates

| crate | role |
|---|---|
| `soil_core` | base `Atom` + `AtomData` registry, comm, domain decomposition, neighbor, regions, groups, method-agnostic bond topology, virial accumulator |
| `soil_derive` | `#[derive(AtomData)]` proc macro |
| `soil_verlet` | velocity-Verlet translational integration |
| `soil_print` | thermo output, dump files (CSV/binary/VTP), restart |
| `soil_deform` | box deformation (strain rate, velocity, target size) + Lees–Edwards `xy` shear |
| `soil_fixes` | method-agnostic position constraint `pin` on base `Atom` state (DEM fixes — `freeze`, velocity damping — live in `dirt_fixes`) |

Each of these crates has its own chapter:

- [Time Integration](./integration.md) — velocity-Verlet kick-drift-kick and
  where the two halves run (`soil_verlet`).
- [Fixes](./fixes.md) — the method-agnostic `pin` constraint and the fix pattern
  (`soil_fixes`).
- [Box Deformation](./deformation.md) — strain rate / velocity / target styles
  and Lees–Edwards `xy` shear (`soil_deform`).

For the deeper machinery — struct-of-arrays invariant, local/ghost partition,
the two-path timestep, and the skin/cutoff/ghost-cutoff chain — see
[Substrate Internals](../reference/internals.md).

> The next chapter, [The AtomData Contract](./atomdata-contract.md), is the part
> you need before writing physics. If you implement `AtomData` by hand rather
> than via the derive, see
> [Writing an AtomData Extension](./writing-atomdata.md).
