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

It knows nothing about contact forces, bonds, damage, or any specific physics.

## The crates

| crate | role |
|---|---|
| `soil_core` | base `Atom` + `AtomData` registry, comm, domain decomposition, neighbor, regions, groups |
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
