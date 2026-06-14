# SOIL — the `AtomData` contract

**SOIL** (Substrate for Off-lattice Interacting Lagrangians) is the
method-agnostic particle tier that sits between [GRASS](https://github.com/SueHeir/grass)
(the App / Plugin / scheduler / coupling framework) and a physics codebase
(DEM today; peridynamics, SPH, … later):

```
GRASS    framework: App, Plugin, Scheduler, IO, coupling      (no particles)
  └─ SOIL   substrate: Atom, domain decomp, comm, neighbor     (no physics)
       └─ DEM / Peridynamics / …   physics: forces, materials  (rides the substrate)
```

> SOIL currently lives in the `soil_core` / `soil_core` / `soil_derive`
> crates and is slated to be renamed `soil_core` / `soil_neighbor` / `soil_derive`.
> This document describes the **stable interface** between the substrate and any
> physics that rides it. It is the one contract a new Tier-2 codebase must honor.

---

## 1. What the substrate owns: the base `Atom`

SOIL owns one resource, `Atom`, holding the fields every particle method needs
regardless of physics:

| field | meaning |
|---|---|
| `tag`, `atom_type`, `origin_index`, `is_ghost` | identity / bookkeeping |
| `pos`, `vel`, `force` (`Vec<[f64; 3]>`) | Newtonian state |
| `mass`, `inv_mass`, `cutoff_radius`, `image` | integration + PBC + neighboring |
| `natoms`, `nlocal`, `nghost`, `ntypes`, `dt` | counts + clock |

The substrate is responsible for **domain decomposition, ghost (halo)
communication, atom migration, and neighbor-list construction** over this base
`Atom`. It knows nothing about contact forces, bonds, or damage.

## 2. How physics extends a particle: `#[derive(AtomData)]`

A Tier-2 crate adds per-particle columns by declaring a struct of
`Vec<f64>` / `Vec<[f64; N]>` fields and deriving `AtomData`:

```rust
use soil_derive::AtomData;   // → soil_derive::AtomData

#[derive(AtomData)]
pub struct DemAtom {
    /// Replicated to ghosts so neighbors can compute torque.
    #[forward]
    pub omega: Vec<[f64; 3]>,

    /// Accumulated on ghosts, summed back to the owner, reset each step.
    #[reverse]
    #[zero]
    pub torque: Vec<[f64; 3]>,

    /// Migrates with the atom but never crosses the ghost boundary.
    pub radius: Vec<f64>,
}
```

It is registered once, in the plugin's `build()`:

```rust
register_atom_data!(app, DemAtom::new());
```

After registration the substrate carries those columns through **every**
migration, ghost exchange, permutation, and restart automatically. Systems
reach the columns by type:

```rust
let dem = store.expect::<DemAtom>("DemAtom not registered");   // Ref<DemAtom>
let mut dem = store.expect_mut::<DemAtom>("…");                // RefMut<DemAtom>
```

That registration call is the entire integration surface. **If your physics
state fits the `AtomData` shape, the substrate moves it for you and you write no
communication code.**

## 3. The four lifecycle hooks (and their invariants)

The derive generates these from the field attributes. Each has a hard
semantic contract — honor it or parallel runs diverge silently.

| hook | fields | direction | semantics on unpack | when it runs |
|---|---|---|---|---|
| `pack` / `unpack` | **all** | owner → owner (other rank) | replace | atom **migration** (crosses a subdomain) |
| `pack_forward` / `unpack_forward` | `#[forward]` | owner → its **ghosts** | **overwrite** | each ghost/border exchange (per neighbor rebuild / pre-force) |
| `pack_reverse` / `unpack_reverse` | `#[reverse]` | **ghost** → owner | **additive (`+=`)** | after force computation |
| `zero` | `#[zero]` | local | resize to `n`, fill `0.0` | start of each timestep |

Plus `pack_all_for_restart` / `unpack_all_from_restart` (all fields, for
checkpointing) and `apply_permutation` / `truncate` / `swap_remove` (keep every
column in lockstep when the substrate reorders or culls atoms).

### The canonical per-step ordering

```
zero(#[zero])  →  forward(#[forward])  →  compute forces  →  reverse(#[reverse])  →  integrate
```

`#[reverse]` accumulators are almost always also `#[zero]`: they must start at
zero before ghosts add into them, or contributions double across steps.

## 4. Rules a Tier-2 codebase MUST follow

1. **Field types only.** Every column is `Vec<f64>` or `Vec<[f64; N]>`. No
   other types — the derive packs them as flat `f64` buffers.
2. **Never resize a column out of band.** The substrate keeps all columns the
   same length as the atom count via `truncate` / `swap_remove` /
   `apply_permutation`. Push/pop a single column yourself and indices desync.
   Resize only inside `zero()` (the derive handles it) or via the substrate's
   grow path.
3. **Classify each field by who needs it:**
   - `#[forward]` — read-only state a *neighbor* needs to compute its force
     (radius via `Atom`, `omega`, temperature, damage). Overwrite semantics →
     the value on a ghost is a replica, never a place to accumulate.
   - `#[reverse]` — a contribution *computed on a ghost* that must reach the
     owner (`force` via `Atom`, `torque`, `heat_flux`). Additive → pair with
     `#[zero]`.
   - neither — state that must follow the atom across subdomains but never
     crosses the ghost boundary (`radius`, bond family, plastic history).
4. **`#[reverse]` ⊇ pair with `#[zero]`** unless you have a specific reason not
   to reset it each step.
5. **Determinism.** Pack/unpack order is field-declaration order. Reverse
   accumulation is `+=` in receipt order; keep per-pair contributions
   associative-safe (don't rely on summation order for bit-exactness).

## 5. Worked example — does the seam hold for peridynamics?

Peridynamics is bond-based and Lagrangian, so it should drop straight onto
SOIL with no substrate change. The test:

```rust
#[derive(AtomData)]
pub struct PeriAtom {
    /// Scalar damage φ ∈ [0,1]; neighbors need it to compute the bond force.
    #[forward]
    pub damage: Vec<f64>,

    /// Pairwise force-state contributions summed from ghost bonds, reset/step.
    #[reverse]
    #[zero]
    pub force_state: Vec<[f64; 3]>,

    /// Weighted volume / dilatation accumulator (state-based PD), reset/step.
    #[reverse]
    #[zero]
    pub dilatation: Vec<f64>,

    /// Reference (undeformed) position — migrates with the atom, never a ghost.
    pub ref_pos: Vec<[f64; 3]>,
}
```

Everything peridynamics needs maps onto an existing attribute:
`damage` is replicated state (`#[forward]`), the force/dilatation accumulators
are ghost-summed (`#[reverse]` + `#[zero]`), and the reference configuration
migrates but never ghosts (no attribute). **No new substrate primitive is
required** — which is the evidence that the GRASS → SOIL → physics tiering is
real and not a relabeling. The one thing PD needs beyond this is a *bond/horizon
neighbor list* with a larger, fixed cutoff; that is a neighbor-list strategy on
the SOIL side, configured per-physics, not a change to the `AtomData` contract.

## 6. The boundary rule

> **Nothing in SOIL may depend on a physics crate.** `soil_*` crates depend on
> `grass_*` and each other — never on `dem_*`, `peri_*`, etc. Physics crates
> depend on `soil_*`. This is already true today (`soil_core` has zero DEM
> dependencies); the rule just keeps it true.

If you ever find yourself wanting to add a physics concept to the substrate to
make a column work, that's the signal it belongs in a Tier-2 crate as an
`AtomData` instead.
