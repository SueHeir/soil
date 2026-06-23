# Choosing Field Attributes

Every `AtomData` column gets classified by **who needs the value and which
direction it has to travel**. Get this right and the substrate moves your state
correctly in parallel; get it wrong and parallel runs diverge silently. This is
the decision tree behind the [AtomData contract](../substrate/atomdata-contract.md).

## The three cases

```
Does a NEIGHBOR need to read this value to compute its force?
│
├─ yes, it's read-only replicated state (radius, omega, temperature, damage)
│     →  #[forward]          (owner → ghosts, overwrite each exchange)
│
├─ it's a contribution COMPUTED ON A GHOST that must reach the owner
│     (force, torque, heat_flux)
│     →  #[reverse] #[zero]  (ghost → owner, additive; reset each step)
│
└─ no neighbor needs it across ranks; it just follows the atom
      (bond family, plastic history, reference position)
      →  no attribute        (migrates with the atom, never ghosts)
```

## Quick reference

| attribute | direction | semantics | example |
|---|---|---|---|
| `#[forward]` | owner → ghosts | overwrite | `radius`, `omega`, temperature |
| `#[reverse]` | ghost → owner | additive (`+=`) | `torque`, `heat_flux` |
| `#[zero]` | local | reset to 0 each step | pairs with every `#[reverse]` |
| *(none)* | migrates only | replace on migration | `radius` history, `ref_pos` |

## Rules worth memorizing

- **`#[reverse]` almost always pairs with `#[zero]`.** A reverse accumulator must
  start at zero before ghosts add into it, or contributions double across steps.
- **Field types only:** every column is `Vec<f64>` or `Vec<[f64; N]>`. The derive
  packs them as flat `f64` buffers.
- **Never resize a column out of band.** The substrate keeps all columns the same
  length as the atom count. Resize only inside `zero()` (the derive handles it).
- **`#[forward]` is a replica, never an accumulator.** Its value on a ghost is
  overwritten each exchange — don't try to sum into it.

## Field order is the wire layout

The derive's generated `pack` / `unpack` serialize fields **in declaration
order** into a flat `f64` buffer, and that same order is the migration *and*
restart-file layout. **Reordering, inserting, or removing a field is a format
break:** old restart files and in-flight migration buffers will deserialize into
the wrong columns. Treat the field list as a versioned wire format.

For reference, the derive macro generates:

- `pack` / `unpack` — serialize **all** fields for atom migration.
- `truncate` / `swap_remove` — resize all field vectors together.
- `apply_permutation` — reorder all field vectors by a permutation.
- `pack_forward` / `unpack_forward` — communicate `#[forward]` fields (overwrite).
- `pack_reverse` / `unpack_reverse` — communicate `#[reverse]` fields (accumulate).
- `zero` — zero out `#[zero]` fields for `n` atoms.

If you need an *irregular* layout that the derive can't express, implement the
trait by hand — see [Writing an AtomData Extension](../substrate/writing-atomdata.md).

> **The derive enforces less than the convention asks.** `#[reverse]` does
> **not** imply `#[zero]` — the two are independent flags, and the macro will
> happily generate a reverse accumulator that is never reset, which double-counts
> across steps. Pair them yourself. Every field must be exactly `Vec<f64>` or
> `Vec<[f64; N]>` with a literal `N`; a type alias such as `type Col = Vec<f64>`
> is *not* recognized and fails to compile. And, as noted above, the derive
> generates no `new()` or `Default` — build the struct literal yourself.

For the full semantics, invariants, and the peridynamics worked example that
proves the seam holds, see [The AtomData Contract](../substrate/atomdata-contract.md).
