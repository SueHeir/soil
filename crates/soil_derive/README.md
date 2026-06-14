# soil_derive

Procedural macro for deriving `AtomData` implementations in [SOIL](https://github.com/SueHeir/soil).

## What it does

Provides a single derive macro, `#[derive(AtomData)]`, that generates a full `AtomData` trait implementation for structs whose fields are per-atom data columns. Every field must be `Vec<f64>` or `Vec<[f64; N]>`; anything else is a compile error. The generated code lets the substrate pack, unpack, permute, and communicate per-atom data without hand-written boilerplate.

> `#[derive(StageEnum)]` and `#[derive(ScheduleSet)]` now live in [`grass_derive`](https://docs.rs/grass_derive) — import them from there (or via `dirt_core::prelude::*`).

## Usage

```rust,ignore
use soil_derive::AtomData;

#[derive(AtomData)]
pub struct DemAtom {
    #[forward]
    pub omega: Vec<[f64; 3]>,   // angular velocity, sent to ghosts (overwrite)
    #[reverse]
    #[zero]
    pub torque: Vec<[f64; 3]>,  // accumulated from ghosts, zeroed each step
    pub radius: Vec<f64>,       // migrated with the atom, no comm
}
```

## Field attributes

| Attribute    | Effect |
|--------------|--------|
| `#[forward]` | Included in `pack_forward` / `unpack_forward` (overwrite on unpack) |
| `#[reverse]` | Included in `pack_reverse` / `unpack_reverse` (additive `+=` on unpack) |
| `#[zero]`    | Included in `zero()` — resized to `n` atoms and filled with zeros |

Attributes may be combined freely; a field with no attribute is still migrated and permuted with the atom.

## Generated methods

- `pack` / `unpack` — serialize **all** fields for atom migration
- `truncate` / `swap_remove` — resize all field vectors together
- `apply_permutation` — reorder all field vectors by a permutation
- `pack_forward` / `unpack_forward` / `forward_comm_size` — communicate `#[forward]` fields
- `pack_reverse` / `unpack_reverse` / `reverse_comm_size` — communicate `#[reverse]` fields (additive)
- `zero` — resize and zero `#[zero]` fields for `n` atoms
- `as_any` / `as_any_mut` — downcast support

## Validation

Produces a clear compile-time error when applied to an enum, union, or tuple/unit struct, or when any field is not `Vec<f64>` or `Vec<[f64; N]>`.

## License

MIT OR Apache-2.0
