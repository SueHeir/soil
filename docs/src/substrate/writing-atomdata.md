# Writing an AtomData Extension

A physics tier attaches its per-particle state by implementing `AtomData` and
registering an instance with `register_atom_data!`. Almost always you get the
implementation for free from `#[derive(AtomData)]` (in `soil_derive`); implement
the trait by hand only for irregular layouts such as `BondStore`, where the
columns aren't simple `Vec<f64>` / `Vec<[f64; N]>` fields.

This page is the hand-implementer's reference. If you are using the derive, the
attribute classification in
[Choosing Field Attributes](../tutorial/field-attributes.md) and the
[AtomData Contract](./atomdata-contract.md) are what you need; come back here
only if you write the `impl` yourself.

## What you must provide vs. what defaults

**Mandatory** — every extension must keep its columns consistent under
structural operations:

- `as_any` / `as_any_mut` — downcasting the registry entry back to your type.
- `truncate` / `swap_remove` — resize all columns together (drop ghosts; remove
  one atom).
- `apply_permutation` — reorder all columns by a spatial re-sort.
- `pack` / `unpack` — full-state serialize for **migration** and **restart**.

**Defaulted** — no-ops unless your state needs them:

- The forward pass: `pack_forward` / `unpack_forward` / `forward_comm_size`.
- The reverse pass: `pack_reverse` / `unpack_reverse` / `reverse_comm_size`.
- `zero`.

(These map onto the same structural ops the substrate mirrors across every
registered extension — see [Substrate Internals](../reference/internals.md).)

## The two contracts that make parallel runs correct

- **forward = overwrite, reverse = accumulate.** A `#[forward]` field is
  read-only replicated state a neighbor needs (radius, omega, temperature): the
  owner's value is *copied over* the ghost's each exchange. A `#[reverse]` field
  is a contribution computed on a ghost that must reach the owner (force, torque,
  heat flux): `unpack_reverse` must `+=` into the owner, and the field should
  also be `#[zero]` so it starts each step at zero.
- **`*_comm_size` must equal the number of `f64`s the matching `pack_*` pushes
  per atom**, or the buffer striding desyncs and ghosts read garbage. The derive
  keeps these in lockstep; a hand-written impl must too.

Field declaration order also defines the migration/restart wire layout, so
reordering fields is a restart-format break. (This invariant is shared with the
derive — see [Choosing Field Attributes](../tutorial/field-attributes.md#field-order-is-the-wire-layout).)
