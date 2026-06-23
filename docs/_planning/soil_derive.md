# Planning: `soil_derive` documentation

Crate path: `crates/soil_derive/`
Source: `crates/soil_derive/src/lib.rs` (single file, 501 lines)
Cargo.toml: `crates/soil_derive/Cargo.toml`
Existing README: `crates/soil_derive/README.md`

---

## Purpose

`soil_derive` is a procedural-macro crate that provides `#[derive(AtomData)]`.
Its only job is to generate a complete `AtomData` trait implementation for any
struct whose fields are all `Vec<f64>` or `Vec<[f64; N]>`. The generated code
lets the SOIL substrate pack, unpack, permute, communicate, and zero per-atom
data columns without hand-written boilerplate.

The `AtomData` trait itself lives in `soil_core::atom`, not here. This crate
only generates the `impl` block. The crate has no runtime code — it is a
`proc-macro = true` lib with three dependencies: `syn`, `quote`, `proc-macro2`.

`#[derive(StageEnum)]` and `#[derive(ScheduleSet)]` previously lived here but
have been moved to `grass_derive`. `soil_derive` now exports exactly one item.

Source: `lib.rs:1–6` (module docstring), `Cargo.toml:11–12`

---

## Public surface to document

### Derive macro

```rust
#[derive(AtomData)]
```

Registered at `lib.rs:200`:
```rust
#[proc_macro_derive(AtomData, attributes(forward, reverse, zero))]
pub fn derive_atom_data(input: TokenStream) -> TokenStream
```

Applicable to: **structs with named fields only**. Enums, unions, tuple structs,
and unit structs all produce a compile-time error (lib.rs:207–226).

### Field attributes (recognized by the derive)

| Attribute | What it controls |
|-----------|-----------------|
| `#[forward]` | Includes the field in `pack_forward` / `unpack_forward`. On unpack, the ghost value is **overwritten** with the owner's value. |
| `#[reverse]` | Includes the field in `pack_reverse` / `unpack_reverse`. On unpack, the owner **accumulates** (`+=`) contributions from the ghost. |
| `#[zero]` | Includes the field in `zero()`. The field is resized to `n` atoms and filled with `0.0` each step. |

Attributes may be combined freely. A field with no attribute is still included in
migration (`pack`/`unpack`), `truncate`, `swap_remove`, and `apply_permutation`,
but never communicated to or from ghosts (lib.rs:143–154, `FieldInfo` struct).

`#[reverse]` does NOT imply `#[zero]` behavior — they are **independent flags**
(lib.rs:147–150: `is_reverse` and `is_zero` are separate booleans). Pairing them
is a strong convention (`#[reverse]` accumulators must start at zero each step),
not a mechanical guarantee. Failing to pair them is a silent logic bug in parallel
runs.

Source: `lib.rs:136–154` (attribute parsing), `lib.rs:392–465` (comm methods),
`lib.rs:468–500` (zero method).

### Field type constraints

Every field must be one of:

- `Vec<f64>` — one `f64` per atom (scalar column)
- `Vec<[f64; N]>` — N `f64`s per atom (fixed-size array column), N must be a
  literal integer

Any other type produces a compile-time error naming the offending field
(`lib.rs:232–243`). The type classifier is `classify_field` (`lib.rs:97–134`),
which matches only the exact forms above — no type aliases, no newtype wrappers.

### Generated methods (with signatures)

The derive emits an `impl AtomData for #name { ... }` block. The following
methods are always generated (for all structs):

```rust
fn as_any(&self) -> &dyn std::any::Any
fn as_any_mut(&mut self) -> &mut dyn std::any::Any

fn truncate(&mut self, n: usize)
fn swap_remove(&mut self, i: usize)

fn pack(&self, i: usize, buf: &mut Vec<f64>)
fn unpack(&mut self, buf: &[f64]) -> usize

fn apply_permutation(&mut self, perm: &[usize], n: usize)
```

The following methods are generated **only if at least one field has the
corresponding attribute** (lib.rs:401–404 and lib.rs:469 guard on empty selection):

```rust
// Only if any field is #[forward]:
fn pack_forward(&self, i: usize, buf: &mut Vec<f64>)
fn unpack_forward(&mut self, i: usize, buf: &[f64]) -> usize
fn forward_comm_size(&self) -> usize

// Only if any field is #[reverse]:
fn pack_reverse(&self, i: usize, buf: &mut Vec<f64>)
fn unpack_reverse(&mut self, i: usize, buf: &[f64]) -> usize
fn reverse_comm_size(&self) -> usize

// Only if any field is #[zero]:
fn zero(&mut self, n: usize)
```

The derive does **NOT** generate:
- `new()` or `Default` — the user must build the value explicitly
  (`lib.rs:17–18` docstring: "the derive does *not* generate a `new()` or
  `Default`")
- `pack_all_for_restart` / `unpack_all_from_restart` — mentioned in
  `SOIL_ATOMDATA_CONTRACT.md:94` as part of the lifecycle but **absent from the
  generated code** (see Doc gaps below)
- Any field accessor, getter, or setter

Source: `lib.rs:319–356` (the `quote!` block assembling the final impl).

#### Semantics of key methods

**`pack(i, buf)`** — pushes all fields for atom `i` into `buf` in declaration
order. Scalar fields push one `f64`; array fields use `extend_from_slice`
(lib.rs:271–281).

**`unpack(buf) -> usize`** — reads from compile-time fixed offsets (not a cursor)
and pushes one new atom-entry onto each field vector. Returns the total number of
`f64`s consumed per atom (lib.rs:360–380). Offsets are computed at macro
expansion time by `build_unpack_stmts`.

**`unpack_forward(i, buf) -> usize`** — overwrites `self.field[i]` with values
from `buf`. For arrays: `self.field[i] = [buf[off], buf[off+1], ...]`
(lib.rs:426–427).

**`unpack_reverse(i, buf) -> usize`** — accumulates: scalar `self.field[i] +=
buf[off]`, array element-wise `self.field[i][j] += buf[off+j]` (lib.rs:419,
lib.rs:430–435).

**`zero(n)`** — for each `#[zero]` field: `resize(n, zero_value)` followed by
`fill(zero_value)`. This means the resize and fill are redundant for elements
0..old_len but guarantees correctness for both growth and shrink
(lib.rs:481–491).

**`apply_permutation(perm, n)`** — builds a scratch vector by collecting
`perm.iter().map(|&p| self.field[p])`, then copies back into `self.field[..n]`.
Allocates one scratch Vec per field per call (lib.rs:288–308).

---

## Config/TOML schema

None. `soil_derive` is a proc-macro crate with no runtime configuration,
no TOML schema, and no feature flags in Cargo.toml. The field attributes
(`#[forward]`, `#[reverse]`, `#[zero]`) are the only configuration surface and
they live in Rust source.

---

## Key behaviors, invariants, and gotchas

1. **Declaration order is the wire layout.** `pack`/`unpack` serialize fields in
   struct declaration order. This is also the migration and restart-file layout.
   Adding, removing, or reordering a field is a breaking format change. Treat the
   field list as a versioned wire format. Source: `lib.rs:39–43` (module docstring),
   `docs/src/tutorial/field-attributes.md:45–51`.

2. **`#[reverse]` does NOT imply `#[zero]`.** The two flags are parsed
   independently (`lib.rs:247–250`). A `#[reverse]` field without `#[zero]` is
   never zeroed — contributions accumulate across steps, which is almost always a
   bug. The docs call this pairing "almost always" required
   (`docs/src/tutorial/field-attributes.md:36–37`), but the macro enforces nothing.

3. **No constructor is generated.** The derive generates zero initialization
   helpers. Users must build the struct literal themselves and pass it to
   `register_atom_data!`. Source: `lib.rs:17–18`. The tutorial reflects this:
   `register_atom_data!(app, SoftAtom { radius: Vec::new() })`
   (`docs/src/tutorial/write-your-own-physics.md:132`).

4. **`pack_forward`/`pack_reverse` etc. are absent when no field has the
   attribute.** If a struct has no `#[forward]` fields, `pack_forward`,
   `unpack_forward`, and `forward_comm_size` are not emitted at all. The trait
   must provide defaults for these or compilation will fail. Source: `lib.rs:401–404`.

5. **`zero()` resizes then fills.** The generated `zero` calls
   `self.field.resize(n, 0.0)` followed immediately by `self.field[..n].fill(0.0)`
   (`lib.rs:481–484`). The fill after resize is redundant for atoms that already
   existed but correct in all cases. New atoms added via resize are guaranteed to
   be zeroed; this is the only method that grows field vectors.

6. **`unpack` uses compile-time fixed offsets, not a cursor.** The unpack offsets
   are computed at macro-expansion time by accumulating sizes in
   `build_unpack_stmts` (`lib.rs:360–380`). The generated code is a series of
   `self.field.push(buf[N])` statements with literal indices, not a runtime
   cursor. This is safe but means adding a field in the middle of an existing
   struct changes all subsequent offsets — another facet of the wire-format
   stability constraint.

7. **`apply_permutation` allocates per field.** Each field requires a scratch Vec
   allocation on every permutation call (`lib.rs:288–308`). For structs with many
   large fields this can be significant. A hand-written impl can avoid this.

8. **Type aliases are not recognized.** `classify_field` checks for the literal
   path `Vec` with one angle-bracketed argument (`lib.rs:98–133`). A type alias
   like `type Column = Vec<f64>` will fail with a compile error.

9. **`forward_comm_size` / `reverse_comm_size` are computed at macro-expansion
   time.** They return `usize` literals baked into the generated code — they
   cannot vary at runtime (`lib.rs:450–451`, `lib.rs:462–463`).

---

## Tutorial outline

### Option A: declare a column using the derive (recommended path)

1. Add `soil_derive` to `[dependencies]`.
2. `use soil_derive::AtomData;`
3. Declare the struct with named `Vec<f64>` / `Vec<[f64; N]>` fields.
4. Annotate each field with `#[forward]`, `#[reverse]`, `#[zero]`, or none,
   using the decision tree in `docs/src/tutorial/field-attributes.md`.
5. In the plugin's `build()`, call
   `register_atom_data!(app, MyAtom { field: Vec::new(), ... });`
6. In systems, fetch via `registry.expect::<MyAtom>("...")`.

### Option B: implement `AtomData` by hand

Required only when field types are not `Vec<f64>` / `Vec<[f64; N]>` (e.g.,
`BondStore` with irregular layout). Reference: `docs/src/substrate/writing-atomdata.md`.
The hand-impl must keep `*_comm_size` in lockstep with the corresponding `pack_*`
method, and must honor the overwrite vs. accumulate semantics.

---

## Doc gaps

### 1. `SOIL_ATOMDATA_CONTRACT.md:94` mentions `pack_all_for_restart` / `unpack_all_from_restart` — the derive does NOT generate these

The contract document (`docs/SOIL_ATOMDATA_CONTRACT.md:94`) lists
`pack_all_for_restart` / `unpack_all_from_restart` as part of the lifecycle
("Plus `pack_all_for_restart` / `unpack_all_from_restart` (all fields, for
checkpointing)"). However, inspecting `lib.rs:319–356` (the final `quote!` block)
shows these methods are **not generated**. The expand block emits: `as_any`,
`as_any_mut`, `truncate`, `swap_remove`, `pack`, `unpack`, `apply_permutation`,
and the conditional forward/reverse/zero methods. There is no `pack_all_for_restart`
anywhere in `lib.rs`. Either:
- These methods exist as defaults on the `AtomData` trait in `soil_core` (which
  `pack`/`unpack` satisfy), or
- The contract document is ahead of the implementation.

This needs verification against `soil_core::atom` trait definition.

### 2. `SOIL_ATOMDATA_CONTRACT.md:64` uses `DemAtom::new()` — contradicts the "no constructor" guarantee

The contract document shows:
```rust
register_atom_data!(app, DemAtom::new());
```
(`docs/SOIL_ATOMDATA_CONTRACT.md:64`)

But `lib.rs:17–18` explicitly states the derive "does *not* generate a `new()` or
`Default`", and the tutorial at `docs/src/tutorial/write-your-own-physics.md:132`
correctly uses the struct-literal form:
```rust
register_atom_data!(app, SoftAtom { radius: Vec::new() });
```

The contract document's `DemAtom::new()` call will compile only if `DemAtom`
defines its own `new()` (which some hand-written types might), or it is a typo/
stale example. The contract doc should be updated to use the struct-literal form,
or add a note that `new()` must be provided separately.

### 3. `field-attributes.md` omits `forward_comm_size` / `reverse_comm_size` from its generated-methods list

`docs/src/tutorial/field-attributes.md:55–60` lists the generated methods but
omits `forward_comm_size`, `reverse_comm_size`, and `as_any`/`as_any_mut`, which
the README (`crates/soil_derive/README.md:42–46`) correctly includes. The tutorial
list should be made complete or cross-referenced to the README.

### 4. No dedicated `soil_derive` reference page in the book

The SUMMARY (`docs/src/SUMMARY.md`) has no entry for `soil_derive` — it appears
only within the AtomData contract and the tutorial prose. Given that the derive is
the primary user-facing API for extending the substrate, it warrants its own
reference page (see Suggested Placement below).

---

## Suggested placement

```
# Reference
- [Substrate Internals](./reference/internals.md)
- [Crate Map](./reference/crates.md)
+ - [soil_derive: AtomData Derive](./reference/soil-derive.md)   ← new
```

The new page should cover:
- The derive macro declaration (single entry point)
- Field attribute table with direction/semantics/example
- Complete generated-method table with signatures (including the conditional ones
  and `as_any`/`as_any_mut`)
- The "no constructor" statement prominently
- Field type constraints and the type-alias gotcha
- Wire-format stability (declaration order = restart layout)
- The `#[reverse]` ⊆ pair-with-`#[zero]` convention vs. the macro's lack of enforcement
- Link to `writing-atomdata.md` for hand-impl path

The tutorial pages (`write-your-own-physics.md`, `field-attributes.md`) should
remain as is — they are the narrative path. This reference page is the look-up
target for "what exactly does the derive generate."
