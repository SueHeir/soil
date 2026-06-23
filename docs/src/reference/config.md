# Config Reference

Every TOML section the substrate reads, with types and defaults. Each section is
deserialized with `#[serde(deny_unknown_fields)]`, so an unknown or misspelled
key is a hard error at parse time rather than a silently ignored setting.

The physics-tier sections (`[deform]`, `[[pin]]`) are documented on their own
pages — [Box Deformation](../substrate/deformation.md) and [Fixes](../substrate/fixes.md) —
and the output sections (`[thermo]`, `[dump]`, `[restart]`, `[vtp]`) live in
`soil_print`. This page covers the core substrate: `[comm]`, `[domain]`,
`[neighbor]`, regions, and groups.

## `[comm]` — processor grid

Splits the box across MPI ranks as a uniform Cartesian grid. Read once, on the
first `[[run]]` stage.

| key | type | default | meaning |
|---|---|---|---|
| `processors_x` | int | `1` | MPI ranks along x |
| `processors_y` | int | `1` | MPI ranks along y |
| `processors_z` | int | `1` | MPI ranks along z |

```toml
[comm]
processors_x = 2
processors_y = 2
processors_z = 1
```

The product `processors_x · processors_y · processors_z` must equal the total MPI
rank count.

## `[domain]` — box extents and boundaries

The global simulation box and its boundary conditions. Read once, on the first
stage; later stages may deform the box but do **not** re-read this section.

| key | type | default | meaning |
|---|---|---|---|
| `x_low` | f64 | `0.0` | low x corner (alias `x_lo`) |
| `x_high` | f64 | `1.0` | high x corner (alias `x_hi`) |
| `y_low` / `y_high` | f64 | `0.0` / `1.0` | y corners (aliases `y_lo` / `y_hi`) |
| `z_low` / `z_high` | f64 | `0.0` / `1.0` | z corners (aliases `z_lo` / `z_hi`) |
| `boundary_x` | string | `"periodic"` | `"periodic"`, `"fixed"`, or `"shrink-wrap"` |
| `boundary_y` | string | `"periodic"` | as above |
| `boundary_z` | string | `"periodic"` | as above |
| `shrink_wrap_padding` | f64 | `0.0` | padding for shrink-wrap faces; `0.0` = auto (use `ghost_cutoff`) |

```toml
[domain]
x_low = 0.0
x_high = 0.1
y_low = 0.0
y_high = 0.1
z_low = 0.0
z_high = 0.2
boundary_x = "periodic"
boundary_y = "periodic"
boundary_z = "fixed"
```

**Shrink-wrap is single-process only** — combining it with MPI panics at startup.
The triclinic (tilted) box and its `xy` tilt are not set here; they are driven at
runtime by `soil_deform`'s Lees–Edwards shear, which writes `domain.tilt`,
`domain.triclinic`, and `domain.boundary_vel` (see
[Box Deformation](../substrate/deformation.md)).

## `[neighbor]` — neighbor list

Controls pair-list construction and the rebuild policy.

| key | type | default | meaning |
|---|---|---|---|
| `skin_fraction` | f64 | `1.0` | pair-cutoff multiplier: `cutoff = (r_i + r_j) · skin_fraction` |
| `bin_size` | f64 | `1.0` | minimum bin edge length; auto-increased if MPI needs it |
| `every` | int | `0` | rebuild every N steps; `0` = displacement-based only |
| `check` | bool | `true` | when `every > 0`, also rebuild on displacement (hybrid) |
| `sort_every` | int | `1000` | spatial re-sort interval for cache locality; `0` = disabled |
| `newton` | bool | `true` | `true` = half list + reverse comm; `false` = full list, no reverse comm |

```toml
[neighbor]
skin_fraction = 1.2
every = 0
newton = true
```

`skin_fraction` feeds the skin / cutoff / ghost-cutoff chain documented in
[Substrate Internals](./internals.md): a larger skin means fewer rebuilds but a
wider ghost halo. The `newton` flag changes how a force system must write ghost
forces — see the `if newton || j < nlocal` guard in the internals chapter.

## Regions

A region is a spatial primitive, written inline as a TOML table tagged by
`type`. Regions are used to place atoms, select groups, and define constraints.
All `type` names and field names are lowercase.

| `type` | fields | shape |
|---|---|---|
| `block` | `min = [x,y,z]`, `max = [x,y,z]` | axis-aligned box |
| `sphere` | `center = [x,y,z]`, `radius` | ball |
| `cylinder` | `center = [a,b]`, `radius`, `axis = "x"\|"y"\|"z"`, `lo`, `hi` | finite cylinder along `axis` |
| `cone` | `center = [a,b]`, `axis`, `rad_lo`, `rad_hi`, `lo`, `hi` | truncated cone (frustum), tapering linearly from `rad_lo` at `lo` to `rad_hi` at `hi` |
| `plane` | `point = [x,y,z]`, `normal = [nx,ny,nz]` | positive-side halfspace |
| `union` | `regions = [ … ]` | OR of sub-regions |
| `intersect` | `regions = [ … ]` | AND of sub-regions |

For `cylinder` and `cone`, `center` is the 2D center in the plane perpendicular
to `axis`, and `lo`/`hi` are the extents *along* the axis.

```toml
{ type = "block",     min = [0,0,0], max = [5,5,5] }
{ type = "sphere",    center = [0,0,0], radius = 2.0 }
{ type = "cylinder",  center = [0,0], radius = 1.0, axis = "z", lo = 0.0, hi = 5.0 }
{ type = "cone",      center = [0,0], axis = "z", rad_lo = 0.004, rad_hi = 0.001, lo = 0.0, hi = 0.01 }
{ type = "plane",     point = [0,0,0], normal = [0,0,1] }
{ type = "intersect", regions = [ { type = "sphere", center = [0,0,0], radius = 3.0 },
                                  { type = "plane",  point = [0,0,0], normal = [0,0,1] } ] }
```

## Groups

A `[[group]]` is a named subset of atoms, selected by atom type, a region, or
both. The built-in `"all"` group always exists and cannot be redefined.

```toml
[[group]]
name = "mobile"                                               # required
type = [1, 2]                                                 # optional: atom_type filter
region = { type = "block", min = [0,0,0], max = [5,5,5] }     # optional
dynamic = false                                               # optional
```

| key | type | required | meaning |
|---|---|---|---|
| `name` | string | yes | group name (also the TOML key for `[[pin]]` etc.) |
| `type` | `[int]` | no | atom-type filter; membership requires a matching type |
| `region` | region | no | spatial filter; membership requires being inside |
| `dynamic` | bool | no | rebuild membership every step |

Membership is the **AND** of the type and region filters. The `dynamic` default is
smart: a group is dynamic if it has a region, static otherwise. A **static** group
(type-only, no region) locks its mask at the first build and skips the per-step
rebuild; a **dynamic** group rebuilds every step at `PreForce`, after ghosts are
fresh, so its mask reflects start-of-force-phase positions. A later `[[run]]`
stage can redefine a group by name.
