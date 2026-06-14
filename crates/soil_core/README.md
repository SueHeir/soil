# soil_core

Core simulation infrastructure for SOIL: particle storage, MPI domain decomposition, communication, neighbor lists, and spatial regions. TOML loading, the multi-stage run loop, the App/Plugin framework, and the MPI abstraction live in the [grass](https://github.com/SueHeir/grass) workspace; this crate re-exports them so existing `use soil_core::{Config, RunPlugin, ...}` keeps working.

This crate is method-agnostic — it knows nothing about physics. DEM-specific extensions (`DemAtom`, contact and force models) live in the [dirt](https://github.com/SueHeir/dirt) crates.

## What It Does

Provides foundational systems for particle-based simulations:

- **Per-atom storage** (`Atom`, `AtomData` trait): struct-of-arrays with extensible per-atom fields registered by plugins via the `AtomDataRegistry`, including MPI forward/reverse/exchange pack-unpack and restart serialization
- **Domain decomposition & boundaries** (`Domain`, `DomainConfig`, `BoundaryType`): box geometry, MPI splitting, periodic/fixed/shrink-wrap conditions
- **Communication wiring** (`CommunicationPlugin`): ghost/border exchange and forward/reverse comm on top of `grass_mpi::CommBackend` (single-process fallback included)
- **Neighbor lists** (`NeighborPlugin`): cell-list construction of pairwise neighbors
- **Bond topology** (`BondStore`, `BondPlugin`): per-atom bond lists with 1-2/1-3 pair exclusions
- **Groups** (`Group`, `GroupRegistry`, `GroupPlugin`): named atom subsets defined over regions
- **Spatial regions** (`Region`): Block, Sphere, Cylinder, Cone, Plane, Union, Intersect with point-containment, random-point, and surface-distance queries
- **Virial stress** (`VirialStress`, `VirialStressPlugin`): per-pair accumulation of the stress tensor
- **Schedule phases** (`ParticleSimScheduleSet`): ordered system sets for the per-step particle loop
- **Re-exports from grass:** `Config`, `InputPlugin`, `RunPlugin`, `RunConfig`, `StageConfig`, `StageOverrides`, `ScheduleSetupSet`, `CommBackend`, etc.

## Key Types

| Type | Purpose | Defined in |
|------|---------|-----------|
| `Atom` | Core per-atom fields in struct-of-arrays layout | `soil_core` |
| `AtomData` | Trait to register plugin-specific data (e.g., `DemAtom`) | `soil_core` |
| `AtomDataRegistry` | Manages atom extensions with MPI pack/unpack | `soil_core` |
| `Domain` | Box geometry, bounds, periodicity | `soil_core` |
| `Region` | Spatial primitives for groups and insertion | `soil_core` |
| `GroupRegistry` | Named atom subsets resolved from regions | `soil_core` |
| `BondStore` | Per-atom bond topology and pair exclusions | `soil_core` |
| `VirialStress` | Accumulated virial stress tensor | `soil_core` |
| `CommBackend` | Abstraction over MPI or serial communication | `grass_mpi` (re-exported) |
| `Config` | TOML table with typed deserialization | `grass_io` (re-exported) |
| `RunConfig` / `StageConfig` | Multi-stage run + per-stage overrides | `grass_io` (re-exported) |

## Quick Start

`soil_core` is the substrate; a runnable simulation is assembled by a physics
tier such as [dirt_core](https://github.com/SueHeir/dirt):

```rust
use dirt_core::prelude::*;

let mut app = App::new();
app.add_plugins(CorePlugins)
   .add_plugins(GranularDefaultPlugins);
app.run();
```

Plugins register per-atom data through the `AtomDataRegistry`; use `#[derive(AtomData)]` (from `soil_derive`) to extend `Atom` with custom fields.

## Features

- `mpi_backend` (default): Enable MPI; disable for serial-only builds

## License

MIT OR Apache-2.0
