//! Core crate for SOIL.
//!
//! Contains all resource types, plugins, and systems for the base simulation
//! framework. DEM-specific extensions (`DemAtom`, force models) live in
//! separate crates (`dirt_atom`, `dirt_granular`).
//!
//! See `examples/minimal_sim.rs` for a runnable serial assembly of the four
//! substrate plugins.
//!
//! # Architecture
//!
//! ## Struct-of-arrays + the parallel-array invariant
//!
//! Per-atom state is struct-of-arrays: [`Atom`] holds one `Vec` per field
//! (`pos`, `vel`, `force`, `mass`, …), and each physics extension registered in
//! the [`AtomDataRegistry`] holds its own per-atom `Vec`s. **Every one of these
//! vectors is indexed by the same atom index `i`.** The invariant the whole
//! crate rests on: any *structural* operation must be applied identically across
//! the base [`Atom`] **and** every registered extension, so the arrays never
//! desync. The structural ops are [`swap_remove`](Atom::swap_remove) (remove one
//! atom), `apply_permutation` (spatial re-sort), and `truncate` (drop ghosts) —
//! mirrored on the registry by
//! [`swap_remove_all`](AtomDataRegistry::swap_remove_all),
//! [`apply_permutation_all`](AtomDataRegistry::apply_permutation_all), and
//! [`truncate_all`](AtomDataRegistry::truncate_all). A new extension gets this
//! for free from `#[derive(AtomData)]`.
//!
//! ## Local atoms and ghosts
//!
//! The arrays are partitioned: indices `0..nlocal` are **local** atoms this rank
//! owns; `nlocal..nlocal+nghost` are **ghosts** — read-only halo copies of atoms
//! owned by neighboring ranks (or periodic images). Force systems iterate local
//! atoms and may read ghost neighbors; ghosts are discarded and rebuilt on each
//! full-rebuild step.
//!
//! ## The two-path timestep ([`CommState`])
//!
//! Each step takes one of two paths, selected by [`CommState`]:
//!
//! - [`CommState::FullRebuild`] — the heavy path: apply PBC, migrate atoms that
//!   crossed a subdomain boundary (exchange), rebuild the ghost halo (borders),
//!   and rebuild the neighbor list.
//! - [`CommState::CommunicateOnly`] — the light path: skip migration and
//!   rebuilds, just forward-communicate fresh ghost positions.
//!
//! Within a step the systems run in the canonical order defined by
//! [`ParticleSimScheduleSet`]: `Setup` → `Pre/Initial/PostInitialIntegration`
//! (first Verlet half-kick + drift; force zeroing happens here at
//! `PostInitialIntegration`) → `Pre/Exchange` → `Pre/Neighbor` →
//! `Pre/Force/PostForce` → `Pre/Final/PostFinalIntegration` (second half-kick +
//! output). Forward comm replicates `#[forward]` state onto ghosts before
//! forces; reverse comm sums `#[reverse]` ghost contributions back to owners
//! after.
//!
//! ## The skin / cutoff / ghost-cutoff chain
//!
//! Neighbor detection and halo width are linked. The pairwise interaction
//! **cutoff** comes from the per-atom `cutoff_radius`. The neighbor **skin**
//! (`Neighbor::skin_fraction`) pads that cutoff so the pair list stays valid for
//! several steps without a rebuild. The **ghost cutoff** (`Neighbor::ghost_cutoff`,
//! mirrored on `Domain::ghost_cutoff`) is the padded reach — it sets how far
//! across a subdomain boundary atoms must be replicated as ghosts so every local
//! atom sees all its real neighbors. If the ghost cutoff is smaller than the
//! skinned interaction cutoff, pairs are silently missed; the substrate derives
//! the ghost cutoff from the skinned cutoff so this stays consistent.
//!
//! Note: `CorePlugins` (a convenience plugin group that bundles input/comm/
//! domain/neighbor/run/output) is **not** defined here — it lives downstream in
//! `dirt_core`. To assemble a sim directly on the substrate, add the plugins
//! yourself: [`AtomPlugin`], [`DomainPlugin`], [`NeighborPlugin`],
//! [`CommunicationPlugin`] (plus `InputPlugin`/`RunPlugin` for the config + run
//! loop), as `examples/minimal_sim.rs` shows.

pub mod atom;
pub mod bond;
pub mod comm;
pub mod domain;
pub mod group;
pub mod input;
pub mod region;
pub mod schedule;
pub mod neighbor;
pub mod precision;
pub mod virial;

/// Internal state controlling the communication/rebuild path each timestep.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum CommState {
    /// Full rebuild: run pbc, exchange, full borders, neighbor build.
    #[default]
    FullRebuild,
    /// Lightweight path: skip pbc/exchange, only forward-comm ghost positions.
    CommunicateOnly,
}

// Re-export all public types at crate root for convenience.
pub use atom::*;
pub use bond::*;
pub use comm::*;
pub use domain::*;
pub use group::{group_includes, Group, GroupDef, GroupPlugin, GroupRegistry};
pub use input::{load_toml, print_banner, Config, Input, InputPlugin};
pub use region::{Axis, Region, SurfaceResult};

// Multi-stage run machinery now lives in grass_io. Re-exported so
// existing SOIL downstream code keeps writing `soil_core::RunPlugin`
// / `Res<RunConfig>` etc. unchanged.
pub use grass_io::{
    set_stage_name, run_read_input, update_cycle, validate_stages, FirstStageOnlyConfigs,
    RunConfig, RunPlugin, RunSchedule, RunState, StageConfig, StageOverrides, RUN_NAMESPACE,
};
// `deep_merge` is also re-exported in case downstream code uses it.
pub use grass_io::deep_merge;

pub use schedule::*;
pub use neighbor::*;
pub use precision::{Accum, Real};
pub use virial::*;

// Re-export toml so downstream users can build Config tables programmatically.
pub use toml;
