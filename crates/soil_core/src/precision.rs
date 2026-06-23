//! Compile-time precision selection for the simulation.
//!
//! SOIL can be built in three precision configs, selected by a single
//! mutually-exclusive cargo feature. This mirrors the LAMMPS accelerator-package
//! design of two numeric types: `numtyp` (compute/storage) and `acctyp`
//! (accumulation). Here they are [`Real`] and [`Accum`].
//!
//! | feature             | [`Real`] | [`Accum`] | use                                   |
//! |---------------------|----------|-----------|---------------------------------------|
//! | `precision-double`  | `f64`    | `f64`     | CPU reference / long runs             |
//! | `precision-mixed`   | `f32`    | `f64`     | fast CPU; matches GPU storage (default) |
//! | `precision-single`  | `f32`    | `f32`     | pure f32; the only thing Apple GPUs do |
//!
//! ## Why two types
//!
//! [`Real`] is the bandwidth-sensitive, per-atom compute/storage type
//! (positions, velocities). [`Accum`] is the drift-sensitive accumulation type
//! (force sums, energy, virial). "Mixed" computes in `f32` but accumulates in
//! `f64`, getting near-single-precision speed without summation drift.
//!
//! ## What does NOT change with precision
//!
//! Wire and persistence formats stay `f64` regardless of config: MPI pack/unpack
//! buffers, `grass_mpi` reductions, and restart files. We convert `Real`/`Accum`
//! <-> `f64` only at those boundaries. This keeps the external `grass` crates
//! untouched and lets a restart file round-trip losslessly across precision
//! builds.

// Exactly one precision feature must be active. The build system defaults to
// `precision-mixed`; selecting two (or none) is a configuration error.
#[cfg(any(
    all(feature = "precision-double", feature = "precision-mixed"),
    all(feature = "precision-double", feature = "precision-single"),
    all(feature = "precision-mixed", feature = "precision-single"),
))]
compile_error!(
    "exactly one of the `precision-double`, `precision-mixed`, `precision-single` \
     features may be enabled at a time"
);

#[cfg(not(any(
    feature = "precision-double",
    feature = "precision-mixed",
    feature = "precision-single"
)))]
compile_error!(
    "one of the `precision-double`, `precision-mixed`, `precision-single` features \
     must be enabled (the default feature set enables `precision-mixed`)"
);

/// Per-atom compute/storage scalar (`numtyp`): positions, velocities, masses,
/// cutoffs, bin geometry. `f32` in mixed/single, `f64` in double.
#[cfg(feature = "precision-double")]
pub type Real = f64;
/// Per-atom compute/storage scalar (`numtyp`).
#[cfg(any(feature = "precision-mixed", feature = "precision-single"))]
pub type Real = f32;

/// Accumulation scalar (`acctyp`): force sums, energy, virial. `f64` in
/// double/mixed (drift-safe), `f32` in single.
#[cfg(any(feature = "precision-double", feature = "precision-mixed"))]
pub type Accum = f64;
/// Accumulation scalar (`acctyp`).
#[cfg(feature = "precision-single")]
pub type Accum = f32;
