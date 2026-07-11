//! Per-atom data storage and the [`AtomData`] extension trait.
//!
//! This module provides:
//! - [`Atom`]: struct-of-arrays storage for core per-atom fields (position,
//!   velocity, force, mass, etc.)
//! - [`AtomData`]: trait for plugin-specific per-atom extensions (e.g. radius,
//!   angular velocity) with pack/unpack support for MPI communication
//! - [`AtomDataRegistry`]: dynamic registry that manages all [`AtomData`]
//!   extensions, keyed by `TypeId`
//! - [`AtomPlugin`]: registers the [`Atom`] and [`AtomDataRegistry`] resources
//!   and the per-step force zeroing system
//!
//! # Writing an `AtomData` extension
//!
//! A physics tier attaches its per-particle state by implementing [`AtomData`]
//! and registering an instance with [`register_atom_data!`](crate::register_atom_data). Almost always you
//! get the implementation for free from `#[derive(AtomData)]` (in `soil_derive`);
//! implement the trait by hand only for irregular layouts such as
//! [`BondStore`](crate::BondStore).
//!
//! ## What you must provide vs. what defaults
//!
//! **Mandatory** (every extension must keep its columns consistent under
//! structural ops): [`as_any`](AtomData::as_any) /
//! [`as_any_mut`](AtomData::as_any_mut), [`snapshot`](AtomData::snapshot),
//! [`truncate`](AtomData::truncate),
//! [`swap_remove`](AtomData::swap_remove), [`pack`](AtomData::pack) /
//! [`unpack`](AtomData::unpack) (migration + restart), and
//! [`apply_permutation`](AtomData::apply_permutation).
//!
//! **Defaulted** (no-ops unless your state needs them): the forward pass
//! ([`pack_forward`](AtomData::pack_forward) /
//! [`unpack_forward`](AtomData::unpack_forward) /
//! [`forward_comm_size`](AtomData::forward_comm_size)), the reverse pass
//! ([`pack_reverse`](AtomData::pack_reverse) /
//! [`unpack_reverse`](AtomData::unpack_reverse) /
//! [`reverse_comm_size`](AtomData::reverse_comm_size)), and
//! [`zero`](AtomData::zero).
//!
//! ## The two contracts that make parallel runs correct
//!
//! - **forward = overwrite, reverse = accumulate.** A `#[forward]` field is
//!   read-only replicated state a neighbor needs (radius, omega, temperature):
//!   the owner's value is *copied over* the ghost's each exchange. A `#[reverse]`
//!   field is a contribution computed on a ghost that must reach the owner
//!   (force, torque, heat flux): `unpack_reverse` must `+=` into the owner, and
//!   the field should also be `#[zero]` so it starts each step at zero.
//! - **`*_comm_size` must equal the number of `f64`s the matching `pack_*`
//!   pushes per atom**, or the buffer striding desyncs and ghosts read garbage.
//!   The derive keeps these in lockstep; a hand-written impl must too.
//!
//! Field declaration order also defines the migration/restart wire layout, so
//! reordering fields is a restart-format break.

use std::{
    any::{Any, TypeId},
    cell::{Ref, RefCell, RefMut},
    panic::{catch_unwind, AssertUnwindSafe},
};

use crate::{Accum, CommState, ParticleSimScheduleSet, Real};
use grass_app::prelude::*;
use grass_scheduler::prelude::*;

/// Number of `f64`s packed/unpacked for one atom's base fields
/// (tag, origin_index, cutoff_radius, atom_type, pos×3, vel×3, force×3, mass, image×3).
pub const ATOM_PACK_SIZE: usize = 17;

// ── AtomData trait ───────────────────────────────────────────────────────────

/// Register an [`AtomData`] extension with the [`AtomDataRegistry`] stored in `app`.
///
/// Call this from a plugin's `build()`. From then on the substrate carries the
/// extension's columns through every migration, ghost exchange, permutation, and
/// restart automatically. Panics if [`AtomPlugin`] has not been added yet (i.e.
/// the [`AtomDataRegistry`] is missing).
///
/// # Example
///
/// In practice you derive [`AtomData`] (`#[derive(AtomData)]` from `soil_derive`)
/// rather than hand-writing it, but the registration call is identical:
///
/// ```no_run
/// use std::any::Any;
/// use grass_app::prelude::*;
/// use soil_core::{apply_permutation_in_place, register_atom_data, AtomData, AtomPlugin};
///
/// // A minimal extension (a real one would use `#[derive(AtomData)]`).
/// #[derive(Default)]
/// struct Charge { q: Vec<f64> }
/// impl AtomData for Charge {
///     fn as_any(&self) -> &dyn Any { self }
///     fn as_any_mut(&mut self) -> &mut dyn Any { self }
///     fn snapshot(&self) -> Box<dyn AtomData> { Box::new(Charge { q: self.q.clone() }) }
///     fn len(&self) -> usize { self.q.len() }
///     fn push_default(&mut self) { self.q.push(0.0); }
///     fn truncate(&mut self, n: usize) { self.q.truncate(n); }
///     fn swap_remove(&mut self, i: usize) { self.q.swap_remove(i); }
///     fn pack(&self, i: usize, buf: &mut Vec<f64>) { buf.push(self.q[i]); }
///     fn unpack(&mut self, buf: &[f64]) -> usize { self.q.push(buf[0]); 1 }
///     fn apply_permutation(&mut self, perm: &[usize], n: usize) {
///         let mut visited = Vec::new();
///         apply_permutation_in_place(&mut self.q, perm, n, &mut visited);
///     }
/// }
///
/// let mut app = App::new();
/// app.add_plugins(AtomPlugin);            // registers the AtomDataRegistry
/// register_atom_data!(app, Charge::default());
/// ```
#[macro_export]
macro_rules! register_atom_data {
    ($app:expr, $value:expr) => {
        let atom_count =
            if let Some(cell) = $app.get_mut_resource(::std::any::TypeId::of::<$crate::Atom>()) {
                cell.borrow()
                    .downcast_ref::<$crate::Atom>()
                    .expect("Failed to downcast Atom — this is a bug in SOIL")
                    .len()
            } else {
                panic!("Atom not found — AtomPlugin must be added first");
            };
        if let Some(cell) =
            $app.get_mut_resource(::std::any::TypeId::of::<$crate::AtomDataRegistry>())
        {
            let mut binder = cell.borrow_mut();
            binder
                .downcast_mut::<$crate::AtomDataRegistry>()
                .expect("Failed to downcast AtomDataRegistry — this is a bug in SOIL")
                .try_register($value, atom_count)
                .unwrap_or_else(|error| panic!("failed to register AtomData: {error}"));
        } else {
            panic!("AtomDataRegistry not found — AtomPlugin must be added first");
        }
    };
}

/// Apply `perm[0..n]` to `values[0..n]` in place, with `perm[new_i] = old_i`.
///
/// `visited` is caller-owned scratch. It is cleared and resized, but its capacity
/// is retained so spatial sorts can reuse the same allocation across every base
/// atom field and registered [`AtomData`] column.
pub fn apply_permutation_in_place<T: Clone>(
    values: &mut [T],
    perm: &[usize],
    n: usize,
    visited: &mut Vec<bool>,
) {
    assert!(n <= values.len(), "permutation length exceeds value slice");
    assert!(n <= perm.len(), "permutation length exceeds perm slice");

    visited.clear();
    visited.resize(n, false);

    for start in 0..n {
        if visited[start] {
            continue;
        }

        let mut current = start;
        let saved = values[start].clone();
        loop {
            visited[current] = true;
            let source = perm[current];
            assert!(source < n, "permutation index out of bounds");
            if source == start {
                values[current] = saved;
                break;
            }
            values[current] = values[source].clone();
            current = source;
        }
    }
}

/// Per-atom extension data (e.g. radius, density for DEM).
///
/// Implement this trait to attach custom per-atom fields to the simulation.
/// The registry handles pack/unpack for MPI ghost and exchange communication
/// automatically. Use the `#[derive(AtomData)]` macro from `soil_derive` for
/// simple struct-of-Vec extensions; implement manually only for complex layouts
/// (like [`BondStore`](crate::BondStore)).
pub trait AtomData: Any {
    /// Upcast to `&dyn Any` for downcasting in the registry.
    fn as_any(&self) -> &dyn Any;
    /// Upcast to `&mut dyn Any` for mutable downcasting in the registry.
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Capture an independent copy for an all-or-nothing particle-store
    /// operation. Implementations must preserve every row and any additional
    /// state which affects later structural operations.
    ///
    /// This is separate from the migration wire format: rollback must not
    /// depend on calling a decoder after that decoder has failed.
    fn snapshot(&self) -> Box<dyn AtomData>;
    /// Number of rows currently stored. This must equal the atom-store length.
    fn len(&self) -> usize;
    /// Whether this extension has no rows.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Append one default row, used when an extension is registered late.
    fn push_default(&mut self);
    /// Reserve capacity for future rows. This is a capacity hint only; it does
    /// not change the extension's row count. The default keeps custom stores
    /// source-compatible.
    fn reserve(&mut self, _additional: usize) {}
    /// Validate the row layout before registration.
    ///
    /// Single-column implementations can use the default. A manual store with
    /// multiple per-atom columns must override this method and check each
    /// column; [`AtomDataRegistry`] calls it before accepting the store.
    fn validate_rows(&self, expected: usize) -> Result<(), AtomDataRegistrationError> {
        let actual = self.len();
        if actual == expected {
            Ok(())
        } else {
            Err(AtomDataRegistrationError::RowCountMismatch {
                type_name: std::any::type_name::<Self>(),
                expected,
                actual,
            })
        }
    }
    /// Shrink all per-atom Vecs to length `n`, discarding trailing entries.
    fn truncate(&mut self, n: usize);
    /// Remove atom at index `i` by swapping with the last element.
    fn swap_remove(&mut self, i: usize);
    /// Serialize all fields for atom `i` into `buf` (exchange/border comm).
    fn pack(&self, i: usize, buf: &mut Vec<f64>);
    /// Deserialize one atom's fields from `buf`, appending to internal Vecs.
    /// Returns the number of `f64`s consumed.
    fn unpack(&mut self, buf: &[f64]) -> usize;
    /// Reorder atoms according to `perm[0..n]` (used by spatial sorting).
    fn apply_permutation(&mut self, perm: &[usize], n: usize);

    /// Pack forward-comm fields (e.g. omega for DEM) into buf.
    fn pack_forward(&self, _i: usize, _buf: &mut Vec<f64>) {}
    /// Unpack forward-comm fields; returns number of f64s consumed.
    fn unpack_forward(&mut self, _i: usize, _buf: &[f64]) -> usize {
        0
    }
    /// Pack reverse-comm fields (e.g. torque for DEM) into buf.
    fn pack_reverse(&self, _i: usize, _buf: &mut Vec<f64>) {}
    /// Unpack reverse-comm fields; returns number of f64s consumed.
    fn unpack_reverse(&mut self, _i: usize, _buf: &[f64]) -> usize {
        0
    }
    /// Zero per-step accumulators (e.g. torque) for atoms 0..n.
    fn zero(&mut self, _n: usize) {}
    /// Reorder atoms with caller-owned cycle scratch.
    ///
    /// Implementations should use this when possible so spatial sorting can reuse
    /// one scratch allocation across all registered extensions. The default keeps
    /// older hand-written implementations source-compatible.
    fn apply_permutation_with_scratch(
        &mut self,
        perm: &[usize],
        n: usize,
        _visited: &mut Vec<bool>,
    ) {
        self.apply_permutation(perm, n);
    }
    /// Number of f64s per atom in forward comm.
    fn forward_comm_size(&self) -> usize {
        0
    }
    /// Number of f64s per atom in reverse comm.
    fn reverse_comm_size(&self) -> usize {
        0
    }
}

/// Why an [`AtomData`] extension could not be registered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtomDataRegistrationError {
    /// The extension type already has a store in the registry.
    Duplicate {
        /// Rust type name of the duplicate extension.
        type_name: &'static str,
    },
    /// A supplied extension did not have one row per atom.
    RowCountMismatch {
        /// Rust type name of the malformed extension.
        type_name: &'static str,
        /// Required atom-row count.
        expected: usize,
        /// Row count reported by the extension.
        actual: usize,
    },
    /// Default-row backfill did not produce the required number of rows.
    DefaultBackfillFailed {
        /// Rust type name of the malformed extension.
        type_name: &'static str,
        /// Required atom-row count.
        expected: usize,
        /// Row count after attempted backfill.
        actual: usize,
    },
}

impl std::fmt::Display for AtomDataRegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Duplicate { type_name } => write!(f, "AtomData type `{type_name}` is already registered"),
            Self::RowCountMismatch { type_name, expected, actual } => write!(f, "AtomData `{type_name}` has {actual} rows; expected {expected}"),
            Self::DefaultBackfillFailed { type_name, expected, actual } => write!(f, "AtomData `{type_name}` default backfill produced {actual} rows; expected {expected}"),
        }
    }
}

impl std::error::Error for AtomDataRegistrationError {}

/// A particle-store operation could not preserve the core/extension row
/// invariant.  Receiving is fallible because `AtomData::unpack` predates this
/// boundary and cannot return an error itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleStoreError {
    /// The core part of a received particle is incomplete.
    TruncatedCoreRecord,
    /// An extension panicked, over-read, or failed to append one aligned row.
    MalformedExtensionRecord,
    /// A caller addressed a ghost/nonexistent local row or supplied a non-bijection.
    InvalidStructuralOperation,
}

impl std::fmt::Display for ParticleStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedCoreRecord => {
                write!(f, "particle payload is shorter than a core record")
            }
            Self::MalformedExtensionRecord => write!(f, "particle extension record is malformed"),
            Self::InvalidStructuralOperation => {
                write!(f, "invalid particle-store structural operation")
            }
        }
    }
}

impl std::error::Error for ParticleStoreError {}

// ── AtomDataRegistry ─────────────────────────────────────────────────────────

/// Dynamic registry of [`AtomData`] extensions, keyed by `TypeId`.
pub struct AtomDataRegistry {
    stores: Vec<(TypeId, RefCell<Box<dyn AtomData>>)>,
    /// Indices into `stores` of extensions that actually carry forward-comm data
    /// (`forward_comm_size() > 0`). The per-step ghost forward comm iterates only
    /// these, so no-op stores (e.g. tangential history) cost neither a `RefCell`
    /// borrow nor a virtual `pack_forward`/`unpack_forward` dispatch per atom.
    /// Rebuilt on every [`register`](Self::register); stores never change after setup.
    forward_stores: Vec<usize>,
    /// Indices into `stores` with reverse-comm data (`reverse_comm_size() > 0`).
    reverse_stores: Vec<usize>,
}

impl Default for AtomDataRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomDataRegistry {
    /// Create an empty registry with no extensions registered.
    pub fn new() -> Self {
        AtomDataRegistry {
            stores: Vec::new(),
            forward_stores: Vec::new(),
            reverse_stores: Vec::new(),
        }
    }

    /// Register a new extension, enforcing one row for every current atom.
    /// Empty stores are default-filled; nonempty stores must already match.
    pub fn try_register<T: AtomData + 'static>(
        &mut self,
        mut data: T,
        atom_count: usize,
    ) -> Result<(), AtomDataRegistrationError> {
        let id = TypeId::of::<T>();
        for (existing_id, _) in &self.stores {
            if *existing_id == id {
                return Err(AtomDataRegistrationError::Duplicate {
                    type_name: std::any::type_name::<T>(),
                });
            }
        }
        if data.is_empty() {
            for _ in 0..atom_count {
                data.push_default();
            }
            match data.validate_rows(atom_count) {
                Ok(()) => {}
                Err(AtomDataRegistrationError::RowCountMismatch { actual, .. }) => {
                    return Err(AtomDataRegistrationError::DefaultBackfillFailed {
                        type_name: std::any::type_name::<T>(),
                        expected: atom_count,
                        actual,
                    });
                }
                Err(error) => return Err(error),
            }
        } else {
            data.validate_rows(atom_count)?;
        }
        self.stores.push((id, RefCell::new(Box::new(data))));
        self.rebuild_comm_caches();
        Ok(())
    }

    /// Recompute which stores carry forward / reverse comm data. Called at
    /// registration time (rare); the result is consumed by the per-atom comm
    /// hot path to skip no-op stores.
    fn rebuild_comm_caches(&mut self) {
        self.forward_stores = (0..self.stores.len())
            .filter(|&i| self.stores[i].1.borrow().forward_comm_size() > 0)
            .collect();
        self.reverse_stores = (0..self.stores.len())
            .filter(|&i| self.stores[i].1.borrow().reverse_comm_size() > 0)
            .collect();
    }

    /// Borrow a registered [`AtomData`] extension immutably, or `None` if not registered.
    pub fn get<T: AtomData + 'static>(&self) -> Option<Ref<'_, T>> {
        let id = TypeId::of::<T>();
        self.stores
            .iter()
            .find(|(tid, _)| *tid == id)
            .map(|(_, cell)| {
                Ref::map(cell.borrow(), |b| {
                    b.as_any()
                        .downcast_ref::<T>()
                        .expect("AtomData downcast failed — type mismatch (this is a bug in SOIL)")
                })
            })
    }

    /// Borrow a registered [`AtomData`] extension mutably, or `None` if not registered.
    pub fn get_mut<T: AtomData + 'static>(&self) -> Option<RefMut<'_, T>> {
        let id = TypeId::of::<T>();
        self.stores
            .iter()
            .find(|(tid, _)| *tid == id)
            .map(|(_, cell)| {
                RefMut::map(cell.borrow_mut(), |b| {
                    b.as_any_mut()
                        .downcast_mut::<T>()
                        .expect("AtomData downcast failed — type mismatch (this is a bug in SOIL)")
                })
            })
    }

    /// Borrow a registered [`AtomData`] extension immutably, panicking with
    /// `context` if it is not registered. Use when the extension is a hard
    /// requirement (its plugin should have registered it during setup).
    pub fn expect<T: AtomData + 'static>(&self, context: &str) -> Ref<'_, T> {
        self.get::<T>().unwrap_or_else(|| {
            panic!(
                "{}: '{}' not registered. Ensure the plugin is added.",
                context,
                std::any::type_name::<T>()
            )
        })
    }

    /// Borrow a registered [`AtomData`] extension mutably, panicking with
    /// `context` if it is not registered. Mutable counterpart of [`expect`](Self::expect).
    pub fn expect_mut<T: AtomData + 'static>(&self, context: &str) -> RefMut<'_, T> {
        self.get_mut::<T>().unwrap_or_else(|| {
            panic!(
                "{}: '{}' not registered. Ensure the plugin is added.",
                context,
                std::any::type_name::<T>()
            )
        })
    }

    /// Truncate all registered extensions to `n` atoms (used when removing ghosts).
    pub fn truncate_all(&self, n: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().truncate(n);
        }
    }

    /// Reserve capacity in each extension alongside the core SoA.
    pub fn reserve_all(&self, additional: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().reserve(additional);
        }
    }

    /// True when every extension has one row per core particle.
    pub fn validate_rows(&self, expected: usize) -> bool {
        self.stores
            .iter()
            .all(|(_, cell)| cell.borrow().len() == expected)
    }

    /// Capture every extension before a fallible structural operation.
    fn snapshot(&self) -> Vec<Box<dyn AtomData>> {
        self.stores
            .iter()
            .map(|(_, cell)| cell.borrow().snapshot())
            .collect()
    }

    /// Restore a captured extension snapshot without calling an extension's
    /// packing or decoding hooks.
    fn restore(&self, snapshot: Vec<Box<dyn AtomData>>) {
        assert_eq!(snapshot.len(), self.stores.len());
        for ((_, cell), saved) in self.stores.iter().zip(snapshot) {
            *cell.borrow_mut() = saved;
        }
    }

    /// Remove atom at index `i` from all registered extensions via swap-remove.
    pub fn swap_remove_all(&self, i: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().swap_remove(i);
        }
    }

    /// Pack all extension fields for atom `i` into `buf` (exchange/border comm).
    pub fn pack_all(&self, i: usize, buf: &mut Vec<f64>) {
        for (_, cell) in &self.stores {
            cell.borrow().pack(i, buf);
        }
    }

    /// Unpack one atom's extension fields from `buf`. Returns total `f64`s consumed.
    pub fn unpack_all(&self, buf: &[f64]) -> usize {
        let mut pos = 0;
        for (_, cell) in &self.stores {
            pos += cell.borrow_mut().unpack(&buf[pos..]);
        }
        pos
    }

    /// Decode one extension record while containing the legacy infallible
    /// callback boundary. A failed decoder is rolled back to `expected_rows`.
    fn unpack_one(&self, expected_rows: usize, buf: &[f64]) -> Result<usize, ParticleStoreError> {
        let decoded = catch_unwind(AssertUnwindSafe(|| self.unpack_all(buf)));
        match decoded {
            Ok(used) if used == buf.len() && self.validate_rows(expected_rows + 1) => Ok(used),
            _ => {
                self.truncate_all(expected_rows);
                Err(ParticleStoreError::MalformedExtensionRecord)
            }
        }
    }

    /// Reorder all extensions according to `perm[0..n]`.
    pub fn apply_permutation_all(&self, perm: &[usize], n: usize) {
        let mut visited = Vec::new();
        self.apply_permutation_all_with_scratch(perm, n, &mut visited);
    }

    /// Reorder all extensions with caller-owned cycle scratch.
    pub fn apply_permutation_all_with_scratch(
        &self,
        perm: &[usize],
        n: usize,
        visited: &mut Vec<bool>,
    ) {
        for (_, cell) in &self.stores {
            cell.borrow_mut()
                .apply_permutation_with_scratch(perm, n, visited);
        }
    }

    /// Pack forward-comm fields for atom `i`. Iterates only stores with
    /// forward-comm data (no-op stores contribute zero bytes anyway, so skipping
    /// them is byte-identical — it just drops the wasted borrow + dispatch).
    pub fn pack_forward_all(&self, i: usize, buf: &mut Vec<f64>) {
        for &si in &self.forward_stores {
            self.stores[si].1.borrow().pack_forward(i, buf);
        }
    }

    /// Unpack forward-comm fields for atom `i`. Returns total `f64`s consumed.
    /// Mirrors [`pack_forward_all`](Self::pack_forward_all) (same store set, same order).
    pub fn unpack_forward_all(&self, i: usize, buf: &[f64]) -> usize {
        let mut pos = 0;
        for &si in &self.forward_stores {
            pos += self.stores[si]
                .1
                .borrow_mut()
                .unpack_forward(i, &buf[pos..]);
        }
        pos
    }

    /// Pack reverse-comm fields (e.g. torque) for atom `i`. Iterates only stores
    /// with reverse-comm data.
    pub fn pack_reverse_all(&self, i: usize, buf: &mut Vec<f64>) {
        for &si in &self.reverse_stores {
            self.stores[si].1.borrow().pack_reverse(i, buf);
        }
    }

    /// Unpack reverse-comm fields for atom `i`. Returns total `f64`s consumed.
    /// Mirrors [`pack_reverse_all`](Self::pack_reverse_all) (same store set, same order).
    pub fn unpack_reverse_all(&self, i: usize, buf: &[f64]) -> usize {
        let mut pos = 0;
        for &si in &self.reverse_stores {
            pos += self.stores[si]
                .1
                .borrow_mut()
                .unpack_reverse(i, &buf[pos..]);
        }
        pos
    }

    /// Zero per-step accumulators (e.g. torque) in all extensions for atoms `0..n`.
    pub fn zero_all(&self, n: usize) {
        for (_, cell) in &self.stores {
            cell.borrow_mut().zero(n);
        }
    }

    /// Total number of `f64`s per atom in forward comm across all extensions.
    pub fn forward_comm_size(&self) -> usize {
        self.stores
            .iter()
            .map(|(_, cell)| cell.borrow().forward_comm_size())
            .sum()
    }

    /// Total number of `f64`s per atom in reverse comm across all extensions.
    pub fn reverse_comm_size(&self) -> usize {
        self.stores
            .iter()
            .map(|(_, cell)| cell.borrow().reverse_comm_size())
            .sum()
    }

    /// Pack all local atoms for restart file serialization.
    /// Returns one `Vec<f64>` per registered extension.
    pub fn pack_all_for_restart(&self, nlocal: usize) -> Vec<Vec<f64>> {
        self.stores
            .iter()
            .map(|(_, cell)| {
                let store = cell.borrow();
                let mut buf = Vec::new();
                for i in 0..nlocal {
                    store.pack(i, &mut buf);
                }
                buf
            })
            .collect()
    }

    /// Restore all extensions from restart file buffers (one buffer per extension).
    ///
    /// `unpack` is **push-based**: each call appends one atom's record to the
    /// store. By the time a restart is read the stores have already been sized
    /// to `nlocal` (empty per-atom slots) by `truncate_all` during setup, so we
    /// must clear each store to length 0 first — otherwise the unpacked atoms
    /// land at indices `[nlocal..2*nlocal)` and every atom reads a stale empty
    /// slot at `[i]`, silently dropping restored per-atom state (e.g. rotational
    /// `DemAtom` columns and per-contact tangential spring history).
    pub fn unpack_all_from_restart(&self, buffers: &[Vec<f64>]) {
        for ((_, cell), buf) in self.stores.iter().zip(buffers.iter()) {
            let mut store = cell.borrow_mut();
            store.truncate(0); // reset: push-based unpack must start from empty to stay index-aligned
            let mut pos = 0;
            while pos < buf.len() {
                pos += store.unpack(&buf[pos..]);
            }
        }
    }

    /// Decode an exact restart row count, rejecting under- and over-consumption.
    fn unpack_restart_rows(&self, expected_rows: usize, buffers: &[Vec<f64>]) -> bool {
        if buffers.len() != self.stores.len() {
            return false;
        }
        for ((_, cell), buf) in self.stores.iter().zip(buffers) {
            let mut store = cell.borrow_mut();
            store.truncate(0);
            let mut pos = 0;
            for _ in 0..expected_rows {
                let used = store.unpack(&buf[pos..]);
                if used == 0 || used > buf.len().saturating_sub(pos) {
                    return false;
                }
                pos += used;
            }
            if pos != buf.len() || store.len() != expected_rows {
                return false;
            }
        }
        self.validate_rows(expected_rows)
    }
}

// ── Atom Vec field macro ─────────────────────────────────────────────────────

/// Enumerates all per-atom Vec fields with their element types.
/// Pass a callback macro name; it receives the full list as
/// `(field, Type), ...` and can generate code uniformly.
#[macro_export]
macro_rules! for_each_atom_vec {
    ($callback:ident) => {
        $callback! {
            (tag, u32),
            (atom_type, u32),
            (origin_index, i32),
            (is_ghost, bool),
            (pos, [Real; 3]),
            (vel, [Real; 3]),
            (force, [Accum; 3]),
            (cutoff_radius, Real),
            (mass, Real),
            (inv_mass, Real),
            (image, [i32; 3]),
        }
    };
}

// ── Atom ──────────────────────────────────────────────────────────────────────

/// Struct-of-arrays storage for all per-atom fields (position, velocity, force, etc.).
#[derive(Clone)]
pub struct Atom {
    /// Total number of atoms across all ranks (global count).
    pub natoms: u64,
    /// Number of local atoms this rank owns; they occupy indices `0..nlocal`.
    pub nlocal: u32,
    /// Number of ghost atoms (halo copies); they occupy `nlocal..nlocal+nghost`.
    pub nghost: u32,

    /// Number of distinct atom types in the simulation.
    pub ntypes: usize,

    /// Integration timestep.
    pub dt: f64,

    /// Globally unique atom identifier (persists across migration/restart).
    pub tag: Vec<u32>,
    /// Per-atom type index in `0..ntypes`.
    pub atom_type: Vec<u32>,
    /// For a ghost, the local index of its owner atom on the sending rank
    /// (set during border packing so reverse comm can accumulate back).
    pub origin_index: Vec<i32>,
    /// `true` for ghost (halo) atoms, `false` for locally owned atoms.
    pub is_ghost: Vec<bool>,

    // Interleaved arrays: field[i] = [x, y, z]
    /// Position `[x, y, z]` per atom.
    pub pos: Vec<[Real; 3]>,
    /// Velocity `[vx, vy, vz]` per atom.
    pub vel: Vec<[Real; 3]>,
    /// Force accumulator `[fx, fy, fz]` per atom (zeroed each step before force systems).
    pub force: Vec<[Accum; 3]>,

    /// Per-atom interaction cutoff radius; feeds the neighbor skin/ghost-cutoff chain.
    pub cutoff_radius: Vec<Real>,
    /// Per-atom mass.
    pub mass: Vec<Real>,
    /// Cached reciprocal mass `1/mass` (kept in sync with `mass`; avoids per-step division).
    pub inv_mass: Vec<Real>,
    /// PBC image flags: number of times atom has crossed each periodic boundary.
    pub image: Vec<[i32; 3]>,
}

impl Default for Atom {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! impl_atom_new {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        /// Create an empty `Atom` store (no atoms; `ntypes = 1`, `dt = 1.0`).
        pub fn new() -> Self {
            Atom {
                natoms: 0,
                nlocal: 0,
                nghost: 0,
                ntypes: 1,
                dt: 1.0,
                $( $field: Vec::new(), )*
            }
        }
    };
}

macro_rules! impl_atom_swap_remove {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        /// Remove the atom at index `i` from every field array by swapping in the
        /// last element. **Must** be mirrored on the [`AtomDataRegistry`] via
        /// [`swap_remove_all`](AtomDataRegistry::swap_remove_all) to keep the
        /// parallel arrays aligned.
        pub fn swap_remove(&mut self, i: usize) {
            $( self.$field.swap_remove(i); )*
        }
    };
}

macro_rules! impl_atom_truncate {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        /// Drop all ghost atoms by truncating every field array to `nlocal`.
        /// Mirror on the registry with
        /// [`truncate_all`](AtomDataRegistry::truncate_all).
        pub fn truncate_to_nlocal(&mut self) {
            let n = self.nlocal as usize;
            $( self.$field.truncate(n); )*
        }
    };
}

macro_rules! impl_atom_truncate_to {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        fn truncate_to(&mut self, n: usize) {
            $( self.$field.truncate(n); )*
        }
    };
}

macro_rules! impl_atom_reserve {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        /// Reserve capacity for `additional` more atoms in every field array
        /// (avoids repeated reallocation when bulk-inserting).
        pub fn reserve(&mut self, additional: usize) {
            $( self.$field.reserve(additional); )*
        }
    };
}

macro_rules! impl_atom_apply_permutation {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        /// Reorder the first `n` atoms in every field array according to
        /// `perm` (spatial re-sort for cache locality). **Must** be mirrored on
        /// the registry via
        /// [`apply_permutation_all`](AtomDataRegistry::apply_permutation_all).
        pub fn apply_permutation(&mut self, perm: &[usize], n: usize) {
            let mut visited = Vec::new();
            self.apply_permutation_with_scratch(perm, n, &mut visited);
        }

        /// Reorder the first `n` atoms using caller-owned cycle scratch.
        pub fn apply_permutation_with_scratch(
            &mut self,
            perm: &[usize],
            n: usize,
            visited: &mut Vec<bool>,
        ) {
            $( apply_permutation_in_place(&mut self.$field, perm, n, visited); )*
        }
    };
}

impl Atom {
    for_each_atom_vec!(impl_atom_new);
    for_each_atom_vec!(impl_atom_swap_remove);
    for_each_atom_vec!(impl_atom_truncate);
    for_each_atom_vec!(impl_atom_truncate_to);
    for_each_atom_vec!(impl_atom_reserve);
    for_each_atom_vec!(impl_atom_apply_permutation);

    /// Total number of atoms (local + ghost) currently stored.
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// Returns true if no atoms are stored.
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// Dimension-indexed position access for comm.rs border detection.
    pub fn pos_component(&self, i: usize, dim: usize) -> f64 {
        self.pos[i][dim] as f64
    }

    /// Returns the highest global tag among all stored atoms, or 0 if empty.
    pub fn get_max_tag(&self) -> u32 {
        self.tag.iter().cloned().max().unwrap_or(0)
    }

    /// Shared packing logic: writes [`ATOM_PACK_SIZE`] `f64`s for atom `i` into `buf`,
    /// with caller-supplied origin_index, position offset, and velocity offset.
    ///
    /// `vel_offset` is the Lees–Edwards streaming-velocity jump applied to ghost
    /// copies that wrap across a moving (sheared) periodic face; it is `[0;3]` for
    /// ordinary ghosts and for exchanged (migrated) atoms.
    fn pack_atom_inner(
        &self,
        i: usize,
        origin_index_val: f64,
        pos_offset: [f64; 3],
        vel_offset: [f64; 3],
        buf: &mut Vec<f64>,
    ) {
        buf.push(self.tag[i] as f64);
        buf.push(origin_index_val);
        buf.push(self.cutoff_radius[i] as f64);
        buf.push(self.atom_type[i] as f64);
        buf.push(self.pos[i][0] as f64 + pos_offset[0]);
        buf.push(self.pos[i][1] as f64 + pos_offset[1]);
        buf.push(self.pos[i][2] as f64 + pos_offset[2]);
        buf.push(self.vel[i][0] as f64 + vel_offset[0]);
        buf.push(self.vel[i][1] as f64 + vel_offset[1]);
        buf.push(self.vel[i][2] as f64 + vel_offset[2]);
        buf.push(self.force[i][0] as f64);
        buf.push(self.force[i][1] as f64);
        buf.push(self.force[i][2] as f64);
        buf.push(self.mass[i] as f64);
        buf.push(self.image[i][0] as f64);
        buf.push(self.image[i][1] as f64);
        buf.push(self.image[i][2] as f64);
    }

    /// Pack atom `i` for MPI exchange (migration to another rank). Sets origin_index to 0.
    pub fn pack_exchange(&self, i: usize, buf: &mut Vec<f64>) {
        self.pack_atom_inner(i, 0.0, [0.0; 3], [0.0; 3], buf);
    }

    /// Pack atom `i` as a ghost for border communication, applying a periodic
    /// position shift and (for sheared periodic faces) a velocity shift.
    /// Stores the local index as `origin_index` so reverse comm can accumulate forces back.
    pub fn pack_border(
        &mut self,
        i: usize,
        change_pos: [f64; 3],
        vel_offset: [f64; 3],
        buf: &mut Vec<f64>,
    ) {
        self.pack_atom_inner(i, i as f64, change_pos, vel_offset, buf);
    }

    /// Unpack one atom from `buf` (ghost or exchanged). Returns [`ATOM_PACK_SIZE`].
    pub fn unpack_atom(&mut self, buf: &[f64], is_ghost: bool) -> usize {
        self.tag.push(buf[0] as u32);
        self.origin_index.push(buf[1] as i32);
        self.cutoff_radius.push(buf[2] as Real);
        self.atom_type.push(buf[3] as u32);
        self.pos
            .push([buf[4] as Real, buf[5] as Real, buf[6] as Real]);
        self.vel
            .push([buf[7] as Real, buf[8] as Real, buf[9] as Real]);
        self.force
            .push([buf[10] as Accum, buf[11] as Accum, buf[12] as Accum]);
        self.mass.push(buf[13] as Real);
        self.inv_mass.push((1.0 / buf[13]) as Real);
        self.image
            .push([buf[14] as i32, buf[15] as i32, buf[16] as i32]);
        self.is_ghost.push(is_ghost);
        ATOM_PACK_SIZE
    }

    /// Convenience method for tests: push a local atom with default velocity/force.
    pub fn push_test_atom(&mut self, tag: u32, pos: [f64; 3], radius: f64, mass: f64) {
        self.tag.push(tag);
        self.atom_type.push(0);
        self.origin_index.push(0);
        self.pos
            .push([pos[0] as Real, pos[1] as Real, pos[2] as Real]);
        self.vel.push([0.0; 3]);
        self.force.push([0.0; 3]);
        self.mass.push(mass as Real);
        self.inv_mass.push((1.0 / mass) as Real);
        self.cutoff_radius.push(radius as Real);
        self.image.push([0, 0, 0]);
        self.is_ghost.push(false);
    }
}

/// Coordinates the structural operations which must keep `Atom` and every
/// registered `AtomData` extension in lockstep.  It intentionally owns no data:
/// force kernels retain direct slice access to the public SoA.
pub struct ParticleStore<'a> {
    atoms: &'a mut Atom,
    registry: &'a AtomDataRegistry,
}

impl<'a> ParticleStore<'a> {
    /// Construct a coordinator after checking the parallel-row precondition.
    pub fn new(atoms: &'a mut Atom, registry: &'a AtomDataRegistry) -> Self {
        assert!(
            registry.validate_rows(atoms.len()),
            "particle extension rows are misaligned"
        );
        Self { atoms, registry }
    }

    /// Reserve capacity in both the core and extension stores.
    pub fn reserve(&mut self, additional: usize) {
        self.atoms.reserve(additional);
        self.registry.reserve_all(additional);
    }

    /// Replace the complete local core state and its restart extension rows.
    /// Extension buffers are consumed exactly; malformed or panicking decoders
    /// leave both the previous core and every extension untouched.
    pub fn replace_from_restart(
        &mut self,
        replacement: Atom,
        extension_buffers: &[Vec<f64>],
    ) -> Result<(), ParticleStoreError> {
        let expected = replacement.nlocal as usize;
        if replacement.nghost != 0 || replacement.len() != expected {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        let snapshot = self.registry.snapshot();
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.registry
                .unpack_restart_rows(expected, extension_buffers)
        }));
        if matches!(result, Ok(true)) {
            *self.atoms = replacement;
            Ok(())
        } else {
            self.registry.restore(snapshot);
            Err(ParticleStoreError::MalformedExtensionRecord)
        }
    }

    /// Run a legacy extension callback behind one all-or-nothing boundary.
    /// Snapshot availability is checked before the core is touched, and a
    /// successful-but-malformed callback is treated the same as a panic.
    fn structural_transaction(
        &mut self,
        expected_rows: usize,
        operation: impl FnOnce(&mut Atom, &AtomDataRegistry),
    ) -> Result<(), ParticleStoreError> {
        let snapshot = self.registry.snapshot();
        let core_before = self.atoms.clone();
        let result = catch_unwind(AssertUnwindSafe(|| operation(self.atoms, self.registry)));
        if result.is_ok()
            && self.registry.validate_rows(expected_rows)
            && self.atoms.len() == expected_rows
        {
            return Ok(());
        }
        *self.atoms = core_before;
        self.registry.restore(snapshot);
        Err(ParticleStoreError::MalformedExtensionRecord)
    }

    /// Remove a locally owned row from core and extensions together.
    pub fn swap_remove(&mut self, index: usize) -> Result<(), ParticleStoreError> {
        // `Vec::swap_remove` would pull the final ghost into the local prefix.
        // Full-rebuild ordering discards ghosts before deletions; reject an
        // accidental out-of-order caller rather than corrupting ownership.
        if self.atoms.nghost != 0
            || index >= self.atoms.nlocal as usize
            || !self.registry.validate_rows(self.atoms.len())
        {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        let expected = self.atoms.len() - 1;
        self.structural_transaction(expected, |atoms, registry| {
            atoms.swap_remove(index);
            registry.swap_remove_all(index);
            atoms.nlocal -= 1;
        })
    }

    /// Drop the ghost suffix from core and extensions together.
    pub fn discard_ghosts(&mut self) -> Result<(), ParticleStoreError> {
        if !self.registry.validate_rows(self.atoms.len()) {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        let nlocal = self.atoms.nlocal as usize;
        self.structural_transaction(nlocal, |atoms, registry| {
            atoms.truncate_to(nlocal);
            registry.truncate_all(nlocal);
            atoms.nghost = 0;
        })
    }

    /// Apply a validated local permutation to all parallel columns.
    pub fn apply_permutation(
        &mut self,
        permutation: &[usize],
        scratch: &mut Vec<bool>,
    ) -> Result<(), ParticleStoreError> {
        let n = self.atoms.nlocal as usize;
        if permutation.len() != n || !self.registry.validate_rows(self.atoms.len()) {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        scratch.clear();
        scratch.resize(n, false);
        for &source in permutation {
            if source >= n || std::mem::replace(&mut scratch[source], true) {
                return Err(ParticleStoreError::InvalidStructuralOperation);
            }
        }
        self.structural_transaction(self.atoms.len(), |atoms, registry| {
            atoms.apply_permutation_with_scratch(permutation, n, scratch);
            registry.apply_permutation_all_with_scratch(permutation, n, scratch);
        })
    }

    fn frame_record<F>(buf: &mut Vec<f64>, pack: F)
    where
        F: FnOnce(&mut Vec<f64>),
    {
        let frame = buf.len();
        buf.push(0.0);
        pack(buf);
        buf[frame] = (buf.len() - frame - 1) as f64;
    }

    /// Pack a local migrant as a length-delimited core-plus-extension record.
    pub fn pack_migrant(&self, index: usize, buf: &mut Vec<f64>) -> Result<(), ParticleStoreError> {
        if index >= self.atoms.len() || !self.registry.validate_rows(self.atoms.len()) {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        Self::frame_record(buf, |buf| {
            self.atoms.pack_exchange(index, buf);
            self.registry.pack_all(index, buf);
        });
        Ok(())
    }

    /// Pack a border ghost record with the same explicit frame used by exchange.
    pub fn pack_border(
        &mut self,
        index: usize,
        change_pos: [f64; 3],
        vel_offset: [f64; 3],
        buf: &mut Vec<f64>,
    ) -> Result<(), ParticleStoreError> {
        if index >= self.atoms.len() || !self.registry.validate_rows(self.atoms.len()) {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        Self::frame_record(buf, |buf| {
            self.atoms.pack_border(index, change_pos, vel_offset, buf);
            self.registry.pack_all(index, buf);
        });
        Ok(())
    }

    fn append_received(
        &mut self,
        payload: &[f64],
        ghost: bool,
    ) -> Result<usize, ParticleStoreError> {
        let Some(&length) = payload.first() else {
            return Err(ParticleStoreError::TruncatedCoreRecord);
        };
        if !length.is_finite()
            || length < ATOM_PACK_SIZE as f64
            || length > (usize::MAX - 1) as f64
            || length.fract() != 0.0
        {
            return Err(ParticleStoreError::MalformedExtensionRecord);
        }
        let record_len = length as usize;
        if payload.len() < record_len + 1 {
            return Err(ParticleStoreError::TruncatedCoreRecord);
        }
        let record = &payload[1..=record_len];
        let rows_before = self.atoms.len();
        let extension_used = self
            .registry
            .unpack_one(rows_before, &record[ATOM_PACK_SIZE..])?;
        debug_assert_eq!(ATOM_PACK_SIZE + extension_used, record_len);
        self.atoms.unpack_atom(record, ghost);
        Ok(record_len + 1)
    }

    /// Append a single ghost record. For a variable-width stream, use the
    /// counted batch API so its exact end can be verified.
    pub fn append_ghost(&mut self, payload: &[f64]) -> Result<usize, ParticleStoreError> {
        self.append_records(payload, 1, true)?;
        Ok(payload.len())
    }

    /// Append a single migrated local record.
    pub fn append_migrant(&mut self, payload: &[f64]) -> Result<usize, ParticleStoreError> {
        self.append_records(payload, 1, false)?;
        Ok(payload.len())
    }

    /// Append exactly `count` ghost records, rolling the whole batch back when
    /// a record is bad or the count/payload boundary disagrees.
    pub fn append_ghost_records(
        &mut self,
        payload: &[f64],
        count: usize,
    ) -> Result<(), ParticleStoreError> {
        self.append_records(payload, count, true)
    }

    /// Append exactly `count` migrant records transactionally.
    pub fn append_migrant_records(
        &mut self,
        payload: &[f64],
        count: usize,
    ) -> Result<(), ParticleStoreError> {
        self.append_records(payload, count, false)
    }

    fn append_records(
        &mut self,
        payload: &[f64],
        count: usize,
        ghost: bool,
    ) -> Result<(), ParticleStoreError> {
        // A migrated atom is locally owned.  Appending it after a ghost suffix
        // would put a local row behind ghosts and violate Atom's prefix layout.
        if (!ghost && self.atoms.nghost != 0) || !self.registry.validate_rows(self.atoms.len()) {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        let rows_before = self.atoms.len();
        let core_before = self.atoms.clone();
        let extension_before = self.registry.snapshot();
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.reserve(count);
            let mut pos = 0;
            for _ in 0..count {
                let used = self.append_received(&payload[pos..], ghost)?;
                if ghost {
                    self.atoms.nghost += 1;
                } else {
                    self.atoms.nlocal += 1;
                }
                pos += used;
            }
            (pos == payload.len())
                .then_some(())
                .ok_or(ParticleStoreError::MalformedExtensionRecord)
        }));
        if matches!(result, Ok(Ok(())))
            && self.atoms.len() == rows_before + count
            && self.registry.validate_rows(rows_before + count)
        {
            return Ok(());
        }
        *self.atoms = core_before;
        self.registry.restore(extension_before);
        Err(ParticleStoreError::MalformedExtensionRecord)
    }
}

/// Compute kinetic energy over local atoms, optionally filtered by a group mask.
pub fn compute_ke(atoms: &Atom, mask: Option<&[bool]>) -> f64 {
    let nlocal = atoms.nlocal as usize;
    let mut ke = 0.0;
    for i in 0..nlocal {
        if let Some(m) = mask {
            if !m[i] {
                continue;
            }
        }
        let v = atoms.vel[i];
        let (vx, vy, vz) = (v[0] as f64, v[1] as f64, v[2] as f64);
        ke += atoms.mass[i] as f64 * (vx * vx + vy * vy + vz * vz);
    }
    0.5 * ke
}

// ── Plugin & systems ──────────────────────────────────────────────────────────

/// Registers the [`Atom`] and [`AtomDataRegistry`] resources and per-step force zeroing.
pub struct AtomPlugin;

impl Plugin for AtomPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(Atom::new())
            .add_resource(AtomDataRegistry::new())
            .add_update_system(
                remove_ghost_atoms.run_if(in_state(CommState::FullRebuild)),
                ParticleSimScheduleSet::PostInitialIntegration,
            )
            .add_update_system(
                zero_all_forces,
                ParticleSimScheduleSet::PostInitialIntegration,
            );
    }
}

/// Truncate ghost atoms before the next communication phase.
/// Gated by `run_if(in_state(CommState::FullRebuild))` — only runs on full rebuild steps.
pub fn remove_ghost_atoms(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    ParticleStore::new(&mut atoms, &registry)
        .discard_ghosts()
        .expect("AtomPlugin found misaligned particle columns");
}

fn zero_all_forces(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    let n = atoms.len();
    atoms.force[..n].fill([0.0; 3]);
    registry.zero_all(n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use soil_derive::AtomData;
    use std::any::Any;

    #[derive(Default, AtomData)]
    struct DerivedRows {
        #[forward]
        scalar: Vec<f64>,
        #[reverse]
        vector: Vec<[f64; 3]>,
    }

    #[derive(Default)]
    struct ManualRows {
        rows: Vec<f64>,
    }

    impl AtomData for ManualRows {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn snapshot(&self) -> Box<dyn AtomData> {
            Box::new(Self {
                rows: self.rows.clone(),
            })
        }
        fn len(&self) -> usize {
            self.rows.len()
        }
        fn push_default(&mut self) {
            self.rows.push(-1.0);
        }
        fn truncate(&mut self, n: usize) {
            self.rows.truncate(n);
        }
        fn swap_remove(&mut self, i: usize) {
            self.rows.swap_remove(i);
        }
        fn pack(&self, i: usize, buf: &mut Vec<f64>) {
            buf.push(self.rows[i]);
        }
        fn unpack(&mut self, buf: &[f64]) -> usize {
            self.rows.push(buf[0]);
            1
        }
        fn apply_permutation(&mut self, perm: &[usize], n: usize) {
            let mut visited = Vec::new();
            self.apply_permutation_with_scratch(perm, n, &mut visited);
        }
        fn apply_permutation_with_scratch(
            &mut self,
            perm: &[usize],
            n: usize,
            visited: &mut Vec<bool>,
        ) {
            apply_permutation_in_place(&mut self.rows, perm, n, visited);
        }
    }

    /// A legacy-style extension which has no transaction-specific hooks. Its
    /// deliberately panicking structural callback verifies that the facade can
    /// recover it using only its ordinary pack/unpack wire contract.
    struct PanickingRows {
        rows: Vec<f64>,
    }

    impl AtomData for PanickingRows {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn snapshot(&self) -> Box<dyn AtomData> {
            Box::new(Self {
                rows: self.rows.clone(),
            })
        }
        fn len(&self) -> usize {
            self.rows.len()
        }
        fn push_default(&mut self) {
            self.rows.push(0.0);
        }
        fn truncate(&mut self, n: usize) {
            self.rows.truncate(n);
        }
        fn swap_remove(&mut self, i: usize) {
            self.rows.swap_remove(i);
            panic!("simulated legacy extension failure");
        }
        fn pack(&self, i: usize, buf: &mut Vec<f64>) {
            buf.push(self.rows[i]);
        }
        fn unpack(&mut self, buf: &[f64]) -> usize {
            self.rows.push(buf[0]);
            1
        }
        fn apply_permutation(&mut self, _perm: &[usize], _n: usize) {}
    }

    /// A deliberately hostile legacy decoder: it corrupts an existing row and
    /// then reports an invalid record.  This exercises receive rollback beyond
    /// simple truncation, which cannot restore overwritten rows.
    struct CorruptingUnpackRows {
        rows: Vec<f64>,
    }

    impl AtomData for CorruptingUnpackRows {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn snapshot(&self) -> Box<dyn AtomData> {
            Box::new(Self {
                rows: self.rows.clone(),
            })
        }
        fn len(&self) -> usize {
            self.rows.len()
        }
        fn push_default(&mut self) {
            self.rows.push(0.0);
        }
        fn truncate(&mut self, n: usize) {
            self.rows.truncate(n);
        }
        fn swap_remove(&mut self, i: usize) {
            self.rows.swap_remove(i);
        }
        fn pack(&self, i: usize, buf: &mut Vec<f64>) {
            buf.push(self.rows[i]);
        }
        fn unpack(&mut self, buf: &[f64]) -> usize {
            // A clean restart decode is valid, so the facade can recover via
            // the documented wire contract.  An incremental receive instead
            // corrupts existing state and reports an invalid record.
            if self.rows.is_empty() {
                self.rows.push(buf[0]);
                1
            } else {
                self.rows[0] = -999.0;
                0
            }
        }
        fn apply_permutation(&mut self, _perm: &[usize], _n: usize) {}
    }

    /// Decoder failure must not become the mechanism used to rebuild the old
    /// state. This fixture panics after mutation, including during any attempted
    /// wire-format replay; its snapshot is the only valid recovery path.
    struct PanicOnDecodeRows {
        rows: Vec<f64>,
    }

    impl AtomData for PanicOnDecodeRows {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn snapshot(&self) -> Box<dyn AtomData> {
            Box::new(Self {
                rows: self.rows.clone(),
            })
        }
        fn len(&self) -> usize {
            self.rows.len()
        }
        fn push_default(&mut self) {
            self.rows.push(0.0);
        }
        fn truncate(&mut self, n: usize) {
            self.rows.truncate(n);
        }
        fn swap_remove(&mut self, i: usize) {
            self.rows.swap_remove(i);
        }
        fn pack(&self, i: usize, buf: &mut Vec<f64>) {
            buf.push(self.rows[i]);
        }
        fn unpack(&mut self, _buf: &[f64]) -> usize {
            self.rows.push(-999.0);
            panic!("simulated decoder failure")
        }
        fn apply_permutation(&mut self, _perm: &[usize], _n: usize) {}
    }

    /// A hand-written two-column store. Unlike a derive-generated store, a
    /// custom layout must explicitly report every column in `validate_rows`.
    #[derive(Default)]
    struct ManualMultiColumnRows {
        scalar: Vec<f64>,
        vector: Vec<[f64; 3]>,
    }

    impl AtomData for ManualMultiColumnRows {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn snapshot(&self) -> Box<dyn AtomData> {
            Box::new(Self {
                scalar: self.scalar.clone(),
                vector: self.vector.clone(),
            })
        }
        fn len(&self) -> usize {
            self.scalar.len()
        }
        fn push_default(&mut self) {
            self.scalar.push(0.0);
            self.vector.push([0.0; 3]);
        }
        fn validate_rows(&self, expected: usize) -> Result<(), AtomDataRegistrationError> {
            for actual in [self.scalar.len(), self.vector.len()] {
                if actual != expected {
                    return Err(AtomDataRegistrationError::RowCountMismatch {
                        type_name: std::any::type_name::<Self>(),
                        expected,
                        actual,
                    });
                }
            }
            Ok(())
        }
        fn truncate(&mut self, n: usize) {
            self.scalar.truncate(n);
            self.vector.truncate(n);
        }
        fn swap_remove(&mut self, i: usize) {
            self.scalar.swap_remove(i);
            self.vector.swap_remove(i);
        }
        fn pack(&self, i: usize, buf: &mut Vec<f64>) {
            buf.push(self.scalar[i]);
            buf.extend_from_slice(&self.vector[i]);
        }
        fn unpack(&mut self, buf: &[f64]) -> usize {
            self.scalar.push(buf[0]);
            self.vector.push([buf[1], buf[2], buf[3]]);
            4
        }
        fn apply_permutation(&mut self, perm: &[usize], n: usize) {
            let mut visited = Vec::new();
            self.apply_permutation_with_scratch(perm, n, &mut visited);
        }
        fn apply_permutation_with_scratch(
            &mut self,
            perm: &[usize],
            n: usize,
            visited: &mut Vec<bool>,
        ) {
            apply_permutation_in_place(&mut self.scalar, perm, n, visited);
            apply_permutation_in_place(&mut self.vector, perm, n, visited);
        }
    }

    #[derive(Default)]
    struct BrokenDefaults;

    impl AtomData for BrokenDefaults {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn snapshot(&self) -> Box<dyn AtomData> {
            Box::new(Self)
        }
        fn len(&self) -> usize {
            0
        }
        fn push_default(&mut self) {}
        fn truncate(&mut self, _: usize) {}
        fn swap_remove(&mut self, _: usize) {}
        fn pack(&self, _: usize, _: &mut Vec<f64>) {}
        fn unpack(&mut self, _: &[f64]) -> usize {
            0
        }
        fn apply_permutation(&mut self, _: &[usize], _: usize) {}
    }

    #[test]
    fn particle_store_coordinates_remove_sort_and_ghost_discard() {
        let mut atoms = Atom::new();
        for (tag, x) in [(10, 0.0), (20, 1.0), (30, 2.0)] {
            atoms.push_test_atom(tag, [x, 0.0, 0.0], 1.0, 1.0);
        }
        atoms.nlocal = 2;
        atoms.nghost = 1;
        let mut registry = AtomDataRegistry::new();
        registry
            .try_register(
                ManualRows {
                    rows: vec![100.0, 200.0, 300.0],
                },
                3,
            )
            .unwrap();

        let mut store = ParticleStore::new(&mut atoms, &registry);
        let mut scratch = Vec::new();
        store.apply_permutation(&[1, 0], &mut scratch).unwrap();
        store.discard_ghosts().unwrap();
        store.swap_remove(0).unwrap();

        assert_eq!(store.atoms.tag, vec![10]);
        assert_eq!(registry.get::<ManualRows>().unwrap().rows, vec![100.0]);
        assert_eq!((store.atoms.nlocal, store.atoms.nghost), (1, 0));
        assert!(registry.validate_rows(store.atoms.len()));
    }

    #[test]
    fn particle_store_counted_receive_accepts_variable_records_and_rolls_back() {
        let mut source_atoms = Atom::new();
        source_atoms.push_test_atom(10, [0.0; 3], 1.0, 1.0);
        source_atoms.push_test_atom(20, [1.0; 3], 1.0, 1.0);
        source_atoms.nlocal = 2;
        let mut source_registry = AtomDataRegistry::new();
        source_registry
            .try_register(
                ManualRows {
                    rows: vec![4.0, 9.0],
                },
                2,
            )
            .unwrap();
        let mut payload = Vec::new();
        for i in 0..2 {
            ParticleStore::new(&mut source_atoms, &source_registry)
                .pack_migrant(i, &mut payload)
                .unwrap();
        }

        let mut atoms = Atom::new();
        let mut registry = AtomDataRegistry::new();
        registry.try_register(ManualRows::default(), 0).unwrap();
        ParticleStore::new(&mut atoms, &registry)
            .append_migrant_records(&payload, 2)
            .unwrap();
        assert_eq!(atoms.tag, vec![10, 20]);
        assert_eq!(registry.get::<ManualRows>().unwrap().rows, vec![4.0, 9.0]);

        let before_tags = atoms.tag.clone();
        let before_rows = registry.get::<ManualRows>().unwrap().rows.clone();
        let mut malformed = payload.clone();
        malformed.push(123.0);
        assert_eq!(
            ParticleStore::new(&mut atoms, &registry).append_ghost_records(&malformed, 2),
            Err(ParticleStoreError::MalformedExtensionRecord)
        );
        assert_eq!(atoms.tag, before_tags);
        assert_eq!(registry.get::<ManualRows>().unwrap().rows, before_rows);
        assert!(registry.validate_rows(atoms.len()));
    }

    #[test]
    fn particle_store_restores_a_legacy_extension_after_structural_panic() {
        let mut atoms = Atom::new();
        atoms.push_test_atom(10, [0.0; 3], 1.0, 1.0);
        atoms.push_test_atom(20, [1.0; 3], 1.0, 1.0);
        atoms.nlocal = 2;
        let mut registry = AtomDataRegistry::new();
        registry
            .try_register(
                PanickingRows {
                    rows: vec![4.0, 9.0],
                },
                2,
            )
            .unwrap();

        assert_eq!(
            ParticleStore::new(&mut atoms, &registry).swap_remove(0),
            Err(ParticleStoreError::MalformedExtensionRecord)
        );
        assert_eq!(atoms.tag, vec![10, 20]);
        assert_eq!(atoms.nlocal, 2);
        assert_eq!(
            registry.get::<PanickingRows>().unwrap().rows,
            vec![4.0, 9.0]
        );
        assert!(registry.validate_rows(atoms.len()));
    }

    #[test]
    fn particle_store_receive_restores_overwritten_legacy_rows() {
        let mut source_atoms = Atom::new();
        source_atoms.push_test_atom(20, [1.0; 3], 1.0, 1.0);
        source_atoms.nlocal = 1;
        let mut source_registry = AtomDataRegistry::new();
        source_registry
            .try_register(CorruptingUnpackRows { rows: vec![7.0] }, 1)
            .unwrap();
        let mut payload = Vec::new();
        ParticleStore::new(&mut source_atoms, &source_registry)
            .pack_migrant(0, &mut payload)
            .unwrap();

        let mut atoms = Atom::new();
        atoms.push_test_atom(10, [0.0; 3], 1.0, 1.0);
        atoms.nlocal = 1;
        let mut registry = AtomDataRegistry::new();
        registry
            .try_register(CorruptingUnpackRows { rows: vec![5.0] }, 1)
            .unwrap();

        assert_eq!(
            ParticleStore::new(&mut atoms, &registry).append_migrant(&payload),
            Err(ParticleStoreError::MalformedExtensionRecord)
        );
        assert_eq!(atoms.tag, vec![10]);
        assert_eq!(atoms.nlocal, 1);
        assert_eq!(
            registry.get::<CorruptingUnpackRows>().unwrap().rows,
            vec![5.0]
        );
        assert!(registry.validate_rows(atoms.len()));
    }

    #[test]
    fn particle_store_rollback_never_replays_a_failing_decoder() {
        let mut atoms = Atom::new();
        atoms.push_test_atom(10, [0.0; 3], 1.0, 1.0);
        atoms.nlocal = 1;
        let mut registry = AtomDataRegistry::new();
        registry
            .try_register(PanicOnDecodeRows { rows: vec![5.0] }, 1)
            .unwrap();
        let payload = vec![ATOM_PACK_SIZE as f64 + 1.0]
            .into_iter()
            .chain(std::iter::repeat(0.0).take(ATOM_PACK_SIZE + 1))
            .collect::<Vec<_>>();

        assert_eq!(
            ParticleStore::new(&mut atoms, &registry).append_migrant(&payload),
            Err(ParticleStoreError::MalformedExtensionRecord)
        );
        assert_eq!(atoms.tag, vec![10]);
        assert_eq!(registry.get::<PanicOnDecodeRows>().unwrap().rows, vec![5.0]);
        assert!(registry.validate_rows(atoms.len()));
    }

    #[test]
    fn particle_store_rejects_migrant_after_ghost_suffix() {
        let mut source = Atom::new();
        source.push_test_atom(20, [1.0; 3], 1.0, 1.0);
        source.nlocal = 1;
        let registry = AtomDataRegistry::new();
        let mut payload = Vec::new();
        ParticleStore::new(&mut source, &registry)
            .pack_migrant(0, &mut payload)
            .unwrap();

        let mut atoms = Atom::new();
        atoms.push_test_atom(10, [0.0; 3], 1.0, 1.0);
        atoms.push_test_atom(99, [9.0; 3], 1.0, 1.0);
        atoms.nlocal = 1;
        atoms.nghost = 1;
        assert_eq!(
            ParticleStore::new(&mut atoms, &registry).append_migrant(&payload),
            Err(ParticleStoreError::InvalidStructuralOperation)
        );
        assert_eq!(atoms.tag, vec![10, 99]);
        assert_eq!((atoms.nlocal, atoms.nghost), (1, 1));
    }

    #[test]
    fn registration_backfills_derived_and_manual_default_rows() {
        let mut registry = AtomDataRegistry::new();
        registry.try_register(DerivedRows::default(), 3).unwrap();
        registry.try_register(ManualRows::default(), 3).unwrap();
        let derived = registry.get::<DerivedRows>().unwrap();
        assert_eq!(derived.scalar, vec![0.0; 3]);
        assert_eq!(derived.vector, vec![[0.0; 3]; 3]);
        assert_eq!(registry.get::<ManualRows>().unwrap().rows, vec![-1.0; 3]);
    }

    #[test]
    fn registration_macro_backfills_after_atoms_already_exist() {
        let mut app = App::new();
        app.add_plugins(AtomPlugin);
        {
            let cell = app
                .get_mut_resource(TypeId::of::<Atom>())
                .expect("AtomPlugin installs Atom");
            let mut binding = cell.borrow_mut();
            let atoms = binding.downcast_mut::<Atom>().unwrap();
            atoms.push_test_atom(1, [0.0; 3], 1.0, 1.0);
            atoms.push_test_atom(2, [1.0; 3], 1.0, 1.0);
        }
        register_atom_data!(app, DerivedRows::default());
        let cell = app
            .get_mut_resource(TypeId::of::<AtomDataRegistry>())
            .unwrap();
        let binding = cell.borrow();
        let registry = binding.downcast_ref::<AtomDataRegistry>().unwrap();
        assert_eq!(registry.get::<DerivedRows>().unwrap().len(), 2);
    }

    #[test]
    fn registration_rejects_duplicates_and_malformed_rows() {
        let mut registry = AtomDataRegistry::new();
        registry
            .try_register(
                ManualRows {
                    rows: vec![1.0, 2.0],
                },
                2,
            )
            .unwrap();
        assert!(matches!(
            registry.try_register(ManualRows::default(), 2),
            Err(AtomDataRegistrationError::Duplicate { .. })
        ));
        assert!(matches!(
            AtomDataRegistry::new().try_register(ManualRows { rows: vec![1.0] }, 2),
            Err(AtomDataRegistrationError::RowCountMismatch {
                expected: 2,
                actual: 1,
                ..
            })
        ));
        assert!(matches!(
            AtomDataRegistry::new().try_register(BrokenDefaults, 2),
            Err(AtomDataRegistrationError::DefaultBackfillFailed {
                expected: 2,
                actual: 0,
                ..
            })
        ));
    }

    #[test]
    fn derive_rejects_misaligned_columns_before_communication() {
        let data = DerivedRows {
            scalar: vec![1.0, 2.0],
            vector: vec![[1.0; 3]],
        };
        assert!(matches!(
            AtomDataRegistry::new().try_register(data, 2),
            Err(AtomDataRegistrationError::RowCountMismatch {
                expected: 2,
                actual: 1,
                ..
            })
        ));
    }

    #[test]
    fn custom_multi_column_store_rejects_a_hidden_short_column() {
        let malformed = ManualMultiColumnRows {
            scalar: vec![1.0, 2.0],
            vector: vec![[1.0; 3]],
        };
        assert_eq!(malformed.len(), 2, "the primary column alone looks valid");
        assert!(matches!(
            AtomDataRegistry::new().try_register(malformed, 2),
            Err(AtomDataRegistrationError::RowCountMismatch {
                expected: 2,
                actual: 1,
                ..
            })
        ));
    }

    #[test]
    fn derived_rows_keep_pack_and_comm_ordering_after_backfill() {
        let mut source = AtomDataRegistry::new();
        source
            .try_register(
                DerivedRows {
                    scalar: vec![4.0],
                    vector: vec![[1.0, 2.0, 3.0]],
                },
                1,
            )
            .unwrap();
        let mut migration = Vec::new();
        source.pack_all(0, &mut migration);
        assert_eq!(migration, vec![4.0, 1.0, 2.0, 3.0]);

        let mut destination = AtomDataRegistry::new();
        destination.try_register(DerivedRows::default(), 0).unwrap();
        assert_eq!(destination.unpack_all(&migration), migration.len());
        let restored = destination.get::<DerivedRows>().unwrap();
        assert_eq!(restored.scalar, vec![4.0]);
        assert_eq!(restored.vector, vec![[1.0, 2.0, 3.0]]);
        drop(restored);

        let mut forward = Vec::new();
        source.pack_forward_all(0, &mut forward);
        assert_eq!(forward, vec![4.0]);
        destination.unpack_forward_all(0, &forward);
        let mut reverse = Vec::new();
        source.pack_reverse_all(0, &mut reverse);
        assert_eq!(reverse, vec![1.0, 2.0, 3.0]);
        destination.unpack_reverse_all(0, &reverse);
        assert_eq!(
            destination.get::<DerivedRows>().unwrap().vector,
            vec![[2.0, 4.0, 6.0]]
        );
    }

    #[test]
    fn apply_permutation_in_place_matches_scratch_collect() {
        let cases: &[&[usize]] = &[
            &[0, 1, 2, 3, 4],
            &[2, 0, 1, 4, 3],
            &[4, 3, 2, 1, 0],
            &[1, 2, 3, 4, 0],
        ];

        for &perm in cases {
            let original = vec![10, 20, 30, 40, 50];
            let mut values = original.clone();
            let expected: Vec<_> = perm.iter().map(|&p| original[p]).collect();
            let mut visited = Vec::new();

            apply_permutation_in_place(&mut values, perm, perm.len(), &mut visited);

            assert_eq!(values, expected);
            assert!(visited.capacity() >= perm.len());
        }
    }

    #[test]
    fn apply_permutation_in_place_only_reorders_prefix() {
        let mut values = vec!["a", "b", "c", "ghost"];
        let mut visited = Vec::new();

        apply_permutation_in_place(&mut values, &[2, 0, 1], 3, &mut visited);

        assert_eq!(values, vec!["c", "a", "b", "ghost"]);
    }
}
