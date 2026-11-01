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
    ops::{Deref, Index, IndexMut},
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
///     unsafe fn push_default(&mut self) { self.q.push(0.0); }
///     unsafe fn truncate(&mut self, n: usize) { self.q.truncate(n); }
///     unsafe fn swap_remove(&mut self, i: usize) { self.q.swap_remove(i); }
///     fn pack(&self, i: usize, buf: &mut Vec<f64>) { buf.push(self.q[i]); }
///     unsafe fn unpack(&mut self, buf: &[f64]) -> usize { self.q.push(buf[0]); 1 }
///     unsafe fn apply_permutation(&mut self, perm: &[usize], n: usize) {
///         let mut visited = Vec::new();
///         apply_permutation_in_place(&mut self.q, perm, n, &mut visited);
///     }
/// }
///
/// let mut app = App::new();
/// app.add_plugins(AtomPlugin);            // registers the AtomDataRegistry
/// register_atom_data!(&mut app, Charge::default());
/// ```
#[macro_export]
macro_rules! register_atom_data {
    (&mut $app:ident, $value:expr) => {
        $crate::register_atom_data_in_app(&mut $app, $value)
    };
    ($app:expr, $value:expr) => {
        $crate::register_atom_data_in_app($app, $value)
    };
}

/// Register an extension while deriving its required row count from the
/// authoritative [`Atom`] resource in `app`.
///
/// This is the function behind [`register_atom_data!`]. It is public only so
/// plugin crates can register data during setup; it intentionally does not
/// accept a caller-supplied row count.
pub fn register_atom_data_in_app<T: AtomData + 'static>(app: &mut App, data: T) {
    let atom_count = if let Some(cell) = app.get_mut_resource(TypeId::of::<Atom>()) {
        cell.borrow()
            .downcast_ref::<Atom>()
            .expect("Failed to downcast Atom — this is a bug in SOIL")
            .len()
    } else {
        panic!("Atom not found — AtomPlugin must be added first");
    };
    if let Some(cell) = app.get_mut_resource(TypeId::of::<AtomDataRegistry>()) {
        let mut binder = cell.borrow_mut();
        binder
            .downcast_mut::<AtomDataRegistry>()
            .expect("Failed to downcast AtomDataRegistry — this is a bug in SOIL")
            .try_register(data, atom_count)
            .unwrap_or_else(|error| panic!("failed to register AtomData: {error}"));
    } else {
        panic!("AtomDataRegistry not found — AtomPlugin must be added first");
    }
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
    ///
    /// # Safety
    ///
    /// The caller must append the matching core row and every other extension
    /// row in the same structural transaction.  Ordinary simulation code must
    /// use [`ParticleStore`] instead of calling this through a registry borrow.
    unsafe fn push_default(&mut self);
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
    ///
    /// # Safety
    ///
    /// The caller must keep the core store and every registered extension
    /// aligned.
    unsafe fn truncate(&mut self, n: usize);
    /// Remove atom at index `i` by swapping with the last element.
    ///
    /// # Safety
    ///
    /// The caller must perform the same removal in the core store and every
    /// registered extension.
    unsafe fn swap_remove(&mut self, i: usize);
    /// Serialize all fields for atom `i` into `buf` (exchange/border comm).
    fn pack(&self, i: usize, buf: &mut Vec<f64>);
    /// Deserialize one atom's fields from `buf`, appending to internal Vecs.
    /// Returns the number of `f64`s consumed.
    ///
    /// # Safety
    ///
    /// The caller must append one matching core row and a row in every other
    /// extension, or roll the operation back.
    unsafe fn unpack(&mut self, buf: &[f64]) -> usize;
    /// Serialize atom `i`'s fields for a ghost (border) record.
    ///
    /// Ghosts only need what force kernels read on a neighbor. The default
    /// sends the full migration record ([`pack`](Self::pack)); a store whose
    /// rows carry owner-only state (contact histories, statistics) should
    /// override this pair to omit it, because the border record is sent for
    /// every ghost on every rebuild. `#[derive(AtomData)]` skips
    /// `#[no_ghost]` fields.
    fn pack_border(&self, i: usize, buf: &mut Vec<f64>) {
        self.pack(i, buf);
    }
    /// Append one ghost row from a [`pack_border`](Self::pack_border) record;
    /// returns the number of `f64`s consumed. Owner-only state is left at its
    /// default.
    ///
    /// # Safety
    ///
    /// Same contract as [`unpack`](Self::unpack).
    unsafe fn unpack_border(&mut self, buf: &[f64]) -> usize {
        unsafe { self.unpack(buf) }
    }
    /// Reorder atoms according to `perm[0..n]` (used by spatial sorting).
    ///
    /// # Safety
    ///
    /// The caller must apply the same validated permutation to the core store
    /// and every registered extension.
    unsafe fn apply_permutation(&mut self, perm: &[usize], n: usize);

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
    /// Pack the forward-comm fields of every row in `rows`, in order.
    ///
    /// Ghost communication calls this once per store and swap instead of once
    /// per atom. The default loops over [`pack_forward`](Self::pack_forward);
    /// because default methods are monomorphized per implementor, that inner
    /// call is static, so every store gets the batched path for free.
    fn pack_forward_rows(&self, rows: &[usize], buf: &mut Vec<f64>) {
        buf.reserve(rows.len() * self.forward_comm_size());
        for &i in rows {
            self.pack_forward(i, buf);
        }
    }
    /// Unpack forward-comm fields into the contiguous rows
    /// `first..first + count`; returns the number of `f64`s consumed.
    fn unpack_forward_rows(&mut self, first: usize, count: usize, buf: &[f64]) -> usize {
        let width = self.forward_comm_size();
        for k in 0..count {
            self.unpack_forward(first + k, &buf[k * width..]);
        }
        count * width
    }
    /// Pack the reverse-comm fields of the contiguous rows `first..first + count`.
    fn pack_reverse_rows(&self, first: usize, count: usize, buf: &mut Vec<f64>) {
        buf.reserve(count * self.reverse_comm_size());
        for i in first..first + count {
            self.pack_reverse(i, buf);
        }
    }
    /// Accumulate reverse-comm fields into every row in `rows`, in order;
    /// returns the number of `f64`s consumed.
    fn unpack_reverse_rows(&mut self, rows: &[usize], buf: &[f64]) -> usize {
        let width = self.reverse_comm_size();
        for (k, &i) in rows.iter().enumerate() {
            self.unpack_reverse(i, &buf[k * width..]);
        }
        rows.len() * width
    }
    /// Zero per-step accumulators (e.g. torque) for atoms 0..n.
    fn zero(&mut self, _n: usize) {}
    /// Reorder atoms with caller-owned cycle scratch.
    ///
    /// Implementations should use this when possible so spatial sorting can reuse
    /// one scratch allocation across all registered extensions. The default keeps
    /// older hand-written implementations source-compatible.
    unsafe fn apply_permutation_with_scratch(
        &mut self,
        perm: &[usize],
        n: usize,
        _visited: &mut Vec<bool>,
    ) {
        unsafe { self.apply_permutation(perm, n) };
    }
    /// Whether every structural hook touches only the rows it is asked to.
    ///
    /// Returning `true` promises that `push_default` and `unpack` only append
    /// one row, that `truncate`, `swap_remove`, and `apply_permutation*` never
    /// panic for indices the caller has already validated, and that none of
    /// them read or write state outside the affected rows. [`ParticleStore`]
    /// then rolls a failed append back by truncation and runs in-place
    /// operations without a whole-store [`snapshot`](Self::snapshot), so
    /// insertion, migration, sorting, and ghost rebuilds cost O(changed rows)
    /// instead of O(all rows) per operation.
    ///
    /// `#[derive(AtomData)]` stores return `true`. The default `false` keeps the
    /// conservative whole-store snapshot for hand-written stores; a hand-written
    /// store whose hooks are plain per-row `Vec` operations should override
    /// this to return `true`.
    fn row_local_structural_hooks(&self) -> bool {
        false
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

/// Assert the complete particle-store invariant at a phase boundary.
///
/// This is intentionally a debug-build check: production force and communication
/// kernels keep their direct SoA access and do not pay a per-step scan. Boundary
/// systems call it before handing rows to communication or rebuild code, where a
/// violated invariant would otherwise become an indexing error far from the
/// structural mutation that caused it.
#[cfg(debug_assertions)]
pub(crate) fn debug_assert_particle_invariant(
    atoms: &Atom,
    registry: &AtomDataRegistry,
    boundary: &str,
) {
    let rows = atoms.len();
    let ownership = (atoms.nlocal as usize).checked_add(atoms.nghost as usize) == Some(rows);
    assert!(
        atoms.validate_rows() && registry.validate_rows(rows) && ownership,
        "particle invariant violated at {boundary}: core_rows={rows}, nlocal={}, nghost={}, extensions_aligned={}",
        atoms.nlocal,
        atoms.nghost,
        registry.validate_rows(rows),
    );
}

#[cfg(not(debug_assertions))]
#[inline]
pub(crate) fn debug_assert_particle_invariant(
    _atoms: &Atom,
    _registry: &AtomDataRegistry,
    _boundary: &str,
) {
}

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
    /// Per store: whether it declared [`AtomData::row_local_structural_hooks`].
    /// Such stores are rolled back by truncation (appends) or need no rollback
    /// (validated in-place operations), so they are never snapshotted.
    row_local: Vec<bool>,
}

impl Default for AtomDataRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomDataRegistry {
    /// Whether an extension of the specified concrete type is registered.
    ///
    /// This supports typed query preparation; ordinary callers should use
    /// [`get`](Self::get) or [`get_mut`](Self::get_mut).
    pub(crate) fn has_type(&self, id: TypeId) -> bool {
        self.stores.iter().any(|(stored, _)| *stored == id)
    }
    /// Create an empty registry with no extensions registered.
    pub fn new() -> Self {
        AtomDataRegistry {
            stores: Vec::new(),
            forward_stores: Vec::new(),
            reverse_stores: Vec::new(),
            row_local: Vec::new(),
        }
    }

    /// Register an extension against the supplied live core store.
    ///
    /// Unlike the internal registration primitive, this API never accepts a
    /// caller-provided row count: the required count comes directly from
    /// `atoms`.  Use it only while setting up a matched Atom/registry pair;
    /// regular plugin setup should prefer [`register_atom_data!`].
    pub fn register_for_atoms<T: AtomData + 'static>(
        &mut self,
        data: T,
        atoms: &Atom,
    ) -> Result<(), AtomDataRegistrationError> {
        self.try_register(data, atoms.len())
    }

    /// Register a new extension, enforcing one row for every current atom.
    /// Empty stores are default-filled; nonempty stores must already match.
    pub(crate) fn try_register<T: AtomData + 'static>(
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
                // SAFETY: registration is the only public setup boundary; it
                // backfills this new extension to the already-fixed core row
                // count before publishing it in the registry.
                unsafe { data.push_default() };
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
        self.row_local = self
            .stores
            .iter()
            .map(|(_, cell)| cell.borrow().row_local_structural_hooks())
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
    /// Existing-row values remain mutable for force and analysis kernels. The
    /// row-changing [`AtomData`] callbacks remain unsafe, so a safe registry
    /// borrow cannot desynchronize this extension from the core store.
    ///
    /// ```compile_fail
    /// use soil_core::{AtomData, AtomDataRegistry, BondStore};
    /// let registry = AtomDataRegistry::new();
    /// registry.get_mut::<BondStore>().unwrap().truncate(0);
    /// ```
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
    ///
    /// Structural operation: kept crate-private so it can only be paired with
    /// the corresponding core operation by [`ParticleStore`].
    pub(crate) fn truncate_all(&self, n: usize) {
        for (_, cell) in &self.stores {
            // SAFETY: ParticleStore pairs this registry-wide operation with
            // the matching core structural transaction.
            unsafe { cell.borrow_mut().truncate(n) };
        }
    }

    /// Append one default row to every registered extension.
    ///
    /// This is deliberately registry-owned: a local particle may only be
    /// created through [`ParticleStore`], which performs this operation in the
    /// same transaction as the core SoA append.
    fn push_default_all(&self) {
        for (_, cell) in &self.stores {
            // SAFETY: called only from ParticleStore's core+extension
            // transaction.
            unsafe { cell.borrow_mut().push_default() };
        }
    }

    /// Reserve capacity in each extension alongside the core SoA.
    pub(crate) fn reserve_all(&self, additional: usize) {
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

    /// Whether any registered store still needs a whole-store snapshot.
    fn has_snapshot_stores(&self) -> bool {
        self.row_local.iter().any(|&row_local| !row_local)
    }

    /// Snapshot only the stores that did not declare row-local structural
    /// hooks. Returns `None` (no allocation, no copying) when every store is
    /// row-local, which is the case for all `#[derive(AtomData)]` stores.
    fn rollback_snapshot(&self) -> Option<Vec<Option<Box<dyn AtomData>>>> {
        if !self.has_snapshot_stores() {
            return None;
        }
        Some(
            self.stores
                .iter()
                .zip(&self.row_local)
                .map(|((_, cell), &row_local)| (!row_local).then(|| cell.borrow().snapshot()))
                .collect(),
        )
    }

    /// Undo a failed append: restore snapshotted stores, and truncate the
    /// row-local stores back to `rows_before` (an append only added rows).
    fn rollback_append(
        &self,
        snapshot: Option<Vec<Option<Box<dyn AtomData>>>>,
        rows_before: usize,
    ) {
        let mut saved = snapshot.map(Vec::into_iter);
        for ((_, cell), &row_local) in self.stores.iter().zip(&self.row_local) {
            let slot = saved.as_mut().and_then(Iterator::next).flatten();
            if let Some(saved) = slot {
                *cell.borrow_mut() = saved;
            } else if row_local {
                // SAFETY: the caller truncates the core to the same row count.
                let mut store = cell.borrow_mut();
                if store.len() > rows_before {
                    unsafe { store.truncate(rows_before) };
                }
            }
        }
    }

    /// Restore the snapshotted (non-row-local) stores after a failed in-place
    /// operation. Row-local stores are run only after every snapshotted store
    /// succeeded, so they are still untouched here.
    fn rollback_in_place(&self, snapshot: Option<Vec<Option<Box<dyn AtomData>>>>) {
        let Some(saved) = snapshot else { return };
        for ((_, cell), slot) in self.stores.iter().zip(saved) {
            if let Some(saved) = slot {
                *cell.borrow_mut() = saved;
            }
        }
    }

    /// Run an in-place structural hook on every store: the snapshotted stores
    /// first (a panic there is recoverable because nothing row-local has been
    /// touched), then the row-local stores, which promised not to fail.
    fn for_each_store_in_place(
        &self,
        expected_rows: usize,
        mut op: impl FnMut(&mut dyn AtomData),
    ) -> bool {
        let snapshot = self.rollback_snapshot();
        let legacy = catch_unwind(AssertUnwindSafe(|| {
            for ((_, cell), &row_local) in self.stores.iter().zip(&self.row_local) {
                if !row_local {
                    op(&mut **cell.borrow_mut());
                }
            }
        }));
        let legacy_ok = legacy.is_ok()
            && self
                .stores
                .iter()
                .zip(&self.row_local)
                .all(|((_, cell), &row_local)| row_local || cell.borrow().len() == expected_rows);
        if !legacy_ok {
            self.rollback_in_place(snapshot);
            return false;
        }
        for ((_, cell), &row_local) in self.stores.iter().zip(&self.row_local) {
            if row_local {
                // A panic here propagates: a store that declared row-local hooks
                // broke that promise and the particle rows can no longer be
                // reconciled.
                op(&mut **cell.borrow_mut());
            }
        }
        true
    }

    /// Pack all extension fields for atom `i` into `buf` (exchange/border comm).
    pub(crate) fn pack_all(&self, i: usize, buf: &mut Vec<f64>) {
        for (_, cell) in &self.stores {
            cell.borrow().pack(i, buf);
        }
    }

    /// Pack the ghost (border) record of every extension for atom `i`.
    pub(crate) fn pack_border_all(&self, i: usize, buf: &mut Vec<f64>) {
        for (_, cell) in &self.stores {
            cell.borrow().pack_border(i, buf);
        }
    }

    /// Unpack one atom's extension fields from `buf`. Returns total `f64`s consumed.
    pub(crate) fn unpack_all(&self, buf: &[f64]) -> usize {
        let mut pos = 0;
        for (_, cell) in &self.stores {
            // SAFETY: ParticleStore contains this legacy decoder behind an
            // all-or-nothing transaction.
            pos += unsafe { cell.borrow_mut().unpack(&buf[pos..]) };
        }
        pos
    }

    /// Unpack one ghost (border) record. Returns total `f64`s consumed.
    fn unpack_border_all(&self, buf: &[f64]) -> usize {
        let mut pos = 0;
        for (_, cell) in &self.stores {
            // SAFETY: as `unpack_all`.
            pos += unsafe { cell.borrow_mut().unpack_border(&buf[pos..]) };
        }
        pos
    }

    /// Decode one extension record while containing the legacy infallible
    /// callback boundary. A failed decoder is rolled back to `expected_rows`.
    fn unpack_one(
        &self,
        expected_rows: usize,
        buf: &[f64],
        ghost: bool,
    ) -> Result<usize, ParticleStoreError> {
        let decoded = catch_unwind(AssertUnwindSafe(|| {
            if ghost {
                self.unpack_border_all(buf)
            } else {
                self.unpack_all(buf)
            }
        }));
        match decoded {
            Ok(used) if used == buf.len() && self.validate_rows(expected_rows + 1) => Ok(used),
            _ => {
                self.truncate_all(expected_rows);
                Err(ParticleStoreError::MalformedExtensionRecord)
            }
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

    /// Pack every forward-comm store for the rows in `rows`: one block per
    /// store (store-major), each holding `rows.len() * width` values. One
    /// borrow and one dynamic call per store, not per atom.
    pub fn pack_forward_rows(&self, rows: &[usize], buf: &mut Vec<f64>) {
        for &si in &self.forward_stores {
            self.stores[si].1.borrow().pack_forward_rows(rows, buf);
        }
    }

    /// Mirror of [`pack_forward_rows`](Self::pack_forward_rows) into the
    /// contiguous ghost rows `first..first + count`. Returns `f64`s consumed.
    pub fn unpack_forward_rows(&self, first: usize, count: usize, buf: &[f64]) -> usize {
        let mut pos = 0;
        for &si in &self.forward_stores {
            pos += self.stores[si]
                .1
                .borrow_mut()
                .unpack_forward_rows(first, count, &buf[pos..]);
        }
        pos
    }

    /// Pack every reverse-comm store for the contiguous ghost rows
    /// `first..first + count`, store-major.
    pub fn pack_reverse_rows(&self, first: usize, count: usize, buf: &mut Vec<f64>) {
        for &si in &self.reverse_stores {
            self.stores[si]
                .1
                .borrow()
                .pack_reverse_rows(first, count, buf);
        }
    }

    /// Mirror of [`pack_reverse_rows`](Self::pack_reverse_rows): accumulate
    /// into the owner rows `rows`. Returns `f64`s consumed.
    pub fn unpack_reverse_rows(&self, rows: &[usize], buf: &[f64]) -> usize {
        let mut pos = 0;
        for &si in &self.reverse_stores {
            pos += self.stores[si]
                .1
                .borrow_mut()
                .unpack_reverse_rows(rows, &buf[pos..]);
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

    /// Decode an exact restart row count, rejecting under- and over-consumption.
    fn unpack_restart_rows(&self, expected_rows: usize, buffers: &[Vec<f64>]) -> bool {
        if buffers.len() != self.stores.len() {
            return false;
        }
        for ((_, cell), buf) in self.stores.iter().zip(buffers) {
            let mut store = cell.borrow_mut();
            unsafe { store.truncate(0) };
            let mut pos = 0;
            for _ in 0..expected_rows {
                let used = unsafe { store.unpack(&buf[pos..]) };
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

/// Slice-shaped public SoA storage: hot kernels can update existing rows, but
/// structural edits cannot bypass [`ParticleStore`].
///
/// ```compile_fail
/// use soil_core::Atom;
/// let mut atoms = Atom::new();
/// atoms.pos.push([0.0; 3]);
/// ```
///
/// ```compile_fail
/// use soil_core::Atom;
/// let mut atoms = Atom::new();
/// atoms.pos.reverse();
/// ```
///
/// ```compile_fail
/// use soil_core::Atom;
/// let mut atoms = Atom::new();
/// atoms.pos.swap(0, 1);
/// ```
///
/// ```compile_fail
/// use soil_core::Atom;
/// let mut atoms = Atom::new();
/// atoms.pos.rotate_left(1);
/// ```
///
/// ```compile_fail
/// use soil_core::Atom;
/// let mut atoms = Atom::new();
/// atoms.push_test_atom(1, [0.0; 3], 1.0, 1.0);
/// ```
///
/// ```compile_fail
/// use soil_core::Atom;
/// let mut atoms = Atom::new();
/// atoms.nlocal = 1;
/// ```
///
/// ```compile_fail
/// use soil_core::Atom;
/// let mut atoms = Atom::new();
/// atoms.nghost = 1;
/// ```
///
/// ```compile_fail
/// use soil_core::AtomColumn;
/// let _column = AtomColumn::from_vec_unchecked(vec![[0.0; 3]]);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AtomColumn<T>(Vec<T>);
impl<T> Deref for AtomColumn<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}
impl<T, I> Index<I> for AtomColumn<T>
where
    Vec<T>: Index<I>,
{
    type Output = <Vec<T> as Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}
impl<T> IndexMut<usize> for AtomColumn<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl<T: PartialEq> PartialEq<Vec<T>> for AtomColumn<T> {
    fn eq(&self, other: &Vec<T>) -> bool {
        self.0 == *other
    }
}
impl<T: PartialEq> PartialEq<AtomColumn<T>> for Vec<T> {
    fn eq(&self, other: &AtomColumn<T>) -> bool {
        *self == other.0
    }
}
impl<T> AtomColumn<T> {
    /// Returns a raw mutable pointer for allocation-free kernels.
    ///
    /// # Safety
    ///
    /// The pointer may only be used to update rows already present in this
    /// column.  It must not be used to construct a slice with a different row
    /// count or to perform a structural operation; use [`ParticleStore`] for
    /// those operations.
    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }

    /// Iterate existing rows mutably without exposing structural operations.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.0.iter_mut()
    }

    /// Mutably access one existing row without exposing structural operations.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.0.get_mut(index)
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    fn push(&mut self, value: T) {
        self.0.push(value);
    }
    fn truncate(&mut self, len: usize) {
        self.0.truncate(len);
    }
    fn swap_remove(&mut self, index: usize) -> T {
        self.0.swap_remove(index)
    }
    fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }
}

/// Struct-of-arrays storage for all per-atom fields (position, velocity, force, etc.).
#[derive(Clone)]
pub struct Atom {
    /// Total number of atoms across all ranks (global count).
    pub(crate) natoms: u64,
    /// Number of local atoms this rank owns; they occupy indices `0..nlocal`.
    pub(crate) nlocal: u32,
    /// Number of ghost atoms (halo copies); they occupy `nlocal..nlocal+nghost`.
    pub(crate) nghost: u32,

    /// Number of distinct atom types in the simulation.
    pub ntypes: usize,

    /// Integration timestep.
    pub dt: f64,

    /// Globally unique atom identifier (persists across migration/restart).
    pub tag: AtomColumn<u32>,
    /// Per-atom type index in `0..ntypes`.
    pub atom_type: AtomColumn<u32>,
    /// For a ghost, the local index of its owner atom on the sending rank
    /// (set during border packing so reverse comm can accumulate back).
    pub origin_index: AtomColumn<i32>,
    /// `true` for ghost (halo) atoms, `false` for locally owned atoms.
    pub is_ghost: AtomColumn<bool>,

    // Interleaved arrays: field[i] = [x, y, z]
    /// Position `[x, y, z]` per atom.
    pub pos: AtomColumn<[Real; 3]>,
    /// Velocity `[vx, vy, vz]` per atom.
    pub vel: AtomColumn<[Real; 3]>,
    /// Force accumulator `[fx, fy, fz]` per atom (zeroed each step before force systems).
    pub force: AtomColumn<[Accum; 3]>,

    /// Per-atom interaction cutoff radius; feeds the neighbor skin/ghost-cutoff chain.
    pub cutoff_radius: AtomColumn<Real>,
    /// Per-atom mass.
    pub mass: AtomColumn<Real>,
    /// Cached reciprocal mass `1/mass` (kept in sync with `mass`; avoids per-step division).
    pub inv_mass: AtomColumn<Real>,
    /// PBC image flags: number of times atom has crossed each periodic boundary.
    pub image: AtomColumn<[i32; 3]>,
}

/// All core columns required to restore a local, ghost-free atom store.
///
/// This is deliberately an off-store transport object: its vectors may be
/// decoded independently, but they cannot become live [`Atom`] columns until
/// [`ParticleStore::replace_from_restart_rows`] verifies the core rows and then atomically pairs
/// core replacement with the registered extension rows.
#[derive(Debug, Default)]
pub struct LocalAtomRows {
    /// Globally unique particle tags.
    pub tag: Vec<u32>,
    /// Per-particle type indices.
    pub atom_type: Vec<u32>,
    /// Particle positions.
    pub pos: Vec<[Real; 3]>,
    /// Particle velocities.
    pub vel: Vec<[Real; 3]>,
    /// Particle force accumulators.
    pub force: Vec<[Accum; 3]>,
    /// Particle masses.
    pub mass: Vec<Real>,
    /// Per-particle interaction cutoff radii.
    pub cutoff_radius: Vec<Real>,
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
                $( $field: AtomColumn::default(), )*
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
        pub(crate) fn swap_remove(&mut self, i: usize) {
            $( self.$field.swap_remove(i); )*
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
        pub(crate) fn reserve(&mut self, additional: usize) {
            $( self.$field.reserve(additional); )*
        }
    };
}

macro_rules! impl_atom_validate_rows {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        /// Whether every core struct-of-arrays column has the same row count.
        ///
        /// `Atom::len()` is defined by the position column, so callers that
        /// construct a replacement atom from an external representation must
        /// check the other columns explicitly before installing it.
        pub fn validate_rows(&self) -> bool {
            let expected = self.pos.len();
            $( self.$field.len() == expected )&&*
        }
    };
}

macro_rules! impl_atom_apply_permutation {
    ( $( ($field:ident, $ty:ty) ),* $(,)? ) => {
        /// Reorder the first `n` atoms using caller-owned cycle scratch.
        pub(crate) fn apply_permutation_with_scratch(
            &mut self,
            perm: &[usize],
            n: usize,
            visited: &mut Vec<bool>,
        ) {
            $( apply_permutation_in_place(self.$field.as_mut_slice(), perm, n, visited); )*
        }
    };
}

impl Atom {
    for_each_atom_vec!(impl_atom_new);
    for_each_atom_vec!(impl_atom_swap_remove);
    for_each_atom_vec!(impl_atom_truncate_to);
    for_each_atom_vec!(impl_atom_reserve);
    for_each_atom_vec!(impl_atom_validate_rows);
    for_each_atom_vec!(impl_atom_apply_permutation);

    /// Build a complete local core replacement from decoded rows.
    ///
    /// The resulting store has no ghosts and derives `inv_mass`, origin indices,
    /// and image counters instead of accepting independently mutable versions of
    /// those structural columns.  It is intentionally not an installation API:
    /// use [`ParticleStore::replace_from_restart`] to pair it with extensions.
    fn from_local_rows(
        natoms: u64,
        dt: f64,
        rows: LocalAtomRows,
    ) -> Result<Self, ParticleStoreError> {
        let n = rows.tag.len();
        if rows.atom_type.len() != n
            || rows.pos.len() != n
            || rows.vel.len() != n
            || rows.force.len() != n
            || rows.mass.len() != n
            || rows.cutoff_radius.len() != n
            || u32::try_from(n).is_err()
        {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }

        let inv_mass = rows.mass.iter().map(|&mass| 1.0 / mass).collect();
        Ok(Self {
            natoms,
            nlocal: n as u32,
            nghost: 0,
            ntypes: 1,
            dt,
            tag: AtomColumn(rows.tag),
            atom_type: AtomColumn(rows.atom_type),
            origin_index: AtomColumn(vec![0; n]),
            is_ghost: AtomColumn(vec![false; n]),
            pos: AtomColumn(rows.pos),
            vel: AtomColumn(rows.vel),
            force: AtomColumn(rows.force),
            cutoff_radius: AtomColumn(rows.cutoff_radius),
            mass: AtomColumn(rows.mass),
            inv_mass: AtomColumn(inv_mass),
            image: AtomColumn(vec![[0; 3]; n]),
        })
    }

    /// Total number of atoms (local + ghost) currently stored.
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// Global particle count last supplied by the owning collective.
    pub fn natoms(&self) -> u64 {
        self.natoms
    }

    /// Update the reporting count obtained from an external collective.
    ///
    /// Unlike the local/ghost partition, this value does not select or relabel
    /// rows and is therefore safe to refresh independently.
    pub fn set_global_natoms(&mut self, natoms: u64) {
        self.natoms = natoms;
    }

    /// Number of locally owned rows at the front of the store.
    pub fn nlocal(&self) -> u32 {
        self.nlocal
    }

    /// Number of ghost rows in the suffix of the store.
    pub fn nghost(&self) -> u32 {
        self.nghost
    }

    /// Set the ownership partition for a deliberately core-only fixture.
    ///
    /// # Safety
    ///
    /// The caller must ensure all registered extension columns have the same
    /// length and that `nlocal + nghost == self.len()`. Production structural
    /// changes must go through [`ParticleStore`].
    pub unsafe fn set_layout_unchecked(&mut self, nlocal: u32, nghost: u32, natoms: u64) {
        self.nlocal = nlocal;
        self.nghost = nghost;
        self.natoms = natoms;
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
    pub(crate) fn pack_exchange(&self, i: usize, buf: &mut Vec<f64>) {
        self.pack_atom_inner(i, 0.0, [0.0; 3], [0.0; 3], buf);
    }

    /// Pack atom `i` as a ghost for border communication, applying a periodic
    /// position shift and (for sheared periodic faces) a velocity shift.
    /// Stores the local index as `origin_index` so reverse comm can accumulate forces back.
    pub(crate) fn pack_border(
        &mut self,
        i: usize,
        change_pos: [f64; 3],
        vel_offset: [f64; 3],
        buf: &mut Vec<f64>,
    ) {
        self.pack_atom_inner(i, i as f64, change_pos, vel_offset, buf);
    }

    /// Unpack one atom from `buf` (ghost or exchanged). Returns [`ATOM_PACK_SIZE`].
    pub(crate) fn unpack_atom(&mut self, buf: &[f64], is_ghost: bool) -> usize {
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

    /// Internal test-only core-row helper.
    ///
    /// # Safety
    ///
    /// This bypasses the synchronized [`ParticleStore`] transaction and must
    /// only be used when there are no registered extensions, followed by the
    /// caller setting a matching ownership partition. Production code must use
    /// `ParticleStore::push_default_local`.
    unsafe fn push_test_atom_unchecked(&mut self, tag: u32, pos: [f64; 3], radius: f64, mass: f64) {
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

    /// Internal test-only core-row helper.
    ///
    /// # Safety
    ///
    /// This bypasses the synchronized [`ParticleStore`] transaction and must
    /// only be used when there are no registered extensions, followed by the
    /// caller setting a matching ownership partition. Production code must use
    /// `ParticleStore::push_default_local`.
    #[cfg(not(test))]
    pub unsafe fn push_test_atom(&mut self, tag: u32, pos: [f64; 3], radius: f64, mass: f64) {
        unsafe { self.push_test_atom_unchecked(tag, pos, radius, mass) }
    }

    #[cfg(test)]
    pub(crate) fn push_test_atom(&mut self, tag: u32, pos: [f64; 3], radius: f64, mass: f64) {
        // SAFETY: crate tests construct core-only fixtures and set their
        // ownership partition before handing them to a ParticleStore.
        unsafe { self.push_test_atom_unchecked(tag, pos, radius, mass) }
    }

    /// Append one local core row with substrate defaults.
    ///
    /// Particle-method plugins fill method-specific extension columns after
    /// construction.  The facade pairs this with extension `push_default`
    /// calls, so the default row never exists in the core alone.
    fn push_default_row(&mut self) {
        self.tag.push(0);
        self.atom_type.push(0);
        self.origin_index.push(0);
        self.is_ghost.push(false);
        self.pos.push([0.0; 3]);
        self.vel.push([0.0; 3]);
        self.force.push([0.0; 3]);
        self.cutoff_radius.push(0.0);
        self.mass.push(1.0);
        self.inv_mass.push(1.0);
        self.image.push([0; 3]);
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
            atoms.validate_rows() && registry.validate_rows(atoms.len()),
            "particle core or extension rows are misaligned"
        );
        Self { atoms, registry }
    }

    /// Read the core SoA from a synchronized store.
    pub fn atoms(&self) -> &Atom {
        self.atoms
    }

    /// Check the complete structural invariant, including ownership partition.
    ///
    /// Row alignment alone is insufficient: a stale `nlocal`/`nghost` pair can
    /// make an otherwise aligned ghost suffix appear locally owned.  Structural
    /// operations must reject that state rather than silently relabelling rows.
    fn has_valid_layout(&self) -> bool {
        self.atoms.validate_rows()
            && self.registry.validate_rows(self.atoms.len())
            && (self.atoms.nlocal as usize).checked_add(self.atoms.nghost as usize)
                == Some(self.atoms.len())
    }

    /// Reserve capacity in both the core and extension stores.
    pub fn reserve(&mut self, additional: usize) {
        self.atoms.reserve(additional);
        self.registry.reserve_all(additional);
    }

    /// Decode and install a complete local restart image atomically with every
    /// registered extension.  The row bundle cannot become a live [`Atom`]
    /// outside this checked transaction.
    pub fn replace_from_restart_rows(
        &mut self,
        natoms: u64,
        dt: f64,
        rows: LocalAtomRows,
        extension_buffers: &[Vec<f64>],
    ) -> Result<(), ParticleStoreError> {
        let replacement = Atom::from_local_rows(natoms, dt, rows)?;
        self.replace_from_restart(replacement, extension_buffers)
    }

    /// Transactionally append one locally-owned particle with core defaults.
    ///
    /// `global_natoms` is supplied by the caller that owns the necessary
    /// collective accounting; a rank-local append must not guess a global
    /// total.  The facade installs that authoritative value only if both core
    /// and every extension accepted their new aligned row.
    pub fn push_default_local(&mut self, global_natoms: u64) -> Result<(), ParticleStoreError> {
        if !self.has_valid_layout() || self.atoms.nghost != 0 {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        let rows_before = self.atoms.len();
        // Only stores without row-local hooks are snapshotted; a row-local
        // append is undone by truncation. The core row is added last, once
        // every extension accepted its default row, so it never needs undoing.
        let snapshot = self.registry.rollback_snapshot();
        let result = catch_unwind(AssertUnwindSafe(|| self.registry.push_default_all()));
        if result.is_err() || !self.registry.validate_rows(rows_before + 1) {
            self.registry.rollback_append(snapshot, rows_before);
            return Err(ParticleStoreError::MalformedExtensionRecord);
        }
        self.atoms.push_default_row();
        self.atoms.nlocal += 1;
        self.atoms.natoms = global_natoms;
        Ok(())
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
        if !self.has_valid_layout()
            || replacement.nghost != 0
            || replacement.len() != expected
            || !replacement.validate_rows()
        {
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

    /// Run an in-place extension operation behind an all-or-nothing boundary,
    /// then apply `core` to the core columns.
    ///
    /// Extensions run first: stores without row-local hooks are snapshotted and
    /// restored if they panic or leave the wrong row count, and the core is not
    /// touched in that case. Row-local stores (every `#[derive(AtomData)]`
    /// store) are neither snapshotted nor copied, so the cost of an operation is
    /// proportional to the rows it changes, not to the size of the store.
    fn in_place_transaction(
        &mut self,
        expected_rows: usize,
        extension: impl FnMut(&mut dyn AtomData),
        core: impl FnOnce(&mut Atom),
    ) -> Result<(), ParticleStoreError> {
        if !self
            .registry
            .for_each_store_in_place(expected_rows, extension)
            || !self.registry.validate_rows(expected_rows)
        {
            return Err(ParticleStoreError::MalformedExtensionRecord);
        }
        core(self.atoms);
        debug_assert_eq!(self.atoms.len(), expected_rows);
        Ok(())
    }

    /// Remove a locally owned row from core and extensions together.
    pub fn swap_remove(&mut self, index: usize) -> Result<(), ParticleStoreError> {
        // `Vec::swap_remove` would pull the final ghost into the local prefix.
        // Full-rebuild ordering discards ghosts before deletions; reject an
        // accidental out-of-order caller rather than corrupting ownership.
        if !self.has_valid_layout() || self.atoms.nghost != 0 || index >= self.atoms.nlocal as usize
        {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        let expected = self.atoms.len() - 1;
        self.in_place_transaction(
            expected,
            // SAFETY: the core performs the same removal below.
            |store| unsafe { store.swap_remove(index) },
            |atoms| {
                atoms.swap_remove(index);
                atoms.nlocal -= 1;
            },
        )
    }

    /// Remove several locally owned rows in one transaction.
    ///
    /// `indices` must be strictly descending; the rows are swap-removed in that
    /// order, which is exactly the result of calling [`swap_remove`](Self::swap_remove)
    /// once per index while scanning the store from the back (the migration
    /// pattern). Stores without row-local hooks are snapshotted once for the
    /// whole batch instead of once per removed row.
    pub fn swap_remove_descending(&mut self, indices: &[usize]) -> Result<(), ParticleStoreError> {
        if indices.is_empty() {
            return Ok(());
        }
        let nlocal = self.atoms.nlocal as usize;
        if !self.has_valid_layout()
            || self.atoms.nghost != 0
            || indices[0] >= nlocal
            || indices.windows(2).any(|w| w[0] <= w[1])
        {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        let expected = self.atoms.len() - indices.len();
        self.in_place_transaction(
            expected,
            |store| {
                for &index in indices {
                    // SAFETY: the core performs the same removals below.
                    unsafe { store.swap_remove(index) };
                }
            },
            |atoms| {
                for &index in indices {
                    atoms.swap_remove(index);
                }
                atoms.nlocal -= indices.len() as u32;
            },
        )
    }

    /// Drop the ghost suffix from core and extensions together.
    pub fn discard_ghosts(&mut self) -> Result<(), ParticleStoreError> {
        if !self.has_valid_layout() {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        let nlocal = self.atoms.nlocal as usize;
        self.in_place_transaction(
            nlocal,
            // SAFETY: the core truncates to the same row count below.
            |store| unsafe { store.truncate(nlocal) },
            |atoms| {
                atoms.truncate_to(nlocal);
                atoms.nghost = 0;
            },
        )
    }

    /// Begin a full ghost rebuild with a local-only, aligned particle prefix.
    ///
    /// This is the one communication boundary that may reset the ownership
    /// counters.  It first drops an existing ghost suffix transactionally, then
    /// derives `nlocal` from the surviving core rows and clears `nghost`.
    /// Callers therefore never have to update these coupled counters beside the
    /// core/extension structural operation.
    pub fn begin_ghost_rebuild(&mut self) -> Result<(), ParticleStoreError> {
        if !self.has_valid_layout() {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        if self.atoms.nghost > 0 {
            self.discard_ghosts()?;
        }
        self.atoms.nlocal = self.atoms.len() as u32;
        self.atoms.nghost = 0;
        Ok(())
    }

    /// Apply a validated local permutation to all parallel columns.
    pub fn apply_permutation(
        &mut self,
        permutation: &[usize],
        scratch: &mut Vec<bool>,
    ) -> Result<(), ParticleStoreError> {
        let n = self.atoms.nlocal as usize;
        if !self.has_valid_layout() || permutation.len() != n {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        scratch.clear();
        scratch.resize(n, false);
        for &source in permutation {
            if source >= n || std::mem::replace(&mut scratch[source], true) {
                return Err(ParticleStoreError::InvalidStructuralOperation);
            }
        }
        let rows = self.atoms.len();
        let applied = self.registry.for_each_store_in_place(rows, |store| {
            // SAFETY: the same validated permutation is applied to the core below.
            unsafe { store.apply_permutation_with_scratch(permutation, n, scratch) }
        });
        if !applied || !self.registry.validate_rows(rows) {
            return Err(ParticleStoreError::MalformedExtensionRecord);
        }
        self.atoms
            .apply_permutation_with_scratch(permutation, n, scratch);
        Ok(())
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
        if !self.has_valid_layout() || index >= self.atoms.nlocal as usize {
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
        if !self.has_valid_layout() || index >= self.atoms.len() {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        Self::frame_record(buf, |buf| {
            self.atoms.pack_border(index, change_pos, vel_offset, buf);
            self.registry.pack_border_all(index, buf);
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
            .unpack_one(rows_before, &record[ATOM_PACK_SIZE..], ghost)?;
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
        if !self.has_valid_layout() || (!ghost && self.atoms.nghost != 0) {
            return Err(ParticleStoreError::InvalidStructuralOperation);
        }
        let rows_before = self.atoms.len();
        let (nlocal_before, nghost_before) = (self.atoms.nlocal, self.atoms.nghost);
        // Appends only add rows, so the core is rolled back by truncation and
        // row-local extensions likewise; only stores without row-local hooks
        // need a snapshot (none, for `#[derive(AtomData)]` stores).
        let extension_before = self.registry.rollback_snapshot();
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
        self.atoms.truncate_to(rows_before);
        self.atoms.nlocal = nlocal_before;
        self.atoms.nghost = nghost_before;
        self.registry.rollback_append(extension_before, rows_before);
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
    debug_assert_particle_invariant(&atoms, &registry, "pre-rebuild ghost removal");
    ParticleStore::new(&mut atoms, &registry)
        .discard_ghosts()
        .expect("AtomPlugin found misaligned particle columns");
    debug_assert_particle_invariant(&atoms, &registry, "post-rebuild ghost removal");
}

fn zero_all_forces(mut atoms: ResMut<Atom>, registry: Res<AtomDataRegistry>) {
    debug_assert_particle_invariant(&atoms, &registry, "force reset");
    let n = atoms.len();
    atoms
        .force
        .iter_mut()
        .take(n)
        .for_each(|force| *force = [0.0; 3]);
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

    #[derive(Default, AtomData)]
    struct OwnerOnlyRows {
        #[forward]
        omega: Vec<[f64; 3]>,
        #[no_ghost]
        history: Vec<[f64; 4]>,
        #[no_ghost]
        #[reverse]
        torque: Vec<[f64; 3]>,
        radius: Vec<f64>,
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
        unsafe fn push_default(&mut self) {
            self.rows.push(-1.0);
        }
        unsafe fn truncate(&mut self, n: usize) {
            self.rows.truncate(n);
        }
        unsafe fn swap_remove(&mut self, i: usize) {
            self.rows.swap_remove(i);
        }
        fn pack(&self, i: usize, buf: &mut Vec<f64>) {
            buf.push(self.rows[i]);
        }
        unsafe fn unpack(&mut self, buf: &[f64]) -> usize {
            self.rows.push(buf[0]);
            1
        }
        unsafe fn apply_permutation(&mut self, perm: &[usize], n: usize) {
            let mut visited = Vec::new();
            unsafe { self.apply_permutation_with_scratch(perm, n, &mut visited) };
        }
        unsafe fn apply_permutation_with_scratch(
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
        unsafe fn push_default(&mut self) {
            self.rows.push(0.0);
        }
        unsafe fn truncate(&mut self, n: usize) {
            self.rows.truncate(n);
        }
        unsafe fn swap_remove(&mut self, i: usize) {
            self.rows.swap_remove(i);
            panic!("simulated legacy extension failure");
        }
        fn pack(&self, i: usize, buf: &mut Vec<f64>) {
            buf.push(self.rows[i]);
        }
        unsafe fn unpack(&mut self, buf: &[f64]) -> usize {
            self.rows.push(buf[0]);
            1
        }
        unsafe fn apply_permutation(&mut self, _perm: &[usize], _n: usize) {}
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
        unsafe fn push_default(&mut self) {
            self.rows.push(0.0);
        }
        unsafe fn truncate(&mut self, n: usize) {
            self.rows.truncate(n);
        }
        unsafe fn swap_remove(&mut self, i: usize) {
            self.rows.swap_remove(i);
        }
        fn pack(&self, i: usize, buf: &mut Vec<f64>) {
            buf.push(self.rows[i]);
        }
        unsafe fn unpack(&mut self, buf: &[f64]) -> usize {
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
        unsafe fn apply_permutation(&mut self, _perm: &[usize], _n: usize) {}
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
        unsafe fn push_default(&mut self) {
            self.rows.push(0.0);
        }
        unsafe fn truncate(&mut self, n: usize) {
            self.rows.truncate(n);
        }
        unsafe fn swap_remove(&mut self, i: usize) {
            self.rows.swap_remove(i);
        }
        fn pack(&self, i: usize, buf: &mut Vec<f64>) {
            buf.push(self.rows[i]);
        }
        unsafe fn unpack(&mut self, _buf: &[f64]) -> usize {
            self.rows.push(-999.0);
            panic!("simulated decoder failure")
        }
        unsafe fn apply_permutation(&mut self, _perm: &[usize], _n: usize) {}
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
        unsafe fn push_default(&mut self) {
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
        unsafe fn truncate(&mut self, n: usize) {
            self.scalar.truncate(n);
            self.vector.truncate(n);
        }
        unsafe fn swap_remove(&mut self, i: usize) {
            self.scalar.swap_remove(i);
            self.vector.swap_remove(i);
        }
        fn pack(&self, i: usize, buf: &mut Vec<f64>) {
            buf.push(self.scalar[i]);
            buf.extend_from_slice(&self.vector[i]);
        }
        unsafe fn unpack(&mut self, buf: &[f64]) -> usize {
            self.scalar.push(buf[0]);
            self.vector.push([buf[1], buf[2], buf[3]]);
            4
        }
        unsafe fn apply_permutation(&mut self, perm: &[usize], n: usize) {
            let mut visited = Vec::new();
            unsafe { self.apply_permutation_with_scratch(perm, n, &mut visited) };
        }
        unsafe fn apply_permutation_with_scratch(
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
        unsafe fn push_default(&mut self) {}
        unsafe fn truncate(&mut self, _: usize) {}
        unsafe fn swap_remove(&mut self, _: usize) {}
        fn pack(&self, _: usize, _: &mut Vec<f64>) {}
        unsafe fn unpack(&mut self, _: &[f64]) -> usize {
            0
        }
        unsafe fn apply_permutation(&mut self, _: &[usize], _: usize) {}
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
        store.begin_ghost_rebuild().unwrap();
        store.swap_remove(0).unwrap();

        assert_eq!(store.atoms.tag, vec![10]);
        assert_eq!(registry.get::<ManualRows>().unwrap().rows, vec![100.0]);
        assert_eq!((store.atoms.nlocal, store.atoms.nghost), (1, 0));
        assert!(registry.validate_rows(store.atoms.len()));
    }

    #[test]
    fn derived_stores_declare_row_local_hooks_and_skip_snapshots() {
        let mut registry = AtomDataRegistry::new();
        registry.try_register(DerivedRows::default(), 0).unwrap();
        assert!(DerivedRows::default().row_local_structural_hooks());
        assert!(!ManualRows::default().row_local_structural_hooks());
        assert!(registry.rollback_snapshot().is_none());
        registry.try_register(ManualRows::default(), 0).unwrap();
        let snapshot = registry
            .rollback_snapshot()
            .expect("legacy store is snapshotted");
        assert_eq!(snapshot.len(), 2);
        assert!(snapshot[0].is_none(), "derived store must not be copied");
        assert!(snapshot[1].is_some());
    }

    #[test]
    fn batched_swap_remove_matches_sequential_backward_removal() {
        let build = || {
            let mut atoms = Atom::new();
            for tag in 0..8u32 {
                atoms.push_test_atom(tag, [tag as f64, 0.0, 0.0], 1.0, 1.0);
            }
            atoms.nlocal = 8;
            let mut registry = AtomDataRegistry::new();
            registry
                .try_register(
                    DerivedRows {
                        scalar: (0..8).map(|i| i as f64).collect(),
                        vector: (0..8).map(|i| [i as f64; 3]).collect(),
                    },
                    8,
                )
                .unwrap();
            registry
                .try_register(
                    ManualRows {
                        rows: (0..8).map(|i| 10.0 * i as f64).collect(),
                    },
                    8,
                )
                .unwrap();
            (atoms, registry)
        };
        let removed = [7, 5, 4, 1];

        let (mut seq_atoms, seq_registry) = build();
        for &i in &removed {
            ParticleStore::new(&mut seq_atoms, &seq_registry)
                .swap_remove(i)
                .unwrap();
        }
        let (mut batch_atoms, batch_registry) = build();
        ParticleStore::new(&mut batch_atoms, &batch_registry)
            .swap_remove_descending(&removed)
            .unwrap();

        assert_eq!(batch_atoms.tag, seq_atoms.tag);
        assert_eq!(batch_atoms.nlocal, 4);
        assert_eq!(
            batch_registry.get::<DerivedRows>().unwrap().scalar,
            seq_registry.get::<DerivedRows>().unwrap().scalar
        );
        assert_eq!(
            batch_registry.get::<ManualRows>().unwrap().rows,
            seq_registry.get::<ManualRows>().unwrap().rows
        );
        // Unsorted or out-of-range batches are rejected without mutation.
        assert_eq!(
            ParticleStore::new(&mut batch_atoms, &batch_registry).swap_remove_descending(&[1, 2]),
            Err(ParticleStoreError::InvalidStructuralOperation)
        );
        assert_eq!(batch_atoms.nlocal, 4);
    }

    #[test]
    fn row_local_store_is_untouched_when_a_legacy_store_panics() {
        let mut atoms = Atom::new();
        atoms.push_test_atom(10, [0.0; 3], 1.0, 1.0);
        atoms.push_test_atom(20, [1.0; 3], 1.0, 1.0);
        atoms.nlocal = 2;
        let mut registry = AtomDataRegistry::new();
        // Registered first, so a naive in-order loop would already have
        // removed its row before the legacy store panics.
        registry
            .try_register(
                DerivedRows {
                    scalar: vec![1.0, 2.0],
                    vector: vec![[1.0; 3], [2.0; 3]],
                },
                2,
            )
            .unwrap();
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
        assert_eq!(
            registry.get::<DerivedRows>().unwrap().scalar,
            vec![1.0, 2.0]
        );
        assert_eq!(
            registry.get::<PanickingRows>().unwrap().rows,
            vec![4.0, 9.0]
        );
        assert!(registry.validate_rows(atoms.len()));
    }

    #[test]
    fn derived_receive_rolls_back_by_truncation() {
        let mut atoms = Atom::new();
        atoms.push_test_atom(10, [0.0; 3], 1.0, 1.0);
        atoms.nlocal = 1;
        let mut registry = AtomDataRegistry::new();
        registry
            .try_register(
                DerivedRows {
                    scalar: vec![3.0],
                    vector: vec![[3.0; 3]],
                },
                1,
            )
            .unwrap();
        let mut payload = Vec::new();
        ParticleStore::new(&mut atoms, &registry)
            .pack_migrant(0, &mut payload)
            .unwrap();
        // Two good records followed by a truncated third: the whole batch must
        // be rejected and the store left exactly as it was.
        let mut batch = payload.clone();
        batch.extend_from_slice(&payload);
        batch.extend_from_slice(&payload[..payload.len() - 2]);
        assert!(ParticleStore::new(&mut atoms, &registry)
            .append_migrant_records(&batch, 3)
            .is_err());
        assert_eq!(atoms.tag, vec![10]);
        assert_eq!((atoms.nlocal, atoms.nghost), (1, 0));
        assert_eq!(registry.get::<DerivedRows>().unwrap().scalar, vec![3.0]);
        assert!(registry.validate_rows(1));

        // And a good batch still lands.
        ParticleStore::new(&mut atoms, &registry)
            .append_migrant_records(&payload, 1)
            .unwrap();
        assert_eq!(atoms.tag, vec![10, 10]);
        assert_eq!(
            registry.get::<DerivedRows>().unwrap().scalar,
            vec![3.0, 3.0]
        );
    }

    #[test]
    fn no_ghost_fields_skip_border_records_and_zero_ghost_rows() {
        let mut atoms = Atom::new();
        atoms.push_test_atom(7, [0.5; 3], 1.0, 1.0);
        atoms.nlocal = 1;
        let mut registry = AtomDataRegistry::new();
        registry
            .try_register(
                OwnerOnlyRows {
                    omega: vec![[1.0, 2.0, 3.0]],
                    history: vec![[4.0, 5.0, 6.0, 7.0]],
                    torque: vec![[8.0, 9.0, 10.0]],
                    radius: vec![0.25],
                },
                1,
            )
            .unwrap();

        let mut border = Vec::new();
        let mut migrant = Vec::new();
        {
            let mut store = ParticleStore::new(&mut atoms, &registry);
            store.pack_border(0, [0.0; 3], [0.0; 3], &mut border).unwrap();
            store.pack_migrant(0, &mut migrant).unwrap();
        }
        // Border records drop the 4 + 3 owner-only values.
        assert_eq!(migrant.len() - border.len(), 7);

        ParticleStore::new(&mut atoms, &registry)
            .append_ghost_records(&border, 1)
            .unwrap();
        assert_eq!((atoms.nlocal, atoms.nghost), (1, 1));
        {
            let rows = registry.get::<OwnerOnlyRows>().unwrap();
            assert_eq!(rows.omega[1], [1.0, 2.0, 3.0]);
            assert_eq!(rows.radius[1], 0.25);
            assert_eq!(rows.history[1], [0.0; 4]);
            assert_eq!(rows.torque[1], [0.0; 3]);
        }

        // Migrants still carry every field.
        let mut fresh = Atom::new();
        let mut fresh_registry = AtomDataRegistry::new();
        fresh_registry.try_register(OwnerOnlyRows::default(), 0).unwrap();
        ParticleStore::new(&mut fresh, &fresh_registry)
            .append_migrant_records(&migrant, 1)
            .unwrap();
        let rows = fresh_registry.get::<OwnerOnlyRows>().unwrap();
        assert_eq!(rows.history[0], [4.0, 5.0, 6.0, 7.0]);
        assert_eq!(rows.torque[0], [8.0, 9.0, 10.0]);
    }

    #[test]
    fn batched_comm_rows_match_per_row_hooks() {
        let mut registry = AtomDataRegistry::new();
        registry
            .try_register(
                DerivedRows {
                    scalar: vec![1.0, 2.0, 3.0, 0.0, 0.0],
                    vector: vec![
                        [1.0, 1.5, 2.0],
                        [2.0, 2.5, 3.0],
                        [3.0, 3.5, 4.0],
                        [0.5; 3],
                        [0.25; 3],
                    ],
                },
                5,
            )
            .unwrap();
        let send = [2usize, 0];
        let mut batched = Vec::new();
        registry.pack_forward_rows(&send, &mut batched);
        let mut per_row = Vec::new();
        for &i in &send {
            registry.pack_forward_all(i, &mut per_row);
        }
        assert_eq!(batched, per_row);
        assert_eq!(registry.unpack_forward_rows(3, 2, &batched), 2);
        assert_eq!(
            registry.get::<DerivedRows>().unwrap().scalar,
            vec![1.0, 2.0, 3.0, 3.0, 1.0]
        );

        // Reverse: ghost rows 3..5 accumulate onto owners [2, 0].
        let mut rev = Vec::new();
        registry.pack_reverse_rows(3, 2, &mut rev);
        assert_eq!(rev, vec![0.5, 0.5, 0.5, 0.25, 0.25, 0.25]);
        assert_eq!(registry.unpack_reverse_rows(&send, &rev), 6);
        let derived = registry.get::<DerivedRows>().unwrap();
        assert_eq!(derived.vector[2], [3.5, 4.0, 4.5]);
        assert_eq!(derived.vector[0], [1.25, 1.75, 2.25]);
    }

    #[test]
    fn particle_store_default_push_is_atomic_and_sets_authoritative_count() {
        let mut atoms = Atom::new();
        let mut registry = AtomDataRegistry::new();
        registry.try_register(ManualRows::default(), 0).unwrap();

        ParticleStore::new(&mut atoms, &registry)
            .push_default_local(17)
            .unwrap();
        assert_eq!((atoms.nlocal, atoms.nghost, atoms.natoms), (1, 0, 17));
        assert_eq!(atoms.mass, vec![1.0]);
        assert_eq!(atoms.inv_mass, vec![1.0]);
        assert_eq!(registry.get::<ManualRows>().unwrap().rows, vec![-1.0]);

        let mut failing_atoms = Atom::new();
        let mut failing_registry = AtomDataRegistry::new();
        failing_registry.try_register(BrokenDefaults, 0).unwrap();
        assert_eq!(
            ParticleStore::new(&mut failing_atoms, &failing_registry).push_default_local(1),
            Err(ParticleStoreError::MalformedExtensionRecord)
        );
        assert!(failing_atoms.is_empty());
        assert_eq!(
            (
                failing_atoms.nlocal,
                failing_atoms.nghost,
                failing_atoms.natoms
            ),
            (0, 0, 0)
        );
        assert!(failing_registry.validate_rows(0));
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
    fn restart_rejects_a_misaligned_core_replacement_without_mutating_state() {
        let mut atoms = Atom::new();
        atoms.push_test_atom(10, [0.0; 3], 1.0, 1.0);
        atoms.nlocal = 1;
        let registry = AtomDataRegistry::new();

        let mut replacement = Atom::new();
        replacement.push_test_atom(20, [1.0; 3], 1.0, 2.0);
        replacement.nlocal = 1;
        replacement.mass.truncate(0);
        assert!(!replacement.validate_rows());

        assert_eq!(
            ParticleStore::new(&mut atoms, &registry).replace_from_restart(replacement, &[]),
            Err(ParticleStoreError::InvalidStructuralOperation)
        );
        assert_eq!(atoms.tag, vec![10]);
        assert_eq!(atoms.mass, vec![1.0]);
        assert!(atoms.validate_rows());
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
    fn particle_store_rejects_stale_ownership_counters_without_relabelling_rows() {
        let mut atoms = Atom::new();
        atoms.push_test_atom(10, [0.0; 3], 1.0, 1.0);
        atoms.push_test_atom(99, [9.0; 3], 1.0, 1.0);
        // The second row is neither accounted as local nor as a ghost.  Its
        // columns happen to align, but accepting it would silently turn an
        // unowned row into a local one at the next rebuild.
        atoms.nlocal = 1;
        atoms.nghost = 0;
        let registry = AtomDataRegistry::new();
        let before_tags = atoms.tag.clone();
        let before_counts = (atoms.nlocal, atoms.nghost);

        assert_eq!(
            ParticleStore::new(&mut atoms, &registry).begin_ghost_rebuild(),
            Err(ParticleStoreError::InvalidStructuralOperation)
        );
        assert_eq!(atoms.tag, before_tags);
        assert_eq!((atoms.nlocal, atoms.nghost), before_counts);

        let mut wire = Vec::new();
        assert_eq!(
            ParticleStore::new(&mut atoms, &registry).pack_migrant(1, &mut wire),
            Err(ParticleStoreError::InvalidStructuralOperation)
        );
        assert!(wire.is_empty());
    }

    #[test]
    #[should_panic(expected = "particle invariant violated at deliberate test boundary")]
    fn debug_boundary_check_reports_extension_row_desynchronization() {
        let mut atoms = Atom::new();
        atoms.push_test_atom(10, [0.0; 3], 1.0, 1.0);
        atoms.nlocal = 1;
        let mut registry = AtomDataRegistry::new();
        registry
            .try_register(ManualRows { rows: vec![4.0] }, atoms.len())
            .unwrap();

        // This simulates a legacy/out-of-band extension mutation. It is not a
        // supported structural route; the debug boundary identifies it before
        // communication can index a mismatched row.
        registry.get_mut::<ManualRows>().unwrap().rows.push(9.0);
        debug_assert_particle_invariant(&atoms, &registry, "deliberate test boundary");
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
        register_atom_data!(&mut app, DerivedRows::default());
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

    #[test]
    fn complete_local_rows_rejects_one_short_core_column_before_installation() {
        let rows = LocalAtomRows {
            tag: vec![1, 2],
            atom_type: vec![0, 0],
            pos: vec![[0.0; 3]; 2],
            vel: vec![[0.0; 3]; 2],
            force: vec![[0.0; 3]; 2],
            mass: vec![1.0, 1.0],
            cutoff_radius: vec![1.0],
        };

        assert!(matches!(
            Atom::from_local_rows(2, 0.01, rows),
            Err(ParticleStoreError::InvalidStructuralOperation)
        ));
    }

    #[test]
    fn complete_local_rows_derives_nonserialized_columns() {
        let atom = Atom::from_local_rows(
            9,
            0.25,
            LocalAtomRows {
                tag: vec![7],
                atom_type: vec![3],
                pos: vec![[1.0, 2.0, 3.0]],
                vel: vec![[4.0, 5.0, 6.0]],
                force: vec![[7.0, 8.0, 9.0]],
                mass: vec![2.0],
                cutoff_radius: vec![0.5],
            },
        )
        .unwrap();

        assert!(atom.validate_rows());
        assert_eq!(
            (atom.natoms, atom.nlocal, atom.nghost, atom.dt),
            (9, 1, 0, 0.25)
        );
        assert_eq!(atom.inv_mass[..], [0.5]);
        assert_eq!(atom.origin_index[..], [0]);
        assert_eq!(atom.image[..], [[0; 3]]);
        assert_eq!(atom.is_ghost[..], [false]);
    }
}
