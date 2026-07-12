//! Typed, zero-copy access to registered particle extensions in systems.
//!
//! [`ParticlesWith`] is deliberately a *view factory*, not a container for
//! [`Ref`]s.  A scheduler system calls [`ParticlesWith::with`] and receives
//! its requested structure-of-arrays views for the duration of that closure.
//! That makes the nested extension borrows provably shorter than the scheduler
//! resource borrow without copying columns or extending a `RefCell` lifetime.

use std::{
    any::{Any, TypeId},
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    marker::PhantomData,
};

use grass_scheduler::{AccessKind, Res, SystemParam};

use crate::{AtomData, AtomDataRegistry};

/// Shared access to a particle extension in [`ParticlesWith`].
pub struct Read<T>(PhantomData<T>);
/// Exclusive access to a particle extension in [`ParticlesWith`].
pub struct Write<T>(PhantomData<T>);
/// An extension access that is absent when its extension was not registered.
pub struct Optional<Q>(PhantomData<Q>);

#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct Requirement {
    id: TypeId,
    name: &'static str,
    write: bool,
    optional: bool,
}

mod private {
    pub trait Sealed {}
}

/// Type-level description of extension views returned by [`ParticlesWith`].
///
/// This trait is sealed: SOIL owns the access declarations so it can validate
/// them before a system runs.  Compose [`Read`], [`Write`], [`Optional`], and
/// tuples (up to eight elements) instead of implementing this trait directly.
pub trait ParticleQuery: private::Sealed + 'static {
    /// The zero-copy SOA view shape supplied to [`ParticlesWith::with`].
    type Item<'a>;

    #[doc(hidden)]
    fn requirements(out: &mut Vec<Requirement>);
    #[doc(hidden)]
    fn is_available(registry: &AtomDataRegistry) -> bool;
    #[doc(hidden)]
    fn has_write() -> bool;
    #[doc(hidden)]
    fn fetch<'a>(registry: &'a AtomDataRegistry) -> Self::Item<'a>;
}

impl<T: AtomData + 'static> private::Sealed for Read<T> {}
impl<T: AtomData + 'static> ParticleQuery for Read<T> {
    type Item<'a> = Ref<'a, T>;

    fn requirements(out: &mut Vec<Requirement>) {
        out.push(Requirement {
            id: TypeId::of::<T>(),
            name: std::any::type_name::<T>(),
            write: false,
            optional: false,
        });
    }
    fn is_available(registry: &AtomDataRegistry) -> bool {
        registry.has_type(TypeId::of::<T>())
    }
    fn has_write() -> bool {
        false
    }
    fn fetch<'a>(registry: &'a AtomDataRegistry) -> Self::Item<'a> {
        registry.expect::<T>("ParticlesWith<Read<_>>")
    }
}

impl<T: AtomData + 'static> private::Sealed for Write<T> {}
impl<T: AtomData + 'static> ParticleQuery for Write<T> {
    type Item<'a> = RefMut<'a, T>;

    fn requirements(out: &mut Vec<Requirement>) {
        out.push(Requirement {
            id: TypeId::of::<T>(),
            name: std::any::type_name::<T>(),
            write: true,
            optional: false,
        });
    }
    fn is_available(registry: &AtomDataRegistry) -> bool {
        registry.has_type(TypeId::of::<T>())
    }
    fn has_write() -> bool {
        true
    }
    fn fetch<'a>(registry: &'a AtomDataRegistry) -> Self::Item<'a> {
        registry.expect_mut::<T>("ParticlesWith<Write<_>>")
    }
}

impl<Q: ParticleQuery> private::Sealed for Optional<Q> {}
impl<Q: ParticleQuery> ParticleQuery for Optional<Q> {
    type Item<'a> = Option<Q::Item<'a>>;

    fn requirements(out: &mut Vec<Requirement>) {
        let start = out.len();
        Q::requirements(out);
        for requirement in &mut out[start..] {
            requirement.optional = true;
        }
    }
    fn is_available(registry: &AtomDataRegistry) -> bool {
        Q::is_available(registry)
    }
    fn has_write() -> bool {
        Q::has_write()
    }
    fn fetch<'a>(registry: &'a AtomDataRegistry) -> Self::Item<'a> {
        Q::is_available(registry).then(|| Q::fetch(registry))
    }
}

macro_rules! tuple_query {
    ($($name:ident),+) => {
        impl<$($name: ParticleQuery),+> private::Sealed for ($($name,)+) {}
        impl<$($name: ParticleQuery),+> ParticleQuery for ($($name,)+) {
            type Item<'a> = ($($name::Item<'a>,)+);

            fn requirements(out: &mut Vec<Requirement>) { $($name::requirements(out);)+ }
            fn is_available(registry: &AtomDataRegistry) -> bool { $($name::is_available(registry))&&+ }
            fn has_write() -> bool { false $(|| $name::has_write())+ }
            fn fetch<'a>(registry: &'a AtomDataRegistry) -> Self::Item<'a> {
                // Each view is borrowed from a distinct registered extension.
                // `validate` rejects a duplicate type before a system runs.
                ($($name::fetch(registry),)+)
            }
        }
    };
}

tuple_query!(A, B);
tuple_query!(A, B, C);
tuple_query!(A, B, C, D);
tuple_query!(A, B, C, D, E);
tuple_query!(A, B, C, D, E, F);
tuple_query!(A, B, C, D, E, F, G);
tuple_query!(A, B, C, D, E, F, G, H);

/// A system parameter that exposes typed, zero-copy SOA extension views.
///
/// ```no_run
/// use soil_core::{AtomData, ParticlesWith, Read, Write};
/// use soil_derive::AtomData as DeriveAtomData;
///
/// #[derive(DeriveAtomData)]
/// struct Velocity { value: Vec<f64> }
/// #[derive(DeriveAtomData)]
/// struct Mass { value: Vec<f64> }
///
/// fn integrate(query: ParticlesWith<'_, (Write<Velocity>, Read<Mass>)>) {
///     query.with(|(mut velocity, mass)| {
///         velocity.value[0] += mass.value[0];
///     });
/// }
/// ```
///
/// Required extensions are checked during scheduler preparation.  Repeating an
/// extension in one query is rejected, including read/read, read/write, and
/// write/write combinations.  `Optional<Read<T>>` instead supplies `None` when
/// `T` is not registered.
pub struct ParticlesWith<'a, Q: ParticleQuery> {
    registry: Res<'a, AtomDataRegistry>,
    marker: PhantomData<Q>,
}

impl<Q: ParticleQuery> ParticlesWith<'_, Q> {
    /// Runs `f` with the requested views, without allocating or copying data.
    ///
    /// The views cannot escape `f`; this permits independent `RefCell` borrows
    /// to be combined into one tuple while preserving their actual lifetime.
    pub fn with<R>(&self, f: impl FnOnce(Q::Item<'_>) -> R) -> R {
        f(Q::fetch(&self.registry))
    }
}

impl<Q: ParticleQuery> SystemParam for ParticlesWith<'_, Q> {
    type Item<'a> = ParticlesWith<'a, Q>;

    fn retrieve<'a>(
        resources: &'a [RefCell<Box<dyn Any>>],
        index: usize,
        locals: *mut HashMap<TypeId, Box<dyn Any>>,
    ) -> Self::Item<'a> {
        ParticlesWith {
            registry: Res::<AtomDataRegistry>::retrieve(resources, index, locals),
            marker: PhantomData,
        }
    }

    fn resource_type_id() -> Option<(TypeId, &'static str)> {
        Some((
            TypeId::of::<AtomDataRegistry>(),
            std::any::type_name::<AtomDataRegistry>(),
        ))
    }

    fn access_kind() -> AccessKind {
        if Q::has_write() {
            AccessKind::Write
        } else {
            AccessKind::Read
        }
    }

    fn validate(resources: &[RefCell<Box<dyn Any>>], index: usize) -> Vec<String> {
        let registry = resources[index].borrow();
        let registry = registry.downcast_ref::<AtomDataRegistry>().unwrap();
        let mut requests = Vec::new();
        Q::requirements(&mut requests);
        let mut errors = Vec::new();
        for (i, left) in requests.iter().enumerate() {
            if !left.optional && !registry.has_type(left.id) {
                errors.push(format!(
                    "particle extension `{}` required by ParticlesWith is not registered; add its plugin before this system",
                    left.name
                ));
            }
            if let Some(right) = requests[..i].iter().find(|right| right.id == left.id) {
                errors.push(format!(
                    "duplicate ParticlesWith access to `{}` is not allowed ({}/{})",
                    left.name,
                    if right.write { "write" } else { "read" },
                    if left.write { "write" } else { "read" },
                ));
            }
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use std::{
        any::Any,
        panic::{catch_unwind, AssertUnwindSafe},
    };

    use grass_scheduler::{IntoSystem, Scheduler, System};
    use soil_derive::AtomData;

    use super::*;

    #[derive(Default, AtomData)]
    struct A {
        values: Vec<f64>,
    }
    #[derive(Default, AtomData)]
    struct B {
        values: Vec<f64>,
    }
    #[derive(Default, AtomData)]
    struct C {
        values: Vec<f64>,
    }

    fn registry(with_b: bool) -> AtomDataRegistry {
        let mut registry = AtomDataRegistry::new();
        registry.try_register(A { values: vec![1.0] }, 1).unwrap();
        if with_b {
            registry.try_register(B { values: vec![2.0] }, 1).unwrap();
        }
        registry
    }

    #[test]
    fn tuple_order_and_write_through_are_zero_copy() {
        let registry = registry(true);
        let (mut a, b) = <(Write<A>, Read<B>) as ParticleQuery>::fetch(&registry);
        {
            assert_eq!(a.values, [1.0]);
            assert_eq!(b.values, [2.0]);
            a.values[0] = b.values[0] + 7.0;
        }
        drop((a, b));
        assert_eq!(registry.get::<A>().unwrap().values, [9.0]);
    }

    #[test]
    fn optional_view_is_none_without_its_extension() {
        let absent = registry(false);
        assert!(<Optional<Read<B>> as ParticleQuery>::fetch(&absent).is_none());
        let present = registry(true);
        assert_eq!(
            <Optional<Read<B>> as ParticleQuery>::fetch(&present)
                .unwrap()
                .values,
            [2.0]
        );
    }

    #[test]
    fn preparation_reports_missing_and_every_duplicate_mode() {
        let resources = vec![RefCell::new(Box::new(registry(false)) as Box<dyn Any>)];
        let missing = <ParticlesWith<'_, Read<B>> as SystemParam>::validate(&resources, 0);
        assert_eq!(missing.len(), 1);
        assert!(missing[0].contains("not registered"));
        for error in [
            <ParticlesWith<'_, (Read<A>, Read<A>)> as SystemParam>::validate(&resources, 0),
            <ParticlesWith<'_, (Write<A>, Read<A>)> as SystemParam>::validate(&resources, 0),
            <ParticlesWith<'_, (Write<A>, Write<A>)> as SystemParam>::validate(&resources, 0),
        ] {
            assert_eq!(error.len(), 1);
            assert!(error[0].contains("duplicate"));
        }
    }

    #[test]
    fn scheduler_uses_read_or_write_registry_access_and_rejects_missing_query() {
        fn reader(_: ParticlesWith<'_, Read<A>>) {}
        fn writer(_: ParticlesWith<'_, Write<A>>) {}
        fn missing(_: ParticlesWith<'_, Read<B>>) {}
        let mut index = HashMap::new();
        index.insert(TypeId::of::<AtomDataRegistry>(), 0);
        let resources = vec![RefCell::new(Box::new(registry(true)) as Box<dyn Any>)];
        let mut read_system = reader.into_system();
        assert!(read_system.prepare(&index, &resources).is_empty());
        assert_eq!(read_system.accesses(), &[(0, AccessKind::Read)]);
        let mut write_system = writer.into_system();
        assert!(write_system.prepare(&index, &resources).is_empty());
        assert_eq!(write_system.accesses(), &[(0, AccessKind::Write)]);

        let mut scheduler = Scheduler::default();
        scheduler.add_resource(registry(false));
        scheduler.add_update_system(missing, crate::ParticleSimScheduleSet::Force);
        assert!(catch_unwind(AssertUnwindSafe(|| scheduler.organize_systems())).is_err());
    }

    #[test]
    fn running_system_receives_tuple_and_optional_views() {
        fn update(query: ParticlesWith<'_, (Write<A>, Read<B>, Optional<Read<C>>)>) {
            query.with(|(mut a, b, c)| {
                assert!(c.is_none());
                a.values[0] = b.values[0] + 5.0;
            });
        }
        let mut scheduler = Scheduler::default();
        scheduler.add_resource(registry(true));
        scheduler.add_update_system(update, crate::ParticleSimScheduleSet::Force);
        scheduler.organize_systems();
        scheduler.run();
        let cell = scheduler
            .resource_cell(TypeId::of::<AtomDataRegistry>())
            .unwrap();
        let registry = cell.borrow();
        assert_eq!(
            registry
                .downcast_ref::<AtomDataRegistry>()
                .unwrap()
                .get::<A>()
                .unwrap()
                .values,
            [7.0]
        );
    }
}
