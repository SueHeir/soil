//! Integration-level allocation guard for `ParticlesWith`.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    any::{Any, TypeId},
    cell::{Cell, RefCell},
    collections::HashMap,
};

use grass_scheduler::SystemParam;
use soil_core::{Atom, AtomData, AtomDataRegistry, ParticleStore, ParticlesWith, Read, Write};
use soil_derive::AtomData as DeriveAtomData;

struct CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegated unchanged to the system allocator.
        let ptr = unsafe { System.alloc(layout) };
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` originated from this allocator.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Default, DeriveAtomData)]
struct Position {
    values: Vec<f64>,
}

#[derive(Default, DeriveAtomData)]
struct Velocity {
    values: Vec<f64>,
}

#[test]
fn particles_with_hot_loop_allocates_nothing() {
    let mut atoms = Atom::new();
    let mut registry = AtomDataRegistry::new();
    {
        let mut store = ParticleStore::new(&mut atoms, &registry);
        store.push_default_local(1).unwrap();
    }
    registry
        .register_for_atoms(Position { values: vec![1.0] }, &atoms)
        .unwrap();
    registry
        .register_for_atoms(Velocity { values: vec![2.0] }, &atoms)
        .unwrap();
    let resources = vec![RefCell::new(Box::new(registry) as Box<dyn Any>)];
    let index = 0;
    let query = || {
        // `with` creates only RefCell-backed SoA views; it neither copies nor
        // allocates extension columns.
        let views = <ParticlesWith<'_, (Write<Position>, Read<Velocity>)> as SystemParam>::retrieve(
            &resources,
            index,
            std::ptr::null_mut::<HashMap<TypeId, Box<dyn Any>>>(),
        );
        views.with(|(mut position, velocity)| position.values[0] += velocity.values[0]);
    };

    // Warm up TLS and the closure before recording its allocation count.
    query();
    ALLOCATIONS.with(|count| count.set(0));
    for _ in 0..1_000 {
        query();
    }
    assert_eq!(ALLOCATIONS.with(Cell::get), 0, "query hot loop allocated");
}
