//! Appending particles one at a time must scale linearly (it used to clone the
//! whole store on every push, which made file loading O(N^2)).

use std::time::Instant;

use soil_core::{Atom, AtomDataRegistry, ParticleStore};

fn push_n(n: usize) -> f64 {
    let mut atoms = Atom::new();
    let registry = AtomDataRegistry::new();
    let t0 = Instant::now();
    let mut store = ParticleStore::new(&mut atoms, &registry);
    for _ in 0..n {
        store.push_default_local(n as u64).unwrap();
    }
    let dt = t0.elapsed().as_secs_f64();
    assert_eq!(atoms.nlocal() as usize, n);
    dt
}

#[test]
fn push_default_local_scales_linearly() {
    let small = push_n(20_000).max(1e-6);
    let large = push_n(80_000);
    // Linear would be ~4x; the old O(N^2) behaviour was ~16x. Allow lots of
    // slack for noisy CI but catch a quadratic regression.
    assert!(
        large / small < 10.0,
        "80k pushes took {large:.4}s vs {small:.4}s for 20k (ratio {:.1})",
        large / small
    );
}
