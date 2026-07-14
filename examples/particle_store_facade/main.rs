//! Executable validation and representative communication-path measurement for
//! `ParticleStore`.  It is intentionally a generic scalar extension rather
//! than a method-specific particle type.
use soil_core::{Atom, AtomData, AtomDataRegistry, ParticleStore, ParticleStoreError};
use soil_derive::AtomData as DeriveAtomData;
use std::{env, fs, time::Instant};

#[derive(Default, DeriveAtomData)]
struct Marker {
    value: Vec<f64>,
}

fn config_value(source: &str, key: &str) -> usize {
    source
        .lines()
        .find_map(|line| line.trim().strip_prefix(&format!("{key} = ")))
        .unwrap_or_else(|| panic!("config contains {key}"))
        .parse()
        .unwrap_or_else(|_| panic!("{key} is an unsigned integer"))
}

fn source(count: usize) -> (Atom, AtomDataRegistry, Vec<f64>) {
    let mut atoms = Atom::new();
    let mut registry = AtomDataRegistry::new();
    registry
        .register_for_atoms(Marker::default(), &atoms)
        .unwrap();
    {
        let mut store = ParticleStore::new(&mut atoms, &registry);
        store.reserve(count);
    }
    for i in 0..count {
        {
            let mut store = ParticleStore::new(&mut atoms, &registry);
            store.push_default_local(count as u64).unwrap();
        }
        // The facade owns structural defaults/counters.  A caller still fills
        // scientific values through direct SoA slices afterwards.
        atoms.tag[i] = (1000 + i) as u32;
        atoms.pos[i] = [i as f64, 0.0, 0.0];
    }
    for (i, value) in registry
        .get_mut::<Marker>()
        .unwrap()
        .value
        .iter_mut()
        .enumerate()
    {
        *value = 0.25 + i as f64;
    }
    let mut payload = Vec::new();
    {
        let store = ParticleStore::new(&mut atoms, &registry);
        for i in 0..count {
            store.pack_migrant(i, &mut payload).unwrap();
        }
    }
    (atoms, registry, payload)
}

fn receive(payload: &[f64], count: usize) -> (Atom, AtomDataRegistry) {
    let mut atoms = Atom::new();
    let mut registry = AtomDataRegistry::new();
    registry
        .register_for_atoms(Marker::default(), &atoms)
        .unwrap();
    ParticleStore::new(&mut atoms, &registry)
        .append_migrant_records(payload, count)
        .unwrap();
    (atoms, registry)
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/particle_store_facade/config.toml".to_string());
    let config = fs::read_to_string(path).expect("read facade validation config");
    let count = config_value(&config, "particle_count");
    let repetitions = config_value(&config, "repetitions");
    assert!(count > 1 && repetitions > 0);

    let (source_atoms, source_registry, payload) = source(count);
    let (mut received, received_registry) = receive(&payload, count);
    let parity = received.tag == source_atoms.tag
        && received.pos == source_atoms.pos
        && received_registry.get::<Marker>().unwrap().value
            == source_registry.get::<Marker>().unwrap().value
        && received.nlocal() as usize == count
        && received.nghost() == 0;
    println!(
        "PARTICLE_STORE case=framed_multi_record_parity expected={count} observed={} pass={parity}",
        received.len()
    );
    assert!(
        parity,
        "framed migration must preserve core and extension order"
    );

    let before_tags = received.tag.clone();
    let before_markers = received_registry.get::<Marker>().unwrap().value.clone();
    let mut malformed = payload.clone();
    malformed.push(1.0);
    let rolled_back = matches!(
        ParticleStore::new(&mut received, &received_registry)
            .append_ghost_records(&malformed, count),
        Err(ParticleStoreError::MalformedExtensionRecord)
    ) && received.tag == before_tags
        && received_registry.get::<Marker>().unwrap().value == before_markers
        && received_registry.validate_rows(received.len());
    println!("PARTICLE_STORE case=trailing_payload_rollback expected={count} observed={} pass={rolled_back}", received.len());
    assert!(
        rolled_back,
        "failed batched receive must restore every parallel row"
    );

    let started = Instant::now();
    for _ in 0..repetitions {
        let (atoms, registry) = receive(&payload, count);
        assert!(registry.validate_rows(atoms.len()));
    }
    let elapsed = started.elapsed().as_nanos() as u64;
    let rows = count * repetitions;
    println!("PARTICLE_STORE case=facade_comm_measurement expected={rows} observed={rows} elapsed_ns={elapsed} pass=true");
}
