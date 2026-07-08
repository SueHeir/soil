# Testing Substrate-Based Physics

This page is for authors building a particle method on SOIL: MD, SPH,
peridynamics, DEM, or another off-lattice method. The goal is a small App test
that exercises the substrate path your method relies on without putting
method-specific setup into SOIL itself.

For a runnable bare assembly, start with
[`examples/minimal_sim`](../../../examples/minimal_sim/README.md). It shows the
four substrate plugins, manual atom seeding, and walking the neighbor pair list.
The pattern below adds the pieces a downstream physics tier usually needs:
declarative config, a registered `AtomData` column, setup systems, update
systems, and schedule warnings.

## The minimal test shape

Use `Config::from_str` to seed TOML in the App, then add `InputPlugin`. When a
`Config` resource already exists, `InputPlugin` skips CLI parsing, so tests stay
programmatic while still using the same config resource as a run from a file.

Add substrate plugins explicitly:

- `AtomPlugin` for `Atom` plus `AtomDataRegistry`.
- `DomainPlugin` for the simulation box and periodic boundaries.
- `NeighborPlugin` for pair lists and skin/ghost-cutoff setup.
- `CommunicationPlugin` for migration, ghost exchange, and reverse
  accumulation. In serial tests, its default backend is enough.

Register method state through `register_atom_data!` after `AtomPlugin` has
installed the registry. Seed local particles with `Atom::push_test_atom`, then
set `nlocal` and `natoms`; `push_test_atom` appends base columns only, so any
registered extension columns must be extended to the same local length.

Schedule systems into the phase that matches their role:

- `ScheduleSetupSet::Setup` reads declarative config and initializes resources
  that do not depend on finalized atoms or domain-derived values.
- `ScheduleSetupSet::PostSetup` computes state after the substrate setup has
  populated the domain, neighbor parameters, and other plugin resources.
- `ParticleSimScheduleSet::*` phases run during each timestep. Most force laws
  use `ParticleSimScheduleSet::Force`; setup or pre-force bookkeeping usually
  belongs in `PreForce`, `PreNeighbor`, or a documented neighboring phase.

For Verlet-style particle loops, register `verlet_schedule_warnings` with the
App. It lets the generic scheduler emit particle-specific warnings such as
"Force is empty" or "only one half of Verlet is present" without teaching GRASS
about particles.

## Complete copyable test

This is intentionally method-neutral. Replace `Marker` and `mark_pairs` with
your method's per-particle columns and force/update system.

```rust
use std::any::TypeId;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use soil_core::{
    register_atom_data, verlet_schedule_warnings, Atom, AtomData, AtomDataRegistry, AtomPlugin,
    CommunicationPlugin, Config, DomainPlugin, InputPlugin, Neighbor, NeighborPlugin,
    ParticleSimScheduleSet, ScheduleSetupSet,
};
use soil_derive::AtomData;

#[derive(AtomData, Default)]
struct Marker {
    // Read-only state neighbors need; copied onto ghosts by forward comm.
    #[forward]
    phase: Vec<f64>,

    // A contribution that may be written on ghosts and accumulated back.
    #[reverse]
    #[zero]
    work: Vec<f64>,
}

#[derive(Default)]
struct SeenSetup(bool);

fn setup_flag(mut seen: ResMut<SeenSetup>) {
    seen.0 = true;
}

fn post_setup_check(seen: Res<SeenSetup>) {
    assert!(seen.0, "Setup systems run before PostSetup systems");
}

fn mark_pairs(
    atoms: Res<Atom>,
    neighbor: Res<Neighbor>,
    registry: Res<AtomDataRegistry>,
) {
    let nlocal = atoms.nlocal as usize;
    let mut marker = registry
        .get_mut::<Marker>()
        .expect("Marker was registered before prepare()");

    for (i, j) in neighbor.pairs(nlocal) {
        marker.work[i] += marker.phase[j];
        if j < nlocal {
            marker.work[j] += marker.phase[i];
        }
    }
}

#[test]
fn substrate_app_test_exercises_atomdata_and_neighbors() {
    let toml = r#"
[domain]
x_low = 0.0
x_high = 1.0
y_low = 0.0
y_high = 1.0
z_low = 0.0
z_high = 1.0
boundary_x = "periodic"
boundary_y = "periodic"
boundary_z = "periodic"

[neighbor]
skin_fraction = 1.2
"#;

    let mut app = App::new();
    app.add_resource(Config::from_str(toml));
    app.add_resource(SeenSetup::default());

    app.add_plugins(InputPlugin)
        .add_plugins(AtomPlugin)
        .add_plugins(DomainPlugin)
        .add_plugins(NeighborPlugin)
        .add_plugins(CommunicationPlugin);

    register_atom_data!(app, Marker::default());

    app.add_setup_system(setup_flag, ScheduleSetupSet::Setup);
    app.add_setup_system(post_setup_check, ScheduleSetupSet::PostSetup);
    app.add_update_system(mark_pairs, ParticleSimScheduleSet::Force);
    app.set_warning_fn(verlet_schedule_warnings);

    {
        let cell = app
            .get_mut_resource(TypeId::of::<Atom>())
            .expect("AtomPlugin registers Atom");
        let mut binder = cell.borrow_mut();
        let atoms = binder.downcast_mut::<Atom>().unwrap();

        atoms.push_test_atom(10, [0.25, 0.5, 0.5], 0.30, 1.0);
        atoms.push_test_atom(20, [0.50, 0.5, 0.5], 0.30, 1.0);
        atoms.nlocal = 2;
        atoms.natoms = 2;
    }

    {
        let cell = app
            .get_mut_resource(TypeId::of::<AtomDataRegistry>())
            .expect("AtomPlugin registers AtomDataRegistry");
        let mut binder = cell.borrow_mut();
        let registry = binder.downcast_mut::<AtomDataRegistry>().unwrap();
        let mut marker = registry.get_mut::<Marker>().unwrap();

        marker.phase.extend([2.0, 3.0]);
        marker.work.resize(2, 0.0);
    }

    app.prepare();
    app.run();

    let registry = app.get_resource_ref::<AtomDataRegistry>().unwrap();
    let marker = registry.get::<Marker>().unwrap();

    assert_eq!(&marker.work[..2], &[3.0, 2.0]);
}
```

## Keeping tests honest

Keep test fixtures small and declarative. Inline TOML is fine for unit tests;
example configs should live as `examples/<name>/config.toml`, not as scripts.
Use `push_test_atom` only for hand-built tests and examples where the atom
population is part of the fixture. For production-style examples, prefer a
plain config/data file plus the same plugin path the user will run.

When a test needs a new per-particle quantity, put it in a downstream
`AtomData` extension. Do not add method-specific fields to `Atom`; SOIL's base
columns are only the state every particle method shares.
