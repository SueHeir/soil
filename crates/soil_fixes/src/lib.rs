//! Method-agnostic position constraints for SOIL.
//!
//! This crate provides fixes that constrain atom *kinematics* using only the
//! base [`Atom`] state (position, velocity, force) — no knowledge of any
//! particular particle method. The DEM-specific fixes (which reach into
//! rotational state such as `omega`/`torque`) live in DIRT's `dirt_fixes`
//! crate instead.
//!
//! # Available Fixes
//!
//! | Fix | TOML key | Description |
//! |-----|----------|-------------|
//! | [`PinDef`] | `[[pin]]` | Hard **translational** position constraint — captures pos at setup, restores every step |
//!
//! # Plugin
//!
//! - [`SoilFixesPlugin`] — registers the `[[pin]]` constraint.
//!
//! # `pin` vs `freeze`
//!
//! `pin` is a *positional* constraint: it captures each atom's setup-time
//! position and **restores** it bit-for-bit every step, so anything that would
//! move the atom (Verlet drift, a stray force) is corrected. It touches only
//! translational state, so it works for any particle method.
//!
//! For full immobilization that *also* freezes rotation, use DIRT's `[[freeze]]`
//! fix, which zeros velocity, force, and (for DEM atoms) angular velocity and
//! torque.

use std::collections::HashMap;

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::Deserialize;

use soil_core::{
    Atom, CommResource, Config, GroupRegistry, ParticleSimScheduleSet, Real, ScheduleSetupSet,
};

/// Pins atoms in a group at their starting positions — a **hard translational
/// constraint**. The captured setup-time position is restored every step, both
/// **before** the Verlet drift step (`PreInitialIntegration`) and **after**
/// forces are computed (`PostForce`), and velocity and force are zeroed.
///
/// This matters for bonded-particle simulations where a pinned atom is the
/// neighbour of a flexible one: any tiny drift of the pinned atom's position
/// during `InitialIntegration` would perturb the bond force on its neighbour.
/// With `PinDef`, the pinned atom is bit-for-bit at its starting position
/// whenever forces are evaluated.
///
/// Position is captured on the **first** step when the group mask is populated
/// (so initial atom migration has settled) and remains fixed thereafter.
///
/// `pin` constrains translation only. To additionally freeze rotation, use
/// DIRT's `[[freeze]]` fix.
///
/// # TOML Configuration
///
/// ```toml
/// [[pin]]
/// group = "anchor"   # (required) name of the atom group to pin
/// ```
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct PinDef {
    /// Name of the atom group to pin.
    pub group: String,
}

/// Runtime state for `[[pin]]` fixes: captured initial positions keyed by
/// group name and then by global atom tag. Population is lazy — filled on the
/// first application for each group.
#[derive(Default)]
pub struct PinState {
    /// `{group_name → {global_tag → initial_pos}}`.
    pub captured: HashMap<String, HashMap<u32, [f64; 3]>>,
}

/// Storage for all configured `[[pin]]` definitions, populated at plugin build
/// time from the TOML config and stored as an [`App`] resource.
pub struct PinRegistry {
    /// All `[[pin]]` definitions.
    pub pins: Vec<PinDef>,
}

/// Plugin that registers the `[[pin]]` translational position constraint.
///
/// Only registers update systems when at least one `[[pin]]` is configured,
/// avoiding per-timestep overhead otherwise.
pub struct SoilFixesPlugin;

impl Plugin for SoilFixesPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# [[pin]]
# group = "anchor"   # hard translational constraint: pos/vel/force restored every step"#,
        )
    }

    fn build(&self, app: &mut App) {
        let config = app
            .get_resource_ref::<Config>()
            .expect("Config resource must exist before SoilFixesPlugin");

        let registry = PinRegistry {
            pins: config.parse_array::<PinDef>("pin"),
        };

        drop(config);

        if registry.pins.is_empty() {
            app.add_resource(registry);
            return;
        }

        app.add_resource(registry)
            .add_resource(PinState::default())
            .add_setup_system(setup_pins, ScheduleSetupSet::PostSetup);

        // PreInitialIntegration: enforce pos/vel/force BEFORE the Verlet drift
        // step, so any would-be drift is cancelled before forces are computed.
        // (On the first step the group mask may not yet be built, in which case
        // the impl is a no-op and the PostForce pass below picks things up.)
        app.add_update_system(apply_pin_pre, ParticleSimScheduleSet::PreInitialIntegration);
        // PostForce: capture (on the first step that has a populated mask) and
        // re-enforce, ensuring FinalIntegration has f=0 and the next step's
        // InitialIntegration has v=0.
        app.add_update_system(apply_pin_post, ParticleSimScheduleSet::PostForce);
    }
}

/// Validates pin group names at setup time and prints a summary on rank 0.
fn setup_pins(registry: Res<PinRegistry>, comm: Res<CommResource>, groups: Res<GroupRegistry>) {
    for f in &registry.pins {
        groups.validate_name(&f.group, "fix pin");
    }
    if comm.rank() != 0 {
        return;
    }
    for f in &registry.pins {
        println!("Fix pin: group='{}' (hard translational constraint)", f.group);
    }
}

/// Shared implementation for `apply_pin_pre` and `apply_pin_post`.
///
/// Pins every atom in a `[[pin]]` group to its captured initial position,
/// zeroing velocity and force. Position is captured **lazily** on the first
/// call where the group mask has been populated. Keyed by global atom tag, so
/// pinned atoms stay correctly attached even after MPI migration.
fn apply_pin_impl(atoms: &mut Atom, registry: &PinRegistry, groups: &GroupRegistry, state: &mut PinState) {
    let nlocal = atoms.nlocal as usize;

    for def in &registry.pins {
        let group = groups.expect(&def.group);
        if group.mask.is_empty() {
            continue;
        }

        if !state.captured.contains_key(&def.group) {
            let mut pinned: HashMap<u32, [f64; 3]> = HashMap::new();
            for i in 0..nlocal {
                if group.mask[i] {
                    // Captured pin positions stored in f64 (lossless across precisions).
                    let p = atoms.pos[i];
                    pinned.insert(atoms.tag[i], [p[0] as f64, p[1] as f64, p[2] as f64]);
                }
            }
            state.captured.insert(def.group.clone(), pinned);
        }

        let pinned = state.captured.get(&def.group).unwrap();
        for i in 0..nlocal {
            if group.mask[i] {
                if let Some(&pos) = pinned.get(&atoms.tag[i]) {
                    atoms.pos[i] = [pos[0] as Real, pos[1] as Real, pos[2] as Real];
                }
                atoms.vel[i] = [0.0; 3];
                atoms.force[i] = [0.0; 3];
            }
        }
    }
}

/// Pre-integration pin enforcement — keeps the Verlet drift step from moving
/// pinned atoms. Runs at [`ParticleSimScheduleSet::PreInitialIntegration`].
fn apply_pin_pre(
    mut atoms: ResMut<Atom>,
    registry: Res<PinRegistry>,
    groups: Res<GroupRegistry>,
    mut state: ResMut<PinState>,
) {
    apply_pin_impl(&mut atoms, &registry, &groups, &mut state);
}

/// Post-force pin enforcement — also does the lazy position capture on the
/// first step the group mask is populated. Runs at
/// [`ParticleSimScheduleSet::PostForce`].
fn apply_pin_post(
    mut atoms: ResMut<Atom>,
    registry: Res<PinRegistry>,
    groups: Res<GroupRegistry>,
    mut state: ResMut<PinState>,
) {
    apply_pin_impl(&mut atoms, &registry, &groups, &mut state);
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use soil_core::group::GroupDef;

    fn make_atoms(n: usize) -> Atom {
        let mut atom = Atom::new();
        for i in 0..n {
            atom.push_test_atom(i as u32, [i as f64, 0.0, 0.0], 0.5, 1.0);
        }
        atom.nlocal = n as u32;
        atom.natoms = n as u64;
        atom.dt = 0.001;
        atom
    }

    fn make_group_registry(name: &str, mask: Vec<bool>) -> GroupRegistry {
        let count = mask.iter().filter(|&&m| m).count();
        let mut registry = GroupRegistry::new();
        registry.groups.push(soil_core::Group {
            name: name.to_string(),
            def: GroupDef {
                name: name.to_string(),
                atom_types: None,
                region: None,
                dynamic: None,
            },
            mask,
            count,
        });
        registry
    }

    fn make_pin_registry(group: &str) -> PinRegistry {
        PinRegistry {
            pins: vec![PinDef { group: group.to_string() }],
        }
    }

    #[test]
    fn pin_captures_initial_position_and_zeros_state() {
        let mut atoms = make_atoms(2);
        atoms.pos[0] = [1.0, 2.0, 3.0];
        atoms.vel[0] = [0.1, 0.2, 0.3];
        atoms.force[0] = [10.0, 20.0, 30.0];

        let groups = make_group_registry("anchor", vec![true, false]);
        let registry = make_pin_registry("anchor");
        let state = PinState::default();

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_resource(state);
        app.add_update_system(apply_pin_post, ParticleSimScheduleSet::PostForce);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        assert_eq!(a.pos[0], [1.0, 2.0, 3.0]);
        assert_eq!(a.vel[0], [0.0, 0.0, 0.0]);
        assert_eq!(a.force[0], [0.0, 0.0, 0.0]);

        let s = app.get_resource_ref::<PinState>().unwrap();
        assert_eq!(s.captured["anchor"].len(), 1);
        assert_eq!(s.captured["anchor"][&a.tag[0]], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn pin_restores_position_from_preloaded_state() {
        let mut atoms = make_atoms(1);
        atoms.pos[0] = [5.123, 5.456, 5.789];
        atoms.vel[0] = [99.0, 99.0, 99.0];
        atoms.force[0] = [42.0, 42.0, 42.0];
        let tag = atoms.tag[0];

        let groups = make_group_registry("anchor", vec![true]);
        let registry = make_pin_registry("anchor");
        let mut state = PinState::default();
        let mut captured = HashMap::new();
        captured.insert(tag, [5.0, 5.0, 5.0]);
        state.captured.insert("anchor".to_string(), captured);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_resource(state);
        app.add_update_system(apply_pin_pre, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        assert_eq!(a.pos[0], [5.0, 5.0, 5.0]);
        assert_eq!(a.vel[0], [0.0, 0.0, 0.0]);
        assert_eq!(a.force[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn pin_lookup_is_tag_based_and_survives_reordering() {
        let mut atoms = make_atoms(2);
        atoms.tag[0] = 20;
        atoms.tag[1] = 10;
        atoms.pos[0] = [999.0, 0.0, 0.0];
        atoms.pos[1] = [7.999, 0.0, 0.0];

        let groups = make_group_registry("anchor", vec![false, true]);
        let registry = make_pin_registry("anchor");
        let mut state = PinState::default();
        let mut captured = HashMap::new();
        captured.insert(10u32, [7.0, 0.0, 0.0]);
        state.captured.insert("anchor".to_string(), captured);

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_resource(state);
        app.add_update_system(apply_pin_pre, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        assert_eq!(a.tag[1], 10, "A's tag preserved");
        assert_eq!(a.pos[1], [7.0, 0.0, 0.0], "pin restored A by tag, not index");
        assert_eq!(a.pos[0], [999.0, 0.0, 0.0], "B untouched (not in pin group)");
    }

    #[test]
    fn pin_noop_when_mask_empty() {
        let mut atoms = make_atoms(1);
        atoms.pos[0] = [1.0, 0.0, 0.0];
        let mut groups = make_group_registry("anchor", vec![]);
        groups.groups[0].mask.clear();
        let registry = make_pin_registry("anchor");
        let state = PinState::default();

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(groups);
        app.add_resource(registry);
        app.add_resource(state);
        app.add_update_system(apply_pin_pre, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let s = app.get_resource_ref::<PinState>().unwrap();
        assert!(s.captured.is_empty(), "no group captured when mask is empty");
        let a = app.get_resource_ref::<Atom>().unwrap();
        assert_eq!(a.pos[0], [1.0, 0.0, 0.0], "pos untouched when mask empty");
    }
}
