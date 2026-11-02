//! Save and restore the particle store through GRASS checkpoints.
//!
//! [`AtomPlugin`](crate::AtomPlugin) registers the local particles, with every
//! [`AtomData`](crate::AtomData) extension, as the checkpoint entry
//! [`PARTICLES_ENTRY`]. That makes `grass_io::save_app` / `restore_app`, and
//! so `save_coupled` / `restore_coupled`, cover any SOIL code without the
//! code doing anything.

use grass_app::prelude::*;
use grass_scheduler::prelude::CurrentState;
use serde::{Deserialize, Serialize};

use crate::{
    Accum, Atom, AtomDataRegistry, CommState, LocalAtomRows, ParticleStore, ParticleStoreError,
    Real,
};

/// Checkpoint entry name of the particle store.
pub const PARTICLES_ENTRY: &str = "soil.particles";

/// The locally owned particles and every registered extension's rows, in the
/// build's own precision so a restore is exact. Ghosts are not saved: a
/// restore asks for a full rebuild, which makes them again.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticleSnapshot {
    /// Global particle count.
    pub natoms: u64,
    /// Time step.
    pub dt: f64,
    /// Tags.
    pub tag: Vec<u32>,
    /// Types.
    pub atom_type: Vec<u32>,
    /// Positions.
    pub pos: Vec<[Real; 3]>,
    /// Velocities.
    pub vel: Vec<[Real; 3]>,
    /// Forces.
    pub force: Vec<[Accum; 3]>,
    /// Masses.
    pub mass: Vec<Real>,
    /// Cutoff radii.
    pub cutoff_radius: Vec<Real>,
    /// Each registered extension's packed rows, in registration order.
    pub extensions: Vec<Vec<f64>>,
}

impl ParticleSnapshot {
    /// Capture the local particles.
    pub fn capture(atoms: &Atom, registry: &AtomDataRegistry) -> Self {
        let n = atoms.nlocal() as usize;
        ParticleSnapshot {
            natoms: atoms.natoms(),
            dt: atoms.dt,
            tag: atoms.tag[..n].to_vec(),
            atom_type: atoms.atom_type[..n].to_vec(),
            pos: atoms.pos[..n].to_vec(),
            vel: atoms.vel[..n].to_vec(),
            force: atoms.force[..n].to_vec(),
            mass: atoms.mass[..n].to_vec(),
            cutoff_radius: atoms.cutoff_radius[..n].to_vec(),
            extensions: registry.pack_all_for_restart(n),
        }
    }

    /// Replace the particle store with this snapshot. Ghosts are dropped;
    /// the caller must ask for a full rebuild before the next force pass.
    pub fn restore(
        self,
        atoms: &mut Atom,
        registry: &AtomDataRegistry,
    ) -> Result<(), ParticleStoreError> {
        ParticleStore::new(atoms, registry).replace_from_restart_rows(
            self.natoms,
            self.dt,
            LocalAtomRows {
                tag: self.tag,
                atom_type: self.atom_type,
                pos: self.pos,
                vel: self.vel,
                force: self.force,
                mass: self.mass,
                cutoff_radius: self.cutoff_radius,
            },
            &self.extensions,
        )
    }
}

/// Register the particle store as the checkpoint entry [`PARTICLES_ENTRY`].
pub(crate) fn register(app: &mut App) {
    use grass_io::CheckpointAppExt;
    app.checkpoint_with(
        PARTICLES_ENTRY,
        |app: &App| {
            let atoms = app.get_resource_ref::<Atom>().expect("Atom");
            let registry = app
                .get_resource_ref::<AtomDataRegistry>()
                .expect("AtomDataRegistry");
            ParticleSnapshot::capture(&atoms, &registry)
        },
        |app: &App, snapshot: ParticleSnapshot| {
            {
                let registry = app
                    .get_resource_ref::<AtomDataRegistry>()
                    .expect("AtomDataRegistry");
                let mut atoms = app.get_resource_mut::<Atom>().expect("Atom");
                snapshot
                    .restore(&mut atoms, &registry)
                    .unwrap_or_else(|e| panic!("restoring `{PARTICLES_ENTRY}`: {e}"));
            }
            // Ghosts and neighbor lists are rebuilt from the restored owners.
            if let Some(mut state) = app.get_resource_mut::<CurrentState<CommState>>() {
                state.0 = CommState::FullRebuild;
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_with(atoms: &[([f64; 3], [Real; 3])]) -> App {
        let mut app = App::new();
        app.add_plugins(crate::AtomPlugin);
        {
            let mut a = app.get_resource_mut::<Atom>().unwrap();
            for (i, (x, v)) in atoms.iter().enumerate() {
                a.push_test_atom(i as u32 + 1, *x, 0.1, 2.0);
                a.vel[i] = *v;
            }
            a.nlocal = atoms.len() as u32;
            a.natoms = atoms.len() as u64;
            a.dt = 1e-4;
        }
        app
    }

    #[test]
    fn particles_round_trip_through_grass_checkpoints() {
        let a = app_with(&[
            ([0.0, 0.1, 0.2], [1.0, 0.0, -0.5]),
            ([0.3, 0.4, 0.5], [0.0, 2.0, 0.0]),
        ]);
        // One ghost row, which is not saved.
        a.get_resource_mut::<Atom>()
            .unwrap()
            .push_test_atom(9, [0.9; 3], 0.1, 2.0);
        let saved = grass_io::save_app(&a);

        let b = app_with(&[]);
        grass_io::restore_app(&b, &saved).unwrap();
        let (x, y) = (
            a.get_resource_ref::<Atom>().unwrap(),
            b.get_resource_ref::<Atom>().unwrap(),
        );
        assert_eq!(y.nlocal(), 2);
        assert_eq!(y.natoms(), 2);
        assert_eq!(y.dt, x.dt);
        assert_eq!(y.tag[..2], x.tag[..2]);
        assert_eq!(y.pos[..2], x.pos[..2]);
        assert_eq!(y.vel[..2], x.vel[..2]);
        assert_eq!(y.pos.len(), 2, "ghosts are rebuilt, not restored");
    }
}
