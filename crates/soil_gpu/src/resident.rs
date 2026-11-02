//! [`ResidentAtoms`]: atoms resident on the GPU, in step with the CPU [`Atom`].
//!
//! A physics tier owns the kernels. This module owns the bookkeeping that lets ordinary
//! CPU systems keep working beside them, through `grass_scheduler`'s coherence hooks:
//!
//! - When a CPU system reads [`Atom`] while the GPU copy is newer, the scheduler calls
//!   the bridge here, which downloads positions, velocities, and any extra columns the
//!   tier registered.
//! - When a CPU system writes [`Atom`], the next call to
//!   [`ResidentAtoms::begin_kernels`] uploads it again and reports that the resident
//!   force must be evaluated afresh.

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;

use grass_app::prelude::*;
use grass_gpu::GpuPlugin;
use grass_scheduler::{CoherenceRegistry, MirrorBridge};
use soil_core::{Atom, AtomDataRegistry, Real};

use crate::gpu_state::GpuState;
use grass_gpu::GpuReal;

/// Moves one auxiliary column of a [`GpuState`] between the GPU and a CPU extension,
/// such as angular velocity into a tier's `AtomData`.
pub struct AtomColumnSync {
    /// Index from [`GpuState::add_aux_dof`].
    pub aux: usize,
    /// Copies the downloaded column into the CPU extension.
    pub to_host: fn(&AtomDataRegistry, &[GpuReal]),
    /// Reads the CPU extension for upload.
    pub from_host: fn(&AtomDataRegistry, usize) -> Vec<GpuReal>,
}

/// A [`GpuState`] as an App resource, with the host-side bookkeeping for it.
#[derive(Default)]
pub struct ResidentAtoms {
    /// The resident state, once a tier has built it.
    pub state: Option<GpuState>,
    /// Extra columns to keep in step with the CPU.
    pub columns: Vec<AtomColumnSync>,
}

/// `[Real; 3]` to the flat xyz floats kernels use, for the first `n` rows of a column.
pub fn to_gpu(column: &[[Real; 3]], n: usize) -> Vec<GpuReal> {
    column[..n]
        .iter()
        .flat_map(|v| [v[0] as GpuReal, v[1] as GpuReal, v[2] as GpuReal])
        .collect()
}

impl ResidentAtoms {
    /// Call before launching kernels, from a system that takes
    /// `ResMut<CoherenceRegistry>`. Uploads the CPU copy if a CPU system wrote it, and
    /// returns `true` in that case: the resident force is no longer valid and the next
    /// window must start with [`GpuState::run_steps`].
    pub fn begin_kernels(
        &mut self,
        coherence: &mut CoherenceRegistry,
        atoms: &Atom,
        registry: &AtomDataRegistry,
    ) -> bool {
        if !coherence.take_host_dirty(TypeId::of::<Atom>()) {
            return false;
        }
        let Some(state) = self.state.as_mut() else {
            return false;
        };
        let n = state.n();
        state.write_pos(&to_gpu(&atoms.pos, n));
        state.write_vel(&to_gpu(&atoms.vel, n));
        for column in &self.columns {
            state.set_aux_state(column.aux, &(column.from_host)(registry, n));
        }
        true
    }

    /// Call after launching kernels. CPU systems that read [`Atom`] now trigger a download.
    pub fn end_kernels(coherence: &mut CoherenceRegistry) {
        coherence.mark_device_dirty(TypeId::of::<Atom>());
    }
}

/// Downloads resident atoms into the CPU [`Atom`] when the scheduler asks.
struct AtomBridge {
    atom: usize,
    registry: usize,
    resident: usize,
}

impl MirrorBridge for AtomBridge {
    fn resolve(&mut self, index: &HashMap<TypeId, usize>) {
        self.atom = index[&TypeId::of::<Atom>()];
        self.registry = index[&TypeId::of::<AtomDataRegistry>()];
        self.resident = index[&TypeId::of::<ResidentAtoms>()];
    }

    fn download(&self, resources: &[RefCell<Box<dyn Any>>]) {
        let mut resident = resources[self.resident].borrow_mut();
        let resident = resident
            .downcast_mut::<ResidentAtoms>()
            .expect("soil_gpu: resident index is not ResidentAtoms");
        let ResidentAtoms { state, columns } = resident;
        let Some(state) = state.as_mut() else { return };
        {
            let mut atoms = resources[self.atom].borrow_mut();
            let atoms = atoms
                .downcast_mut::<Atom>()
                .expect("soil_gpu: atom index is not Atom");
            state.sync_state_to_host();
            for (host, gpu) in atoms.pos.iter_mut().zip(state.pos().chunks_exact(3)) {
                *host = [gpu[0] as Real, gpu[1] as Real, gpu[2] as Real];
            }
            for (host, gpu) in atoms.vel.iter_mut().zip(state.vel().chunks_exact(3)) {
                *host = [gpu[0] as Real, gpu[1] as Real, gpu[2] as Real];
            }
        }
        let registry = resources[self.registry].borrow();
        let registry = registry
            .downcast_ref::<AtomDataRegistry>()
            .expect("soil_gpu: registry index is not AtomDataRegistry");
        for column in columns.iter() {
            (column.to_host)(registry, state.aux_state(column.aux));
        }
    }
}

/// Installs the shared GPU device, the [`ResidentAtoms`] resource, and the bridge that
/// keeps CPU systems reading [`Atom`] correct. Requires `AtomPlugin`.
pub struct ResidentAtomsPlugin;

impl Plugin for ResidentAtomsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(GpuPlugin::default());
        app.add_resource(ResidentAtoms::default());
        let cell = app
            .get_mut_resource(TypeId::of::<CoherenceRegistry>())
            .expect("GpuPlugin installs the CoherenceRegistry");
        cell.borrow_mut()
            .downcast_mut::<CoherenceRegistry>()
            .expect("CoherenceRegistry resource has the wrong type")
            .register(
                TypeId::of::<Atom>(),
                Box::new(AtomBridge {
                    atom: usize::MAX,
                    registry: usize::MAX,
                    resident: usize::MAX,
                }),
            );
    }
}
