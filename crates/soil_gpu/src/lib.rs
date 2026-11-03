//! GPU storage and particle kernels for SOIL, on the shared `grass_gpu` device.
//!
//! This is the particle substrate on the GPU. Anything a different particle method
//! would also need lives here; the physics lives in a tier above, such as `dirt_gpu`.
//!
//! - [`GpuState`]: atom columns resident on the device, and the velocity Verlet loop.
//! - [`GpuForce`]: the hook a physics tier implements to add its force kernel.
//! - [`CellList`]: neighbor build on the device.
//! - [`NeighborSlots`]: persistent per-neighbor state, such as contact springs or bonds.
//! - [`Boundary`]: planar boundary geometry.
//! - [`ResidentAtoms`]: a [`GpuState`] as an App resource, kept in step with the CPU
//!   [`Atom`](soil_core::Atom) so ordinary CPU systems keep working.
//!
//! Kernels are CubeCL `#[cube]` functions generic over the float type: `GpuReal` is `f32`
//! on the Mac and `f64` on CUDA, whatever the build's [`Real`](soil_core::Real) is. Values
//! are cast on the way up and down.

pub mod boundary;
pub use boundary::{Boundary, Plane};

pub mod cell_list;
pub use cell_list::{CellList, DeviceColumn, Grid};

pub mod gpu_state;
pub use gpu_state::{GpuForce, GpuState, StepColumns};

pub mod neighbor_slots;
pub use neighbor_slots::NeighborSlots;

pub mod resident;
pub use resident::{AtomColumnSync, ResidentAtoms, ResidentAtomsPlugin};

pub use grass_gpu::{GpuDevice, GpuReal};
