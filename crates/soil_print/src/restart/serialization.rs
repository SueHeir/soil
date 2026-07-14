use std::{fs::File, io::BufWriter};

use serde::{Deserialize, Serialize};
use soil_core::{Atom, AtomDataRegistry};

use super::RestartConfig;

/// Serializable snapshot of all atom state for restart files.
///
/// Positions, velocities, and forces are stored as separate x/y/z vectors for
/// serialization compatibility. Legacy rotational fields (`omega_*`, `torque_*`,
/// etc.) are kept for backwards compatibility with old restart files but are no
/// longer written — rotational data now lives in `atom_data_buffers` via `AtomData`.
#[derive(Serialize, Deserialize)]
pub(crate) struct RestartData {
    pub(crate) natoms: u64,
    pub(crate) total_cycle: usize,
    pub(crate) dt: f64,
    pub(crate) tag: Vec<u32>,
    pub(crate) atom_type: Vec<u32>,
    pub(crate) pos_x: Vec<f64>,
    pub(crate) pos_y: Vec<f64>,
    pub(crate) pos_z: Vec<f64>,
    pub(crate) vel_x: Vec<f64>,
    pub(crate) vel_y: Vec<f64>,
    pub(crate) vel_z: Vec<f64>,
    pub(crate) force_x: Vec<f64>,
    pub(crate) force_y: Vec<f64>,
    pub(crate) force_z: Vec<f64>,
    pub(crate) mass: Vec<f64>,
    pub(crate) cutoff_radius: Vec<f64>,
    pub(crate) atom_data_buffers: Vec<Vec<f64>>,
    // Legacy fields for backwards compatibility with old restart files.
    #[serde(default)]
    pub(crate) omega_x: Vec<f64>,
    #[serde(default)]
    pub(crate) omega_y: Vec<f64>,
    #[serde(default)]
    pub(crate) omega_z: Vec<f64>,
    #[serde(default)]
    pub(crate) torque_x: Vec<f64>,
    #[serde(default)]
    pub(crate) torque_y: Vec<f64>,
    #[serde(default)]
    pub(crate) torque_z: Vec<f64>,
    #[serde(default)]
    pub(crate) ang_mom_x: Vec<f64>,
    #[serde(default)]
    pub(crate) ang_mom_y: Vec<f64>,
    #[serde(default)]
    pub(crate) ang_mom_z: Vec<f64>,
    #[serde(default)]
    pub(crate) quaternion: Vec<[f64; 4]>,
}

impl RestartData {
    /// Build a `RestartData` snapshot from the current atom state.
    ///
    /// Only local atoms (indices `0..nlocal`) are included; ghost atoms are excluded.
    pub(crate) fn from_atoms(atoms: &Atom, registry: &AtomDataRegistry, step: usize) -> Self {
        let nlocal = atoms.nlocal() as usize;
        RestartData {
            natoms: atoms.natoms(),
            total_cycle: step,
            dt: atoms.dt,
            tag: atoms.tag[..nlocal].to_vec(),
            atom_type: atoms.atom_type[..nlocal].to_vec(),
            // Restart files keep an f64 wire format regardless of build precision,
            // so an f32 run can round-trip / restart from an f64 checkpoint losslessly.
            pos_x: atoms.pos[..nlocal].iter().map(|p| p[0] as f64).collect(),
            pos_y: atoms.pos[..nlocal].iter().map(|p| p[1] as f64).collect(),
            pos_z: atoms.pos[..nlocal].iter().map(|p| p[2] as f64).collect(),
            vel_x: atoms.vel[..nlocal].iter().map(|v| v[0] as f64).collect(),
            vel_y: atoms.vel[..nlocal].iter().map(|v| v[1] as f64).collect(),
            vel_z: atoms.vel[..nlocal].iter().map(|v| v[2] as f64).collect(),
            force_x: atoms.force[..nlocal].iter().map(|v| v[0] as f64).collect(),
            force_y: atoms.force[..nlocal].iter().map(|v| v[1] as f64).collect(),
            force_z: atoms.force[..nlocal].iter().map(|v| v[2] as f64).collect(),
            mass: atoms.mass[..nlocal].iter().map(|&m| m as f64).collect(),
            cutoff_radius: atoms.cutoff_radius[..nlocal]
                .iter()
                .map(|&r| r as f64)
                .collect(),
            atom_data_buffers: registry.pack_all_for_restart(nlocal),
            // Legacy fields left empty — rotational data now in atom_data_buffers via DemAtom.
            omega_x: Vec::new(),
            omega_y: Vec::new(),
            omega_z: Vec::new(),
            torque_x: Vec::new(),
            torque_y: Vec::new(),
            torque_z: Vec::new(),
            ang_mom_x: Vec::new(),
            ang_mom_y: Vec::new(),
            ang_mom_z: Vec::new(),
            quaternion: Vec::new(),
        }
    }
}

/// Serialize restart data to disk in the configured format (bincode or JSON).
pub(crate) fn write_restart_inner(
    data: &RestartData,
    base_dir: &str,
    step: usize,
    rank: i32,
    restart_config: &RestartConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    match restart_config.format.as_str() {
        "json" => {
            let filename = format!("{}/restart_{}_rank{}.json", base_dir, step, rank);
            let file = File::create(&filename)?;
            serde_json::to_writer(BufWriter::new(file), data)?;
        }
        _ => {
            let filename = format!("{}/restart_{}_rank{}.bin", base_dir, step, rank);
            let file = File::create(&filename)?;
            bincode::serialize_into(BufWriter::new(file), data)?;
        }
    }
    Ok(())
}
