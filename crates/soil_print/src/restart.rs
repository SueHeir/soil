use std::{fs, fs::File, io::BufWriter};

use grass_scheduler::prelude::*;
use serde::{Deserialize, Serialize};
use soil_core::{Accum, Atom, AtomDataRegistry, CommResource, Input, Real, RunConfig, RunState};

use crate::{dump::dump_atoms_inner, DumpConfig, DumpRegistry};

// ── Restart config ──────────────────────────────────────────────────────────

/// Default restart format: bincode.
fn default_restart_format() -> String {
    "bincode".to_string()
}

/// TOML `[restart]` — restart (checkpoint) file write/read settings.
///
/// # TOML Fields
///
/// | Field      | Type    | Default      | Description                                  |
/// |------------|---------|--------------|----------------------------------------------|
/// | `interval` | `usize` | `0`         | Write restart every N steps (0 = disabled)   |
/// | `format`   | `String`| `"bincode"` | `"bincode"` (compact) or `"json"` (readable) |
/// | `read`     | `bool`  | `false`     | Read latest restart file at startup          |
///
/// When `read = true`, the system scans the restart directory for the highest-numbered
/// restart file matching this rank and format, then restores atom state from it.
#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct RestartConfig {
    /// Write restart every N steps (0 = disabled).
    #[serde(default)]
    pub interval: usize,
    /// File format: `"bincode"` (compact binary) or `"json"` (human-readable).
    #[serde(default = "default_restart_format")]
    pub format: String,
    /// Whether to read the latest restart file at startup.
    #[serde(default)]
    pub read: bool,
}

impl Default for RestartConfig {
    fn default() -> Self {
        RestartConfig {
            interval: 0,
            format: "bincode".to_string(),
            read: false,
        }
    }
}

// ── RestartData ─────────────────────────────────────────────────────────────

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
    // Legacy fields for backwards compatibility with old restart files
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
        let nlocal = atoms.nlocal as usize;
        RestartData {
            natoms: atoms.natoms,
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
            // Legacy fields left empty — rotational data now in atom_data_buffers via DemAtom
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

fn restart_base_dir(input: &Input) -> String {
    match input.output_dir.as_deref() {
        Some(dir) => format!("{}/restart", dir),
        None => "restart".to_string(),
    }
}

// ── Restart write ───────────────────────────────────────────────────────────

/// Write restart (checkpoint) files at the configured interval.
///
/// Each MPI rank writes its own restart file containing only its local atoms.
/// The file format is determined by `[restart] format` (bincode or JSON).
#[allow(clippy::too_many_arguments)]
pub fn write_restart(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    restart_config: Res<RestartConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
) {
    let stage = run_config.current_stage(scheduler_manager.index);
    let interval = stage
        .overrides
        .get("restart_interval")
        .and_then(|v| v.as_integer())
        .map(|i| i as usize)
        .unwrap_or(restart_config.interval);
    if interval == 0 {
        return;
    }
    let step = run_state.total_cycle;
    if !step.is_multiple_of(interval) {
        return;
    }

    let rank = comm.rank();
    let base_dir = restart_base_dir(&input);
    fs::create_dir_all(&base_dir).ok();

    let data = RestartData::from_atoms(&atoms, &registry, step);

    if let Err(e) = write_restart_inner(&data, &base_dir, step, rank, &restart_config) {
        eprintln!("WARNING: Restart write failed at step {}: {}", step, e);
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

// ── Stage-end save ──────────────────────────────────────────────────────────

/// Write dump + restart files when a stage with `save_at_end = true` finishes.
///
/// Runs before `update_cycle` so the stage index is still valid. This ensures
/// that the final state of each stage is captured even if the regular dump/restart
/// intervals don't align with the stage boundary.
#[allow(clippy::too_many_arguments)]
pub fn check_stage_end_save(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    dump_config: Res<DumpConfig>,
    restart_config: Res<RestartConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    dump_registry: Res<DumpRegistry>,
) {
    let index = scheduler_manager.index;
    if index >= run_config.num_stages() {
        return;
    }
    let stage = run_config.current_stage(index);
    if !stage.save_at_end {
        return;
    }

    let remaining = run_state.cycle_remaining[index];
    // Don't save for skipped stages (remaining == 0)
    if remaining == 0 {
        return;
    }

    // check_stage_end_save runs .before("update_cycle"), so the cycle counter
    // hasn't been incremented for the current step yet.  After update_cycle runs,
    // count will be count+1, which equals remaining on the final step.
    let count = run_state.cycle_count[index];
    let is_last_step = count + 1 == remaining;
    let is_advancing = scheduler_manager.advance_requested;

    if !is_last_step && !is_advancing {
        return;
    }

    let step = run_state.total_cycle;
    let rank = comm.rank();
    let stage_label = stage.name.as_deref().unwrap_or("(unnamed)");

    if rank == 0 {
        println!(
            "Stage {} [{}] finished — saving dump + restart at step {}",
            index, stage_label, step
        );
    }

    // Write dump
    if let Err(e) = dump_atoms_inner(
        &atoms,
        &registry,
        &comm,
        step,
        &input,
        &dump_config,
        &dump_registry,
    ) {
        eprintln!(
            "WARNING: Stage-end dump write failed at step {}: {}",
            step, e
        );
    }

    // Write restart
    let base_dir = restart_base_dir(&input);
    fs::create_dir_all(&base_dir).ok();

    let data = RestartData::from_atoms(&atoms, &registry, step);

    if let Err(e) = write_restart_inner(&data, &base_dir, step, rank, &restart_config) {
        eprintln!(
            "WARNING: Stage-end restart write failed at step {}: {}",
            step, e
        );
    }
}

// ── Restart read ────────────────────────────────────────────────────────────

/// Read the latest restart file at startup and restore atom state.
///
/// Scans the restart directory for files matching the current rank and format,
/// selects the one with the highest step number, and deserializes it to restore
/// all atom data (positions, velocities, forces, mass, radius, and any `AtomData`
/// extensions stored in `atom_data_buffers`).
///
/// Only runs when `[restart] read = true` and only on the first stage.
pub fn read_restart(
    restart_config: Res<RestartConfig>,
    comm: Res<CommResource>,
    input: Res<Input>,
    mut atoms: ResMut<Atom>,
    registry: Res<AtomDataRegistry>,
    mut run_state: ResMut<RunState>,
) {
    if !restart_config.read {
        return;
    }

    let rank = comm.rank();
    let base_dir = restart_base_dir(&input);

    // Find the latest restart file for this rank
    let ext = match restart_config.format.as_str() {
        "json" => "json",
        _ => "bin",
    };

    let prefix = "restart_";
    let suffix = format!("_rank{}.{}", rank, ext);

    let mut latest_step: Option<usize> = None;
    if let Ok(entries) = fs::read_dir(&base_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(prefix) && name.ends_with(&suffix) {
                let mid = &name[prefix.len()..name.len() - suffix.len()];
                if let Ok(step) = mid.parse::<usize>() {
                    if latest_step.map_or(true, |prev| step > prev) {
                        latest_step = Some(step);
                    }
                }
            }
        }
    }

    let step = match latest_step {
        Some(s) => s,
        None => {
            if rank == 0 {
                println!("Restart: no restart files found in {}", base_dir);
            }
            return;
        }
    };

    let filename = format!("{}/restart_{}_rank{}.{}", base_dir, step, rank, ext);
    if rank == 0 {
        println!("Restart: reading from {}", filename);
    }

    let file = File::open(&filename).unwrap_or_else(|e| {
        eprintln!("ERROR: Failed to open restart file '{}': {}", filename, e);
        std::process::exit(1);
    });
    let data: RestartData = match ext {
        "json" => serde_json::from_reader(std::io::BufReader::new(file)).unwrap_or_else(|e| {
            eprintln!("ERROR: Failed to parse restart JSON '{}': {}", filename, e);
            std::process::exit(1);
        }),
        _ => bincode::deserialize_from(std::io::BufReader::new(file)).unwrap_or_else(|e| {
            eprintln!(
                "ERROR: Failed to deserialize restart bincode '{}': {}",
                filename, e
            );
            std::process::exit(1);
        }),
    };

    let n = data.tag.len();

    // Clear existing atoms and repopulate from restart data
    atoms.natoms = data.natoms;
    atoms.nlocal = n as u32;
    atoms.nghost = 0;
    atoms.dt = data.dt;

    atoms.tag = data.tag;
    atoms.atom_type = data.atom_type;
    atoms.origin_index = vec![0; n];
    atoms.is_ghost = vec![false; n];
    // Restart wire format is f64; convert back to the build's storage precision.
    atoms.pos = data
        .pos_x
        .iter()
        .zip(data.pos_y.iter())
        .zip(data.pos_z.iter())
        .map(|((&x, &y), &z)| [x as Real, y as Real, z as Real])
        .collect();
    atoms.vel = data
        .vel_x
        .iter()
        .zip(data.vel_y.iter())
        .zip(data.vel_z.iter())
        .map(|((&x, &y), &z)| [x as Real, y as Real, z as Real])
        .collect();
    atoms.force = data
        .force_x
        .iter()
        .zip(data.force_y.iter())
        .zip(data.force_z.iter())
        .map(|((&x, &y), &z)| [x as Accum, y as Accum, z as Accum])
        .collect();
    atoms.mass = data.mass.iter().map(|&m| m as Real).collect();
    atoms.inv_mass = atoms.mass.iter().map(|&m| 1.0 / m).collect();
    atoms.cutoff_radius = data.cutoff_radius.iter().map(|&r| r as Real).collect();

    // Restore AtomData (DemAtom, etc.) from generic buffers
    if !data.atom_data_buffers.is_empty() {
        registry.unpack_all_from_restart(&data.atom_data_buffers);
    }

    run_state.total_cycle = data.total_cycle;
    if rank == 0 {
        println!("Restart: loaded {} atoms from step {}", n, data.total_cycle);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────
