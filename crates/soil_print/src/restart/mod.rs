use std::{fs, fs::File};

use grass_scheduler::prelude::*;
use serde::Deserialize;
use soil_core::{Accum, Atom, AtomDataRegistry, CommResource, Input, Real, RunConfig, RunState};

use crate::{dump::dump_atoms_inner, DumpConfig, DumpRegistry};

pub(crate) mod serialization;
pub(crate) use serialization::{write_restart_inner, RestartData};

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
#[derive(Deserialize, Clone)]
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

fn restart_base_dir(input: &Input) -> String {
    match input.output_dir.as_deref() {
        Some(dir) => format!("{}/restart", dir),
        None => "restart".to_string(),
    }
}

/// Verify that an enabled restart input is readable and decodes before setup
/// begins.  This deliberately performs no mutation: `read_restart` remains the
/// sole owner of restoring atom state after the fallible application boundary
/// has accepted the input.
pub fn preflight_restart_input(
    restart_config: &RestartConfig,
    comm: &dyn soil_core::CommBackend,
    input: &Input,
) -> Result<(), String> {
    if !restart_config.read {
        return Ok(());
    }
    let base_dir = restart_base_dir(input);
    let ext = if restart_config.format == "json" {
        "json"
    } else {
        "bin"
    };
    let suffix = format!("_rank{}.{}", comm.rank(), ext);
    let latest = fs::read_dir(&base_dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            name.strip_prefix("restart_")
                .and_then(|tail| tail.strip_suffix(&suffix))
                .and_then(|step| step.parse::<usize>().ok())
        })
        .max();
    let Some(step) = latest else {
        return Ok(()); // preserve the historical "no checkpoint yet" behavior
    };
    let filename = format!("{base_dir}/restart_{step}_rank{}.{}", comm.rank(), ext);
    let file = File::open(&filename)
        .map_err(|error| format!("failed to open restart file '{filename}': {error}"))?;
    if ext == "json" {
        serde_json::from_reader::<_, RestartData>(std::io::BufReader::new(file))
            .map_err(|error| format!("failed to parse restart JSON '{filename}': {error}"))?;
    } else {
        bincode::deserialize_from::<_, RestartData>(std::io::BufReader::new(file)).map_err(
            |error| format!("failed to deserialize restart bincode '{filename}': {error}"),
        )?;
    }
    Ok(())
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
        panic!("restart file disappeared after startup validation '{filename}': {e}")
    });
    let data: RestartData = match ext {
        "json" => serde_json::from_reader(std::io::BufReader::new(file)).unwrap_or_else(|e| {
            panic!("restart JSON changed after startup validation '{filename}': {e}")
        }),
        _ => bincode::deserialize_from(std::io::BufReader::new(file)).unwrap_or_else(|e| {
            panic!("restart binary changed after startup validation '{filename}': {e}")
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
