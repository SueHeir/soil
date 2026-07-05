use std::{fs, fs::File, io::Write};

use grass_scheduler::prelude::*;
use serde::{Deserialize, Serialize};
use soil_core::{Atom, AtomDataRegistry, CommResource, Input, RunConfig, RunState};

use crate::DumpRegistry;

// ── VTP config ──────────────────────────────────────────────────────────────

/// TOML `[vtp]` — ParaView `.vtp` visualization output settings.
///
/// # TOML Fields
///
/// | Field      | Type    | Default | Description                              |
/// |------------|---------|---------|------------------------------------------|
/// | `interval` | `usize` | `0`    | Write VTP every N steps (0 = disabled)   |
///
/// VTP files are written to `{output_dir}/vtp/{step}CYCLE_{rank}RANK.vtp` and
/// include particle positions, radii, velocity magnitudes, ghost flags, and any
/// fields registered via [`DumpRegistry`].
#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct VtpConfig {
    /// Write VTP every N steps (0 = disabled).
    #[serde(default)]
    pub interval: usize,
}

// ── VTP output ──────────────────────────────────────────────────────────────

/// Write a single VTP `<DataArray>` element with per-point scalar data.
fn write_vtp_data_array(
    file: &mut File,
    vtp_type: &str,
    name: &str,
    n: usize,
    value_fn: impl Fn(usize) -> String,
) -> std::io::Result<()> {
    writeln!(
        file,
        "<DataArray type=\"{}\" Name=\"{}\" format=\"ascii\">",
        vtp_type, name
    )?;
    for i in 0..n {
        writeln!(file, "{}", value_fn(i))?;
    }
    write!(file, "</DataArray>")?;
    Ok(())
}

/// Write ParaView VTP output at the configured interval.
///
/// Each MPI rank writes its own `.vtp` file containing local + ghost atoms.
/// Output includes positions, radii, velocity magnitude, ghost flags, and any
/// fields registered via [`DumpRegistry`].
#[allow(clippy::too_many_arguments)]
pub fn print_vtp(
    atoms: Res<Atom>,
    registry: Res<AtomDataRegistry>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    input: Res<Input>,
    vtp_config: Res<VtpConfig>,
    run_config: Res<RunConfig>,
    scheduler_manager: Res<SchedulerManager>,
    dump_registry: Res<DumpRegistry>,
) {
    let count = run_state.total_cycle;
    let rank = comm.rank();
    let stage = run_config.current_stage(scheduler_manager.index);
    let interval = stage
        .overrides
        .get("vtp_interval")
        .and_then(|v| v.as_integer())
        .map(|i| i as usize)
        .unwrap_or(vtp_config.interval);
    if interval == 0 || !count.is_multiple_of(interval) {
        return;
    }
    if let Err(e) = print_vtp_inner(&atoms, &registry, count, rank, &input, &dump_registry) {
        eprintln!("WARNING: VTP write failed at step {}: {}", count, e);
    }
}

/// Inner VTP write logic, separated for error handling via `?`.
fn print_vtp_inner(
    atoms: &Atom,
    registry: &AtomDataRegistry,
    count: usize,
    rank: i32,
    input: &Input,
    dump_reg: &DumpRegistry,
) -> std::io::Result<()> {
    let base_dir = match input.output_dir.as_deref() {
        Some(dir) => format!("{}/vtp", dir),
        None => "vtp".to_string(),
    };
    let filename = format!("{}/{}CYCLE_{}RANK.vtp", base_dir, count, rank);
    fs::create_dir_all(&base_dir)?;
    let mut file = File::create(&filename)?;

    let n = atoms.len();
    let nlocal = atoms.nlocal as usize;

    // XML header and PolyData opening
    write!(&mut file, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n<PolyData>\n")?;
    writeln!(&mut file, "<Piece NumberOfPoints=\"{}\">", n)?;

    // Points (positions)
    write!(
        &mut file,
        "<Points><DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">"
    )?;
    for i in 0..n {
        writeln!(
            &mut file,
            "{} {} {}",
            atoms.pos[i][0], atoms.pos[i][1], atoms.pos[i][2]
        )?;
    }
    write!(&mut file, "</DataArray>\n</Points>\n")?;

    // Per-point data arrays
    writeln!(&mut file, "<PointData Scalars=\"\" Vectors=\"\">")?;
    write_vtp_data_array(&mut file, "Float32", "Radius", n, |i| {
        format!("{}", atoms.cutoff_radius[i])
    })?;
    write_vtp_data_array(&mut file, "Float32", "Vel_Mag", n, |i| {
        let vmag =
            (atoms.vel[i][0].powi(2) + atoms.vel[i][1].powi(2) + atoms.vel[i][2].powi(2)).sqrt();
        format!("{vmag}")
    })?;
    write_vtp_data_array(&mut file, "Int32", "IsGhost", n, |i| {
        if i >= nlocal {
            "1".to_string()
        } else {
            "0".to_string()
        }
    })?;

    // Registered scalar callbacks
    for (name, f) in dump_reg.scalar_fns.borrow().iter() {
        let data = f(atoms, registry);
        write_vtp_data_array(&mut file, "Float32", name, n, |i| {
            if i < data.len() {
                format!("{}", data[i])
            } else {
                "0".to_string()
            }
        })?;
    }

    // Registered vector callbacks
    for (name, f) in dump_reg.vector_fns.borrow().iter() {
        let data = f(atoms, registry);
        writeln!(
            &mut file,
            "<DataArray type=\"Float32\" Name=\"{}\" NumberOfComponents=\"3\" format=\"ascii\">",
            name
        )?;
        for i in 0..n {
            if i < data.len() {
                writeln!(&mut file, "{} {} {}", data[i][0], data[i][1], data[i][2])?;
            } else {
                writeln!(&mut file, "0 0 0")?;
            }
        }
        write!(&mut file, "</DataArray>")?;
    }

    write!(
        &mut file,
        "</PointData>\n</Piece>\n</PolyData>\n</VTKFile>\n"
    )?;
    Ok(())
}
