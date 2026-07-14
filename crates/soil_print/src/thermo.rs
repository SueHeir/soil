use std::{collections::HashMap, time::Instant};

use grass_scheduler::prelude::*;
use serde::Deserialize;
use soil_core::{
    compute_ke, Atom, CommResource, GroupRegistry, Neighbor, RunConfig, RunState, VirialStress,
};

// ── Thermo config ───────────────────────────────────────────────────────────

/// TOML `[thermo]` — configures which columns appear in thermo console output.
///
/// # TOML Fields
///
/// | Field     | Type             | Default                                             | Description                        |
/// |-----------|------------------|------------------------------------------------------|------------------------------------|
/// | `columns` | `[String]` (opt) | `["step","atoms","ke","neighbors","walltime","stepps"]` | Column names to display         |
///
/// Column names can use `"compute/group"` syntax (e.g. `"ke/mobile"`) to filter
/// by a named atom group. Built-in compute names: `step`, `atoms`, `ke`, `temp`,
/// `neighbors`, `walltime`, `stepps`. Any value pushed via [`Thermo::set`] is also
/// available as a column.
#[derive(Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct ThermoConfig {
    /// Column names to display. If `None`, uses the default set.
    #[serde(default)]
    pub columns: Option<Vec<String>>,
}

// ── Thermo column ───────────────────────────────────────────────────────────

/// A parsed thermo column specification, produced from a raw string like `"ke/mobile"`.
pub struct ThermoColumn {
    /// The original column string from config (e.g. `"ke/mobile"`).
    pub raw: String,
    /// The compute name portion (e.g. `"ke"`).
    pub compute_name: String,
    /// Optional group name filter (e.g. `Some("mobile")`).
    pub group_name: Option<String>,
    /// Formatted header string for console display (e.g. `"Ke/mobile"`).
    pub header: String,
    /// Column display width in characters (minimum 12).
    pub width: usize,
}

/// Parse a raw column spec string (e.g. `"ke/mobile"`) into a [`ThermoColumn`].
pub(crate) fn parse_thermo_column(raw: &str) -> ThermoColumn {
    let parts: Vec<&str> = raw.splitn(2, '/').collect();
    let compute_name = parts[0].to_string();
    let group_name = parts.get(1).map(|s| s.to_string());

    let header = if let Some(ref g) = group_name {
        format!("{}/{}", capitalize(&compute_name), g)
    } else {
        capitalize(&compute_name)
    };

    let width = header.len().max(12);

    ThermoColumn {
        raw: raw.to_string(),
        compute_name,
        group_name,
        header,
        width,
    }
}

/// Capitalize the first character of a string.
pub(crate) fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// Returns the default thermo column names when no `[thermo] columns` is specified.
pub(crate) fn default_columns() -> Vec<String> {
    vec![
        "step".into(),
        "atoms".into(),
        "ke".into(),
        "neighbors".into(),
        "walltime".into(),
        "stepps".into(),
    ]
}

// ── Thermo ──────────────────────────────────────────────────────────────────

/// Runtime state for thermo console output.
///
/// Tracks the print interval, wall-clock timing for steps-per-second calculation,
/// parsed column specifications, and user-pushed values from other plugins.
///
/// Other plugins can push custom values via [`Thermo::set`], which become available
/// as thermo columns if listed in the `[thermo] columns` config.
pub struct Thermo {
    /// Print thermo output every N steps.
    pub interval: usize,
    /// Wall-clock timestamp of the last thermo print (for steps/sec calculation).
    pub start_time: Instant,
    /// The step number at which thermo was last printed.
    pub last_printed_step: usize,
    /// Parsed column specifications for output formatting.
    pub columns: Vec<ThermoColumn>,
    /// User-pushed named values (e.g. `"pe"` → `42.0`), available as thermo columns.
    pub values: HashMap<String, f64>,
}

impl Default for Thermo {
    fn default() -> Self {
        Self::new()
    }
}

impl Thermo {
    /// Create a new `Thermo` with default interval of 100 steps.
    pub fn new() -> Self {
        Thermo {
            interval: 100,
            start_time: Instant::now(),
            last_printed_step: 0,
            columns: Vec::new(),
            values: HashMap::new(),
        }
    }

    /// Push a named value into the thermo value map.
    ///
    /// The value becomes available as a thermo column if its name is listed in
    /// `[thermo] columns`. Values are overwritten on each call, so plugins should
    /// call this every thermo interval to keep values current.
    pub fn set(&mut self, name: &str, value: f64) {
        self.values.insert(name.to_string(), value);
    }
}

// ── Dump config ─────────────────────────────────────────────────────────────

// ── Thermo systems ──────────────────────────────────────────────────────────

/// Setup system for thermo output: parses column specs and prints the header.
///
/// Runs at the start of each stage to update the print interval and reset the
/// wall-clock timer. Column specifications are parsed only once (on the first stage).
pub fn setup_thermo(
    config: Res<RunConfig>,
    thermo_config: Res<ThermoConfig>,
    scheduler_manager: Res<SchedulerManager>,
    comm: Res<CommResource>,
    run_state: Res<RunState>,
    groups: Option<Res<GroupRegistry>>,
    mut thermo: ResMut<Thermo>,
    mut virial: Option<ResMut<VirialStress>>,
) {
    let index = scheduler_manager.index;
    if index >= config.num_stages() {
        return;
    }
    thermo.interval = config
        .current_stage(index)
        .overrides
        .get("thermo")
        .and_then(|v| v.as_integer())
        .map(|i| i as usize)
        .unwrap_or(100);
    if let Some(ref mut v) = virial {
        v.set_interval(thermo.interval);
    }
    thermo.start_time = Instant::now();
    thermo.last_printed_step = run_state.total_cycle;

    // Parse column specifications (only on first stage, or if columns empty).
    if thermo.columns.is_empty() {
        let col_names = thermo_config
            .columns
            .clone()
            .unwrap_or_else(default_columns);
        thermo.columns = col_names.iter().map(|s| parse_thermo_column(s)).collect();
    }

    for col in thermo.columns.iter() {
        if let Some(ref gname) = col.group_name {
            match groups.as_ref() {
                Some(groups) => groups
                    .validate_name(gname, &format!("thermo column '{}'", col.raw))
                    .expect("thermo group references are validated by PrintPlugin::try_build"),
                None => {
                    panic!(
                        "thermo group plugin was not installed before validated thermo setup: {}",
                        col.raw
                    );
                }
            }
        }
    }

    if comm.rank() == 0 {
        println!();
        let header: String = thermo
            .columns
            .iter()
            .map(|c| format!("{:<width$}", c.header, width = c.width))
            .collect::<Vec<_>>()
            .join(" ");
        let total_width: usize = thermo.columns.iter().map(|c| c.width).sum::<usize>()
            + thermo.columns.len().saturating_sub(1);
        println!("{}", header);
        println!("{}", "-".repeat(total_width));
    }
}

/// Print thermo output to console at the configured interval.
///
/// All MPI ranks participate in allreduce operations (for KE, atom counts, etc.),
/// but only rank 0 prints the formatted output line.
#[allow(clippy::too_many_arguments)]
pub fn print_thermo(
    mut atoms: ResMut<Atom>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    neighbor: Res<Neighbor>,
    groups: Option<Res<GroupRegistry>>,
    mut thermo: ResMut<Thermo>,
) {
    let step = run_state.total_cycle;
    if !step.is_multiple_of(thermo.interval) {
        return;
    }

    // Pre-compute values that need allreduce (all ranks must participate).
    // We compute these eagerly so all ranks call allreduce together.
    let local_ke_all = compute_ke(&atoms, None);
    let global_ke_all = comm.all_reduce_sum_f64(local_ke_all);
    // Refresh the global atom count here (reporting-only; the ghost-rebuild path no
    // longer maintains it). Piggybacks on this thermo collective — no new barrier.
    let global_atoms_all = comm.all_reduce_sum_f64(atoms.nlocal() as f64);
    atoms.set_global_natoms(global_atoms_all as u64);
    let local_neighbors = neighbor.neighbor_indices.len() as f64;
    let global_neighbors = comm.all_reduce_sum_f64(local_neighbors);

    // Pre-compute group-filtered KE and atom counts for any group columns.
    // Each group that appears needs its own allreduce.
    let mut group_ke: HashMap<String, f64> = HashMap::new();
    let mut group_count: HashMap<String, f64> = HashMap::new();
    for col in thermo.columns.iter() {
        if let Some(ref gname) = col.group_name {
            if group_ke.contains_key(gname) {
                continue;
            }
            if let Some(group) = groups.as_ref().and_then(|groups| groups.get(gname)) {
                let local_ke = compute_ke(&atoms, Some(&group.mask));
                let local_count = group.count as f64;
                group_ke.insert(gname.clone(), comm.all_reduce_sum_f64(local_ke));
                group_count.insert(gname.clone(), comm.all_reduce_sum_f64(local_count));
            }
        }
    }

    if comm.rank() == 0 {
        let elapsed = thermo.start_time.elapsed().as_secs_f64();
        let steps_since = (step - thermo.last_printed_step) as f64;
        let steps_per_sec = if elapsed > 1e-9 {
            steps_since / elapsed
        } else {
            0.0
        };

        let mut parts: Vec<String> = Vec::new();
        for col in thermo.columns.iter() {
            let val_str = match col.compute_name.as_str() {
                "step" => format!("{:<width$}", step, width = col.width),
                "atoms" => {
                    if let Some(ref gname) = col.group_name {
                        let n = group_count.get(gname).copied().unwrap_or(0.0) as u64;
                        format!("{:<width$}", n, width = col.width)
                    } else {
                        format!("{:<width$}", atoms.natoms(), width = col.width)
                    }
                }
                "ke" => {
                    let ke = if let Some(ref gname) = col.group_name {
                        group_ke.get(gname).copied().unwrap_or(0.0)
                    } else {
                        global_ke_all
                    };
                    format!("{:<width$.6e}", ke, width = col.width)
                }
                "temp" => {
                    let (ke, n) = if let Some(ref gname) = col.group_name {
                        (
                            group_ke.get(gname).copied().unwrap_or(0.0),
                            group_count.get(gname).copied().unwrap_or(0.0),
                        )
                    } else {
                        (global_ke_all, global_atoms_all)
                    };
                    let ndof = 3.0 * n - 3.0;
                    let temp = if ndof > 0.0 { 2.0 * ke / ndof } else { 0.0 };
                    format!("{:<width$.6}", temp, width = col.width)
                }
                "neighbors" => {
                    format!("{:<width$}", global_neighbors as usize, width = col.width)
                }
                "walltime" => {
                    format!("{:<width$.4}", elapsed, width = col.width)
                }
                "stepps" => {
                    format!("{:<width$.1}", steps_per_sec, width = col.width)
                }
                other => {
                    // User-pushed value from Thermo::set()
                    if let Some(&v) = thermo.values.get(other) {
                        format!("{:<width$.6e}", v, width = col.width)
                    } else {
                        format!("{:<width$}", "N/A", width = col.width)
                    }
                }
            };
            parts.push(val_str);
        }
        println!("{}", parts.join(" "));
        thermo.start_time = Instant::now();
        thermo.last_printed_step = step;
    }
}

// ── Virial stress output ────────────────────────────────────────────────────

/// MPI-reduce each virial stress component and push to thermo values.
///
/// Publishes `virial_xx`, `virial_yy`, `virial_zz`, `virial_xy`, `virial_xz`,
/// `virial_yz` as thermo columns. Only runs on thermo output steps.
pub fn output_virial_to_thermo(
    virial: Option<Res<VirialStress>>,
    run_state: Res<RunState>,
    comm: Res<CommResource>,
    mut thermo: ResMut<Thermo>,
) {
    if !run_state.total_cycle.is_multiple_of(thermo.interval) {
        return;
    }
    let virial = match virial {
        Some(v) => v,
        None => return,
    };
    let xx = comm.all_reduce_sum_f64(virial.xx);
    let yy = comm.all_reduce_sum_f64(virial.yy);
    let zz = comm.all_reduce_sum_f64(virial.zz);
    let xy = comm.all_reduce_sum_f64(virial.xy);
    let xz = comm.all_reduce_sum_f64(virial.xz);
    let yz = comm.all_reduce_sum_f64(virial.yz);
    thermo.set("virial_xx", xx);
    thermo.set("virial_yy", yy);
    thermo.set("virial_zz", zz);
    thermo.set("virial_xy", xy);
    thermo.set("virial_xz", xz);
    thermo.set("virial_yz", yz);
}
