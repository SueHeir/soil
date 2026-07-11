//! Box deformation fix for SOIL.
//!
//! Continuously modifies simulation box boundaries during a run, analogous to
//! LAMMPS `fix deform`. Supports per-axis independent control with three styles,
//! plus Lees–Edwards `xy` simple shear:
//!
//! - **erate** — engineering strain rate: `L(t) = L0 * (1 + rate * dt * step)`
//! - **vel** — constant velocity on box faces: `L(t) = L0 + velocity * dt * step`
//! - **final** — linear ramp to a target value over the run
//! - **xy** — Lees–Edwards simple shear (see below)
//!
//! Atom positions are remapped proportionally (affine transformation) when the
//! box changes, ensuring particles stay inside the domain.
//!
//! ## Center-symmetric, not one-sided
//!
//! For `erate` and `vel`, the box is resized **symmetrically about its center**:
//! *both* faces move so the box center stays put. There is no single-platen mode
//! here — `vel = v` moves each face by `v/2·dt·step`, not one face by `v·dt·step`.
//! Use a `[[pin]]` group plus a velocity-driven wall in a physics tier if you
//! need a genuinely one-sided platen.
//!
//! # Configuration
//!
//! ```toml
//! [deform]
//! # Constant engineering strain rate on z-axis (compression = negative)
//! z = { style = "erate", rate = -0.001 }
//!
//! # Constant velocity on x-axis faces
//! # x = { style = "vel", velocity = 0.01 }
//!
//! # Ramp y-axis to target bounds over the run
//! # y = { style = "final", lo = 0.0, hi = 0.02 }
//!
//! # Lees–Edwards simple shear: tilt x (flow) as a function of y (gradient)
//! # xy = { style = "erate", rate = 0.5 }   # shear-strain rate γ̇ (units 1/s)
//!
//! # Remap atom positions (default: true)
//! remap = true
//! ```
//!
//! # Lees–Edwards `xy` simple shear
//!
//! `xy = { style = "erate", rate = γ̇ }` drives the box into a **triclinic**
//! (tilted) state to impose simple shear in the x (flow) direction as a function
//! of y (gradient). Key facts:
//!
//! - `rate` is a **shear-strain rate** γ̇ (units 1/s), *not* a length rate. The
//!   tilt grows as `xy(t) = γ̇·L_y·t` and the gradient-direction edge moves at
//!   `Δv = γ̇·L_y`. Ghost copies wrapping the sheared y-face pick up that
//!   streaming-velocity jump.
//! - It **requires periodic x and y boundaries**; invalid setup is reported as
//!   an actionable configuration error.
//! - Only `style = "erate"` is supported for `xy`.
//! - On the first shear step a linear streaming profile `v_x = γ̇·(y − y_center)`
//!   is imposed once so the system starts near steady simple shear.
//! - The tilt is box-flip-wrapped into `[−Lx/2, Lx/2]` so the ghost/stencil reach
//!   stays bounded.
//!
//! # Scheduling
//!
//! The deform system runs at [`ParticleSimScheduleSet::PreInitialIntegration`], updating
//! domain bounds and remapping atoms before the Verlet position update. Every box
//! change sets `domain.bounds_changed` and clears the neighbor list's last-build
//! positions, forcing a neighbor rebuild on the next step.
//!
//! # Multi-stage / time-origin semantics
//!
//! The `[deform]` config is **re-read at the start of every `[[run]]` stage**
//! (the setup system runs at [`ScheduleSetupSet::PostSetup`]), so per-stage
//! overrides take effect. At each stage boundary:
//!
//! - the current domain bounds are snapshotted as `lo_0`/`hi_0` (the deformation
//!   origin), so strain is measured from the *start of the stage*, not the start
//!   of the run;
//! - the elapsed-step counter `step` is reset to 0;
//! - a stage with **no** `[deform]` section clears all axes and the xy shear, so
//!   deformation does not silently persist from a previous stage.

#![warn(missing_docs)]

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::Deserialize;
use soil_core::Neighbor;
use soil_core::{
    stage_tables, Atom, CommResource, Config, Domain, ParticleSimScheduleSet, Real,
    ScheduleSetupSet, StageOverrides,
};
use std::fmt;

// ── TOML config structs ─────────────────────────────────────────────────────

/// Serde default for boolean fields that should be `true` when omitted.
fn default_true() -> bool {
    true
}

/// Human-readable axis names indexed by dimension (0=x, 1=y, 2=z).
const AXIS_NAMES: [&str; 3] = ["x", "y", "z"];

/// Per-axis deformation definition parsed from TOML.
///
/// Each axis (`x`, `y`, `z`) can independently specify a deformation style.
/// Only the fields relevant to the chosen style are required; unused fields
/// should be omitted.
///
/// ```toml
/// z = { style = "erate", rate = -0.001 }
/// x = { style = "vel", velocity = 0.01 }
/// y = { style = "final", lo = 0.0, hi = 0.02 }
/// ```
#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct AxisDeformDef {
    /// Deformation style: `"erate"`, `"vel"`, or `"final"`.
    pub style: String,
    /// Engineering strain rate (required for `"erate"` style). Negative = compression.
    #[serde(default)]
    pub rate: Option<f64>,
    /// Constant velocity applied to box faces (required for `"vel"` style).
    /// Positive = expansion, negative = compression.
    #[serde(default)]
    pub velocity: Option<f64>,
    /// Target lower bound (required for `"final"` style).
    #[serde(default)]
    pub lo: Option<f64>,
    /// Target upper bound (required for `"final"` style).
    #[serde(default)]
    pub hi: Option<f64>,
}

/// Top-level `[deform]` TOML configuration section.
///
/// Each axis can be independently controlled (or omitted to leave it unchanged).
/// When `remap` is true (the default), atom positions are affinely scaled to
/// follow the box deformation, preserving their relative positions within the
/// domain.
///
/// # Example
///
/// ```toml
/// [deform]
/// z = { style = "erate", rate = -0.001 }  # uniaxial compression on z
/// remap = true
/// ```
#[derive(Deserialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct DeformConfig {
    /// X-axis deformation definition (omit to leave x unchanged).
    #[serde(default)]
    pub x: Option<AxisDeformDef>,
    /// Y-axis deformation definition (omit to leave y unchanged).
    #[serde(default)]
    pub y: Option<AxisDeformDef>,
    /// Z-axis deformation definition (omit to leave z unchanged).
    #[serde(default)]
    pub z: Option<AxisDeformDef>,
    /// xy-shear (Lees–Edwards simple shear) definition: tilts the box in the x
    /// (flow) direction as a function of y (gradient). `xy = { style = "erate",
    /// rate = γ̇ }` drives the tilt at engineering shear-strain rate γ̇ (units 1/s),
    /// i.e. `xy(t) = γ̇·L_y·t`, with the streaming-velocity remap applied on
    /// y-crossings. Requires periodic x and y. Omit for no shear.
    #[serde(default)]
    pub xy: Option<AxisDeformDef>,
    /// Whether to remap atom positions proportionally when the box changes.
    /// Default: `true`.
    #[serde(default = "default_true")]
    pub remap: bool,
}

// ── Runtime types ───────────────────────────────────────────────────────────

/// Deformation style for a single axis.
#[derive(Clone, Debug)]
pub enum DeformStyle {
    /// Engineering strain rate: L(t) = L0 * (1 + rate * dt * step)
    Erate {
        /// Engineering strain rate (per unit time).
        rate: f64,
    },
    /// Constant velocity on box faces: L(t) = L0 + velocity * dt * step
    Vel {
        /// Face velocity (length per unit time).
        velocity: f64,
    },
    /// Linear ramp to target lo/hi over the run duration.
    Final {
        /// Target lower box bound at the end of the run.
        target_lo: f64,
        /// Target upper box bound at the end of the run.
        target_hi: f64,
    },
}

/// User-facing validation error for malformed `[deform]` configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeformConfigError {
    message: String,
}

impl DeformConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for DeformConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DeformConfigError {}

fn exit_deform_error(err: DeformConfigError) -> ! {
    eprintln!("ERROR: {err}");
    std::process::exit(1);
}

/// Per-axis deformation state.
#[derive(Clone, Debug)]
pub struct AxisDeform {
    /// How this axis deforms (strain rate, face velocity, or ramp-to-target).
    pub style: DeformStyle,
    /// Initial lower bound at start of deformation.
    pub lo_0: f64,
    /// Initial upper bound at start of deformation.
    pub hi_0: f64,
}

/// Runtime resource holding the current deformation state for all axes.
///
/// Registered as a resource by [`DeformPlugin`]. The setup system reinitializes
/// this each stage from the (possibly overridden) `[deform]` config, and the
/// update system mutates `step` and domain bounds every timestep.
pub struct DeformState {
    /// Per-axis deform (`None` = axis not deforming). Indices: 0=x, 1=y, 2=z.
    pub axes: [Option<AxisDeform>; 3],
    /// xy Lees–Edwards shear-strain rate γ̇ (`None` = no shear). When set, the box
    /// is driven triclinic: `xy(t) = γ̇·L_y·t` (box-flip-wrapped to `[-Lx/2, Lx/2]`)
    /// and the gradient-direction edge moves at `Δv = γ̇·L_y`.
    pub shear_xy_rate: Option<f64>,
    /// Whether the initial linear streaming profile `v_x = γ̇·(y − y_center)` has
    /// been imposed on the local atoms (done once at the first shear step).
    pub shear_initialized: bool,
    /// Whether to remap atom positions affinely when the box changes.
    pub remap: bool,
    /// Number of timesteps elapsed since the current stage's deformation started.
    pub step: usize,
    /// Whether the deform state has been initialized with domain bounds.
    pub initialized: bool,
}

impl DeformState {
    /// Returns `true` if at least one axis or the xy shear has an active definition.
    pub fn has_any(&self) -> bool {
        self.axes.iter().any(|a| a.is_some()) || self.shear_xy_rate.is_some()
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Plugin that enables simulation box deformation during a run.
///
/// Registers the [`DeformState`] resource and two systems:
/// - **Setup** ([`ScheduleSetupSet::PostSetup`]): reads the `[deform]` config
///   (including per-stage overrides) and initializes axis definitions.
/// - **Update** ([`ParticleSimScheduleSet::PreInitialIntegration`]): applies the
///   deformation each timestep before the Verlet position update.
///
/// Add this plugin to your app to enable `[deform]` TOML configuration:
///
/// ```ignore
/// app.add_plugin(DeformPlugin);
/// ```
pub struct DeformPlugin;

impl Plugin for DeformPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# [deform]
# Per-axis box deformation (compression, expansion, strain control)
# z = { style = "erate", rate = -0.001 }    # engineering strain rate
# x = { style = "vel", velocity = 0.01 }    # constant velocity
# y = { style = "final", lo = 0.0, hi = 0.02 }  # ramp to target
# remap = true  # remap atom positions (default: true)"#,
        )
    }

    fn build(&self, app: &mut App) {
        Config::load::<DeformConfig>(app, "deform");
        install_deform(app);
    }

    fn try_build(&self, app: &mut App) -> Result<(), AppError> {
        if let Some(config) = app.get_resource_ref::<Config>() {
            for table in stage_tables(&config).map_err(|e| AppError::message(e.to_string()))? {
                let config = Config::from_table(table)
                    .try_section::<DeformConfig>("deform")
                    .map_err(|e| {
                        AppError::message(format!("failed to parse [deform] section: {e}"))
                    })?;
                validate_deform_config(&config, &Domain::new(), 1)
                    .map_err(|e| AppError::message(e.to_string()))?;
            }
        }
        // Register a default (empty) DeformState. The setup system re-reads
        // the merged stage-aware config each stage, so we don't parse axes here.
        Config::try_load::<DeformConfig>(app, "deform")
            .map_err(|e| AppError::message(e.to_string()))?;

        install_deform(app);
        Ok(())
    }
}

fn install_deform(app: &mut App) {
        let state = DeformState {
            axes: [None, None, None],
            shear_xy_rate: None,
            shear_initialized: false,
            remap: true,
            step: 0,
            initialized: false,
        };

        // Always register systems — the setup system reads per-stage config and
        // populates axes (or clears them) accordingly. The update system early-
        // returns when no axes are active.
        app.add_resource(state)
            .add_setup_system(setup_deform, ScheduleSetupSet::PostSetup)
            .add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
}

/// Parse an optional axis definition from config into a runtime [`AxisDeform`].
///
/// Returns `None` if `def` is `None` (axis not configured). The `lo_0`/`hi_0`
/// fields are initialized to zero here and filled in by the setup system once
/// the current domain bounds are known.
///
fn parse_axis_def(
    def: &Option<AxisDeformDef>,
    axis_name: &str,
) -> Result<Option<AxisDeform>, DeformConfigError> {
    let Some(def) = def.as_ref() else {
        return Ok(None);
    };
    let style = match def.style.as_str() {
        "erate" => {
            let rate = def.rate.ok_or_else(|| {
                DeformConfigError::new(format!(
                    "[deform] {axis_name}: style 'erate' requires a 'rate' field"
                ))
            })?;
            DeformStyle::Erate { rate }
        }
        "vel" => {
            let velocity = def.velocity.ok_or_else(|| {
                DeformConfigError::new(format!(
                    "[deform] {axis_name}: style 'vel' requires a 'velocity' field"
                ))
            })?;
            DeformStyle::Vel { velocity }
        }
        "final" => {
            let lo = def.lo.ok_or_else(|| {
                DeformConfigError::new(format!(
                    "[deform] {axis_name}: style 'final' requires a 'lo' field"
                ))
            })?;
            let hi = def.hi.ok_or_else(|| {
                DeformConfigError::new(format!(
                    "[deform] {axis_name}: style 'final' requires a 'hi' field"
                ))
            })?;
            DeformStyle::Final {
                target_lo: lo,
                target_hi: hi,
            }
        }
        other => {
            return Err(DeformConfigError::new(format!(
                "[deform] {axis_name}: unknown style '{other}'. Use 'erate', 'vel', or 'final'."
            )));
        }
    };

    Ok(Some(AxisDeform {
        style,
        lo_0: 0.0,
        hi_0: 0.0,
    }))
}

/// Parse the `[deform] xy = { style = "erate", rate = γ̇ }` shear definition into a
/// shear-strain rate. Returns `None` if no shear is configured.
fn parse_shear_rate(def: &Option<AxisDeformDef>) -> Result<Option<f64>, DeformConfigError> {
    let Some(def) = def.as_ref() else {
        return Ok(None);
    };
    match def.style.as_str() {
        "erate" => {
            let rate = def.rate.ok_or_else(|| {
                DeformConfigError::new(
                    "[deform] xy: style 'erate' requires a 'rate' field (shear-strain rate)",
                )
            })?;
            Ok(Some(rate))
        }
        other => Err(DeformConfigError::new(format!(
            "[deform] xy: unsupported style '{other}'. Lees-Edwards shear only supports style 'erate'."
        ))),
    }
}

fn validate_deform_config(
    config: &DeformConfig,
    domain: &Domain,
    mpi_ranks: i32,
) -> Result<([Option<AxisDeform>; 3], Option<f64>), DeformConfigError> {
    let axes = [
        parse_axis_def(&config.x, "x")?,
        parse_axis_def(&config.y, "y")?,
        parse_axis_def(&config.z, "z")?,
    ];
    let shear_xy_rate = parse_shear_rate(&config.xy)?;
    let has_active_deform = axes.iter().any(|axis| axis.is_some()) || shear_xy_rate.is_some();

    if has_active_deform && mpi_ranks > 1 {
        return Err(DeformConfigError::new(format!(
            "soil_deform does not support active [deform] with MPI rank count {mpi_ranks}: deformation currently performs an unsupported subdomain-bound update by setting each rank's sub_domain_low/sub_domain_high to the global box. Run with one MPI rank or disable [deform] until MPI subdomain deformation is implemented."
        )));
    }

    if shear_xy_rate.is_some() && (!domain.is_periodic(0) || !domain.is_periodic(1)) {
        return Err(DeformConfigError::new(format!(
            "[deform] xy shear requires periodic x and y boundaries; got x={:?}, y={:?}",
            domain.boundary_type[0], domain.boundary_type[1]
        )));
    }

    Ok((axes, shear_xy_rate))
}

fn next_axis_bounds(
    axis: &AxisDeform,
    axis_name: &str,
    dt: f64,
    step: usize,
    total_steps: usize,
) -> Result<(f64, f64), DeformConfigError> {
    let lo_0 = axis.lo_0;
    let hi_0 = axis.hi_0;
    let l0 = hi_0 - lo_0;

    let (new_lo, new_hi) = match &axis.style {
        DeformStyle::Erate { rate } => {
            // L(t) = L0 * (1 + rate * dt * step)
            // Symmetric expansion/contraction about center
            let scale = 1.0 + rate * dt * step as f64;
            let new_l = l0 * scale;
            let center = (lo_0 + hi_0) * 0.5;
            (center - new_l * 0.5, center + new_l * 0.5)
        }
        DeformStyle::Vel { velocity } => {
            // L(t) = L0 + velocity * dt * step
            // Symmetric: each face moves by velocity/2 * dt * step
            let delta = velocity * dt * step as f64;
            let new_l = l0 + delta;
            let center = (lo_0 + hi_0) * 0.5;
            (center - new_l * 0.5, center + new_l * 0.5)
        }
        DeformStyle::Final {
            target_lo,
            target_hi,
        } => {
            // Linear interpolation from initial to target
            let frac = if total_steps > 0 {
                (step as f64) / (total_steps as f64)
            } else {
                1.0
            };
            let frac = frac.min(1.0);
            let new_lo = lo_0 + (target_lo - lo_0) * frac;
            let new_hi = hi_0 + (target_hi - hi_0) * frac;
            (new_lo, new_hi)
        }
    };

    if new_hi - new_lo <= 0.0 {
        return Err(DeformConfigError::new(format!(
            "[deform] {axis_name}: box would collapse at step {step} (lo={new_lo}, hi={new_hi}); reduce compression rate/velocity or choose final bounds with hi > lo"
        )));
    }

    Ok((new_lo, new_hi))
}

fn needs_stage_total_steps(state: &DeformState) -> bool {
    state.axes.iter().flatten().any(|axis| {
        matches!(
            axis.style,
            DeformStyle::Final {
                target_lo: _,
                target_hi: _
            }
        )
    })
}

fn current_stage_total_steps(
    run_state: Option<&soil_core::RunState>,
    scheduler_manager: Option<&SchedulerManager>,
) -> Result<usize, DeformConfigError> {
    let run_state = run_state.ok_or_else(|| {
        DeformConfigError::new(
            "[deform] final style requires RunState to determine the current stage's total steps; add RunPlugin before DeformPlugin",
        )
    })?;
    let scheduler_manager = scheduler_manager.ok_or_else(|| {
        DeformConfigError::new(
            "[deform] final style requires SchedulerManager to determine the current run stage; use the grass scheduler/RunPlugin",
        )
    })?;
    let stage_idx = scheduler_manager.index;

    let count_len = run_state.cycle_count.len();
    let total_len = run_state.cycle_remaining.len();
    if stage_idx >= count_len || stage_idx >= total_len {
        return Err(DeformConfigError::new(format!(
            "[deform] final style cannot determine total steps for current stage {stage_idx}: RunState has cycle_count len {count_len} and cycle_remaining len {total_len}; ensure RunPlugin initialized this stage before deformation"
        )));
    }

    Ok(run_state.cycle_remaining[stage_idx] as usize)
}

// ── Setup system ────────────────────────────────────────────────────────────

/// Re-read deformation config from stage overrides and initialize state.
///
/// This runs at the start of every stage, so per-stage `[deform]` overrides
/// inside `[[run]]` blocks take effect. When a stage has no `[deform]`
/// section, all axes are cleared so deformation doesn't persist from a
/// previous stage.
fn setup_deform(
    mut state: ResMut<DeformState>,
    domain: Res<Domain>,
    comm: Res<CommResource>,
    stage_overrides: Res<StageOverrides>,
) {
    // Re-read config from the merged (global + current stage) config table.
    let config: DeformConfig = stage_overrides.section("deform");

    // Parse axis definitions fresh from this stage's config.
    let (axes, shear_xy_rate) = validate_deform_config(&config, &domain, comm.size())
        .unwrap_or_else(|err| exit_deform_error(err));
    state.axes = axes;
    state.shear_xy_rate = shear_xy_rate;
    state.shear_initialized = false;
    state.remap = config.remap;

    // Set initial bounds from current domain and reset step counter.
    for (dim, axis) in state.axes.iter_mut().enumerate() {
        if let Some(ref mut ax) = axis {
            ax.lo_0 = domain.boundaries_low[dim];
            ax.hi_0 = domain.boundaries_high[dim];
        }
    }
    state.initialized = true;
    state.step = 0;

    if comm.rank() == 0 && state.has_any() {
        for (dim, axis) in state.axes.iter().enumerate() {
            if let Some(ref ax) = axis {
                let range = ax.hi_0 - ax.lo_0;
                match &ax.style {
                    DeformStyle::Erate { rate } => {
                        println!(
                            "Deform {}: erate rate={:.6e}, L0={:.6e} [{:.6e}, {:.6e}]",
                            AXIS_NAMES[dim], rate, range, ax.lo_0, ax.hi_0
                        );
                    }
                    DeformStyle::Vel { velocity } => {
                        println!(
                            "Deform {}: vel velocity={:.6e}, L0={:.6e} [{:.6e}, {:.6e}]",
                            AXIS_NAMES[dim], velocity, range, ax.lo_0, ax.hi_0
                        );
                    }
                    DeformStyle::Final {
                        target_lo,
                        target_hi,
                    } => {
                        println!(
                            "Deform {}: final [{:.6e}, {:.6e}] -> [{:.6e}, {:.6e}]",
                            AXIS_NAMES[dim], ax.lo_0, ax.hi_0, target_lo, target_hi
                        );
                    }
                }
            }
        }
        if let Some(rate) = state.shear_xy_rate {
            let ly = domain.boundaries_high[1] - domain.boundaries_low[1];
            println!(
                "Deform xy: Lees-Edwards shear erate={:.6e} (γ̇), Δv=γ̇·Ly={:.6e}",
                rate,
                rate * ly
            );
        }
    }
}

// ── Update system ───────────────────────────────────────────────────────────

/// Apply box deformation each timestep. Runs at [`ParticleSimScheduleSet::PreInitialIntegration`].
///
/// 1. Compute new domain bounds based on deformation style and current step.
/// 2. Remap atom positions proportionally (affine transform).
/// 3. Update domain resource (bounds, size, volume, sub-domain).
/// 4. Signal neighbor rebuild needed by clearing last-build positions.
///
/// If any axis collapses to zero or negative size (e.g., excessive compression
/// rate), the user receives a clear error and the process exits without a panic.
fn apply_deform(
    mut atoms: ResMut<Atom>,
    mut domain: ResMut<Domain>,
    mut state: ResMut<DeformState>,
    mut neighbor: ResMut<Neighbor>,
    run_state: Option<Res<soil_core::RunState>>,
    scheduler_manager: Option<Res<SchedulerManager>>,
) {
    if !state.initialized || !state.has_any() {
        return;
    }

    state.step += 1;
    let dt = atoms.dt;
    let step = state.step;
    let nlocal = atoms.nlocal as usize;

    // Total steps for the current stage, needed by "final" style to compute
    // interpolation fraction. RunState stores the configured stage step count in
    // `cycle_remaining[current_stage]`; validate that the scheduler stage has a
    // matching entry rather than silently treating missing data as a one-step run.
    let total_steps = if needs_stage_total_steps(&state) {
        current_stage_total_steps(run_state.as_deref(), scheduler_manager.as_deref())
            .unwrap_or_else(|err| exit_deform_error(err))
    } else {
        0
    };

    let mut box_changed = false;

    for dim in 0..3 {
        let axis = match &state.axes[dim] {
            Some(ax) => ax,
            None => continue,
        };

        let old_lo = domain.boundaries_low[dim];
        let old_hi = domain.boundaries_high[dim];
        let old_size = old_hi - old_lo;

        let (new_lo, new_hi) = next_axis_bounds(axis, AXIS_NAMES[dim], dt, step, total_steps)
            .unwrap_or_else(|err| exit_deform_error(err));

        let new_size = new_hi - new_lo;

        // Remap atom positions proportionally
        if state.remap && old_size > 0.0 {
            let scale = new_size / old_size;
            let new_center = (new_lo + new_hi) * 0.5;
            let old_center = (old_lo + old_hi) * 0.5;

            for i in 0..nlocal {
                // Affine transform: shift to old center, scale, shift to new center.
                // Geometry is f64; widen pos, transform, store back as Real.
                atoms.pos[i][dim] =
                    (new_center + (atoms.pos[i][dim] as f64 - old_center) * scale) as Real;
            }
        }

        // Update domain bounds
        domain.boundaries_low[dim] = new_lo;
        domain.boundaries_high[dim] = new_hi;
        domain.size[dim] = new_size;

        // Update sub-domain bounds (single-processor: same as global)
        // For multi-proc, this is a simplification; full MPI would need
        // re-decomposition, but for single-proc it's correct.
        domain.sub_domain_low[dim] = new_lo;
        domain.sub_domain_high[dim] = new_hi;
        domain.sub_length[dim] = new_size;

        box_changed = true;
    }

    // ── Lees–Edwards xy shear (triclinic) ────────────────────────────────────
    if let Some(rate) = state.shear_xy_rate {
        let lx = domain.size[0];
        let ly = domain.size[1];
        // Streaming-velocity jump across the moving gradient (y) box edge.
        let dv = rate * ly;

        // One-time linear streaming profile v_x = γ̇·(y − y_center), so the system
        // starts near steady simple shear (centered → ~zero net x-momentum).
        if !state.shear_initialized {
            let yc = 0.5 * (domain.boundaries_low[1] + domain.boundaries_high[1]);
            for i in 0..nlocal {
                atoms.vel[i][0] += (rate * (atoms.pos[i][1] as f64 - yc)) as Real;
            }
            state.shear_initialized = true;
        }

        // Engineering tilt xy(t) = γ̇·Ly·t, box-flip-wrapped into [−Lx/2, Lx/2].
        // The wrap is an exact relabeling to an equivalent periodic image (b → b−a),
        // which keeps the ghost/stencil reach bounded; atoms re-image in `pbc`.
        let raw = rate * dt * step as f64 * ly;
        let xy = raw - lx * (raw / lx).round();
        domain.tilt[0] = xy;
        domain.boundary_vel[0] = dv;
        domain.triclinic = true;
        // Tilt grows each step → rebuild the tilt-dependent neighbor stencil.
        domain.bounds_changed = true;
        box_changed = true;
    }

    if box_changed {
        // Update volume
        domain.volume = domain.size[0] * domain.size[1] * domain.size[2];

        // Recompute the neighbor bin grid for the new box extents. Without this,
        // a deforming box keeps a stale grid: on expansion atoms spread past the
        // grid, clamp into edge cells, and a stencil offset from an edge cell can
        // run past `bin_total_cells` → out-of-bounds in the (unchecked) neighbor
        // build (a SIGSEGV seen on dilute box expansion). `sort_atoms_by_bin`
        // calls `recompute_bins` when this flag is set.
        domain.bounds_changed = true;

        // Force neighbor rebuild by clearing last build positions
        neighbor.last_build_pos.clear();
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use grass_app::App;
    use soil_core::BoundaryType;
    use soil_core::RunState;

    // Local test helper: build `n` atoms laid out along x. Kept inline so SOIL
    // builds standalone without depending on any physics-tier test crate.
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

    #[test]
    fn malformed_deform_is_returned_by_fallible_plugin_registration() {
        let mut app = App::new();
        app.add_resource(Config::from_str(
            r#"[deform]
x = { style = "erate" }
"#,
        ));

        let error = match app.try_add_plugins(DeformPlugin) {
            Ok(_) => panic!("malformed configuration unexpectedly registered"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("requires a 'rate' field"),
            "unexpected error: {error}"
        );
    }

    fn make_domain(lo: [f64; 3], hi: [f64; 3]) -> Domain {
        let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let mut d = Domain::new();
        d.boundaries_low = lo;
        d.boundaries_high = hi;
        d.sub_domain_low = lo;
        d.sub_domain_high = hi;
        d.sub_length = size;
        d.size = size;
        d.volume = size[0] * size[1] * size[2];
        d
    }

    fn make_neighbor() -> Neighbor {
        Neighbor::new()
    }

    fn make_run_state(total_steps: u32) -> RunState {
        let mut rs = RunState::new();
        rs.cycle_count.push(0);
        rs.cycle_remaining.push(total_steps);
        rs
    }

    fn make_multi_stage_run_state(stage_steps: &[u32]) -> RunState {
        let mut rs = RunState::new();
        for &steps in stage_steps {
            rs.cycle_count.push(0);
            rs.cycle_remaining.push(steps);
        }
        rs
    }

    fn make_scheduler_manager(index: usize) -> SchedulerManager {
        let mut scheduler_manager = SchedulerManager::new();
        scheduler_manager.index = index;
        scheduler_manager
    }

    #[test]
    fn erate_shrinks_box_symmetrically() {
        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let mut atoms = make_atoms(0);
        atoms.dt = 0.001;
        let neighbor = make_neighbor();
        let run_state = make_run_state(1000);

        let state = DeformState {
            axes: [
                None,
                None,
                Some(AxisDeform {
                    style: DeformStyle::Erate { rate: -1.0 },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
            ],
            remap: true,
            shear_xy_rate: None,
            shear_initialized: false,
            step: 0,
            initialized: true,
        };

        // After 1 step with rate=-1.0, dt=0.001:
        // scale = 1 + (-1.0) * 0.001 * 1 = 0.999
        // new L = 1.0 * 0.999 = 0.999
        // center = 0.5 => new_lo = 0.5 - 0.4995 = 0.0005, new_hi = 0.5 + 0.4995 = 0.9995
        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let d = app.get_resource_ref::<Domain>().unwrap();
        assert!((d.boundaries_low[2] - 0.0005).abs() < 1e-12);
        assert!((d.boundaries_high[2] - 0.9995).abs() < 1e-12);
        assert!((d.size[2] - 0.999).abs() < 1e-12);
        // x and y should be unchanged
        assert!((d.boundaries_low[0]).abs() < 1e-12);
        assert!((d.boundaries_high[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn vel_expands_box() {
        let mut atoms = make_atoms(0);
        atoms.dt = 0.01;
        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let neighbor = make_neighbor();
        let run_state = make_run_state(100);

        let state = DeformState {
            axes: [
                Some(AxisDeform {
                    style: DeformStyle::Vel { velocity: 1.0 },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
                None,
                None,
            ],
            remap: true,
            shear_xy_rate: None,
            shear_initialized: false,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        // After 1 step: delta = 1.0 * 0.01 * 1 = 0.01
        // new_l = 1.0 + 0.01 = 1.01, center = 0.5
        // new_lo = 0.5 - 0.505 = -0.005, new_hi = 0.5 + 0.505 = 1.005
        let d = app.get_resource_ref::<Domain>().unwrap();
        assert!((d.size[0] - 1.01).abs() < 1e-12);
    }

    #[test]
    fn final_ramps_to_target() {
        let mut atoms = make_atoms(0);
        atoms.dt = 0.001;
        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let neighbor = make_neighbor();
        let run_state = make_run_state(10); // 10 total steps

        let state = DeformState {
            axes: [
                None,
                Some(AxisDeform {
                    style: DeformStyle::Final {
                        target_lo: 0.1,
                        target_hi: 0.9,
                    },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
                None,
            ],
            remap: true,
            shear_xy_rate: None,
            shear_initialized: false,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_resource(make_scheduler_manager(0));
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();

        // Run 10 steps
        for _ in 0..10 {
            app.run();
        }

        let d = app.get_resource_ref::<Domain>().unwrap();
        // After 10 steps of 10 total: frac = 1.0
        // new_lo = 0.0 + (0.1 - 0.0) * 1.0 = 0.1
        // new_hi = 1.0 + (0.9 - 1.0) * 1.0 = 0.9
        assert!((d.boundaries_low[1] - 0.1).abs() < 1e-10);
        assert!((d.boundaries_high[1] - 0.9).abs() < 1e-10);
    }

    #[test]
    fn final_uses_current_stage_total_steps_in_multi_stage_run_state() {
        let mut atoms = make_atoms(0);
        atoms.dt = 0.001;
        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let neighbor = make_neighbor();
        let run_state = make_multi_stage_run_state(&[3, 0, 4]);

        let state = DeformState {
            axes: [
                Some(AxisDeform {
                    style: DeformStyle::Final {
                        target_lo: -0.5,
                        target_hi: 1.5,
                    },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
                None,
                None,
            ],
            remap: true,
            shear_xy_rate: None,
            shear_initialized: false,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_resource(make_scheduler_manager(2));
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();

        for _ in 0..2 {
            app.run();
        }

        {
            let d = app.get_resource_ref::<Domain>().unwrap();
            assert!((d.boundaries_low[0] - (-0.25)).abs() < 1e-12);
            assert!((d.boundaries_high[0] - 1.25).abs() < 1e-12);
        }

        for _ in 0..2 {
            app.run();
        }

        let d = app.get_resource_ref::<Domain>().unwrap();
        assert!((d.boundaries_low[0] - (-0.5)).abs() < 1e-12);
        assert!((d.boundaries_high[0] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn final_zero_step_stage_uses_zero_total_without_one_step_fallback() {
        let mut atoms = make_atoms(0);
        atoms.dt = 0.001;
        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let neighbor = make_neighbor();
        let run_state = make_multi_stage_run_state(&[5, 0]);

        let state = DeformState {
            axes: [
                None,
                Some(AxisDeform {
                    style: DeformStyle::Final {
                        target_lo: 0.2,
                        target_hi: 0.8,
                    },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
                None,
            ],
            remap: true,
            shear_xy_rate: None,
            shear_initialized: false,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_resource(make_scheduler_manager(1));
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let d = app.get_resource_ref::<Domain>().unwrap();
        assert!((d.boundaries_low[1] - 0.2).abs() < 1e-12);
        assert!((d.boundaries_high[1] - 0.8).abs() < 1e-12);
    }

    #[test]
    fn final_missing_run_state_reports_validation_error() {
        let scheduler_manager = make_scheduler_manager(0);
        let err = current_stage_total_steps(None, Some(&scheduler_manager)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] final style requires RunState to determine the current stage's total steps; add RunPlugin before DeformPlugin"
        );
    }

    #[test]
    fn final_stale_run_state_stage_index_reports_validation_error() {
        let run_state = make_run_state(4);
        let scheduler_manager = make_scheduler_manager(1);
        let err =
            current_stage_total_steps(Some(&run_state), Some(&scheduler_manager)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] final style cannot determine total steps for current stage 1: RunState has cycle_count len 1 and cycle_remaining len 1; ensure RunPlugin initialized this stage before deformation"
        );
    }

    #[test]
    fn remap_scales_atom_positions() {
        let mut atoms = Atom::new();
        atoms.dt = 0.001;
        // Place atom at center of box
        atoms.push_test_atom(0, [0.5, 0.5, 0.5], 0.1, 1.0);
        // Place atom at 3/4 of box
        atoms.push_test_atom(1, [0.75, 0.5, 0.75], 0.1, 1.0);
        atoms.nlocal = 2;
        atoms.natoms = 2;

        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let neighbor = make_neighbor();
        let run_state = make_run_state(1000);

        let state = DeformState {
            axes: [
                None,
                None,
                Some(AxisDeform {
                    style: DeformStyle::Erate { rate: -1.0 },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
            ],
            remap: true,
            shear_xy_rate: None,
            shear_initialized: false,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        let d = app.get_resource_ref::<Domain>().unwrap();

        // Atom 0 at center should stay at center
        let center_z = (d.boundaries_low[2] + d.boundaries_high[2]) * 0.5;
        assert!(
            (a.pos[0][2] as f64 - center_z).abs() < 1e-12,
            "Center atom should stay at center: {} vs {}",
            a.pos[0][2],
            center_z
        );

        // x positions should be unchanged (no x deform)
        assert!((a.pos[0][0] - 0.5).abs() < 1e-12);
        assert!((a.pos[1][0] - 0.75).abs() < 1e-12);
    }

    #[test]
    fn no_remap_leaves_atoms_in_place() {
        let mut atoms = Atom::new();
        atoms.dt = 0.001;
        atoms.push_test_atom(0, [0.5, 0.5, 0.5], 0.1, 1.0);
        atoms.nlocal = 1;
        atoms.natoms = 1;

        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let neighbor = make_neighbor();
        let run_state = make_run_state(1000);

        let state = DeformState {
            axes: [
                None,
                None,
                Some(AxisDeform {
                    style: DeformStyle::Erate { rate: -1.0 },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
            ],
            remap: false,
            shear_xy_rate: None,
            shear_initialized: false,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let a = app.get_resource_ref::<Atom>().unwrap();
        // Position should be unchanged
        assert!((a.pos[0][2] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn multiple_axes_deform_independently() {
        let mut atoms = make_atoms(0);
        atoms.dt = 0.01;
        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let neighbor = make_neighbor();
        let run_state = make_run_state(100);

        let state = DeformState {
            axes: [
                Some(AxisDeform {
                    style: DeformStyle::Vel { velocity: 1.0 },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
                None,
                Some(AxisDeform {
                    style: DeformStyle::Erate { rate: -0.5 },
                    lo_0: 0.0,
                    hi_0: 1.0,
                }),
            ],
            remap: true,
            shear_xy_rate: None,
            shear_initialized: false,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let d = app.get_resource_ref::<Domain>().unwrap();

        // X: vel=1.0, dt=0.01, step=1 => delta=0.01, new_l=1.01
        assert!((d.size[0] - 1.01).abs() < 1e-12);

        // Y: unchanged
        assert!((d.size[1] - 1.0).abs() < 1e-12);

        // Z: erate=-0.5, dt=0.01, step=1 => scale = 1 + (-0.5)*0.01*1 = 0.995
        assert!((d.size[2] - 0.995).abs() < 1e-12);
    }

    #[test]
    fn volume_updates_correctly() {
        let mut atoms = make_atoms(0);
        atoms.dt = 0.001;
        let domain = make_domain([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
        let neighbor = make_neighbor();
        let run_state = make_run_state(1000);

        let state = DeformState {
            axes: [
                None,
                None,
                Some(AxisDeform {
                    style: DeformStyle::Erate { rate: -1.0 },
                    lo_0: 0.0,
                    hi_0: 4.0,
                }),
            ],
            remap: true,
            shear_xy_rate: None,
            shear_initialized: false,
            step: 0,
            initialized: true,
        };

        let mut app = App::new();
        app.add_resource(atoms);
        app.add_resource(domain);
        app.add_resource(state);
        app.add_resource(neighbor);
        app.add_resource(run_state);
        app.add_update_system(apply_deform, ParticleSimScheduleSet::PreInitialIntegration);
        app.organize_systems();
        app.run();

        let d = app.get_resource_ref::<Domain>().unwrap();
        let expected_vol = d.size[0] * d.size[1] * d.size[2];
        assert!((d.volume - expected_vol).abs() < 1e-10);
    }

    #[test]
    fn parse_erate_config() {
        let def = AxisDeformDef {
            style: "erate".to_string(),
            rate: Some(-0.001),
            velocity: None,
            lo: None,
            hi: None,
        };
        let axis = parse_axis_def(&Some(def), "z").unwrap().unwrap();
        match axis.style {
            DeformStyle::Erate { rate } => assert!((rate - (-0.001)).abs() < 1e-15),
            _ => panic!("Expected Erate style"),
        }
    }

    #[test]
    fn parse_vel_config() {
        let def = AxisDeformDef {
            style: "vel".to_string(),
            rate: None,
            velocity: Some(0.5),
            lo: None,
            hi: None,
        };
        let axis = parse_axis_def(&Some(def), "x").unwrap().unwrap();
        match axis.style {
            DeformStyle::Vel { velocity } => assert!((velocity - 0.5).abs() < 1e-15),
            _ => panic!("Expected Vel style"),
        }
    }

    #[test]
    fn parse_final_config() {
        let def = AxisDeformDef {
            style: "final".to_string(),
            rate: None,
            velocity: None,
            lo: Some(0.1),
            hi: Some(0.9),
        };
        let axis = parse_axis_def(&Some(def), "y").unwrap().unwrap();
        match axis.style {
            DeformStyle::Final {
                target_lo,
                target_hi,
            } => {
                assert!((target_lo - 0.1).abs() < 1e-15);
                assert!((target_hi - 0.9).abs() < 1e-15);
            }
            _ => panic!("Expected Final style"),
        }
    }

    #[test]
    fn none_axis_returns_none() {
        assert!(parse_axis_def(&None, "x").unwrap().is_none());
    }

    #[test]
    fn erate_missing_rate_reports_validation_error() {
        let def = AxisDeformDef {
            style: "erate".to_string(),
            rate: None,
            velocity: None,
            lo: None,
            hi: None,
        };
        let err = parse_axis_def(&Some(def), "z").unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] z: style 'erate' requires a 'rate' field"
        );
    }

    #[test]
    fn vel_missing_velocity_reports_validation_error() {
        let def = AxisDeformDef {
            style: "vel".to_string(),
            rate: None,
            velocity: None,
            lo: None,
            hi: None,
        };
        let err = parse_axis_def(&Some(def), "x").unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] x: style 'vel' requires a 'velocity' field"
        );
    }

    #[test]
    fn final_missing_bounds_report_validation_errors() {
        let missing_lo = AxisDeformDef {
            style: "final".to_string(),
            rate: None,
            velocity: None,
            lo: None,
            hi: Some(1.0),
        };
        let err = parse_axis_def(&Some(missing_lo), "y").unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] y: style 'final' requires a 'lo' field"
        );

        let missing_hi = AxisDeformDef {
            style: "final".to_string(),
            rate: None,
            velocity: None,
            lo: Some(0.0),
            hi: None,
        };
        let err = parse_axis_def(&Some(missing_hi), "y").unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] y: style 'final' requires a 'hi' field"
        );
    }

    #[test]
    fn unknown_axis_style_reports_validation_error() {
        let def = AxisDeformDef {
            style: "wiggle".to_string(),
            rate: None,
            velocity: None,
            lo: None,
            hi: None,
        };
        let err = parse_axis_def(&Some(def), "x").unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] x: unknown style 'wiggle'. Use 'erate', 'vel', or 'final'."
        );
    }

    #[test]
    fn xy_unsupported_style_reports_validation_error() {
        let def = AxisDeformDef {
            style: "vel".to_string(),
            rate: None,
            velocity: Some(0.5),
            lo: None,
            hi: None,
        };
        let err = parse_shear_rate(&Some(def)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] xy: unsupported style 'vel'. Lees-Edwards shear only supports style 'erate'."
        );
    }

    #[test]
    fn xy_erate_missing_rate_reports_validation_error() {
        let def = AxisDeformDef {
            style: "erate".to_string(),
            rate: None,
            velocity: None,
            lo: None,
            hi: None,
        };
        let err = parse_shear_rate(&Some(def)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] xy: style 'erate' requires a 'rate' field (shear-strain rate)"
        );
    }

    #[test]
    fn xy_non_periodic_boundaries_report_validation_error() {
        let config = DeformConfig {
            xy: Some(AxisDeformDef {
                style: "erate".to_string(),
                rate: Some(0.5),
                velocity: None,
                lo: None,
                hi: None,
            }),
            ..DeformConfig::default()
        };
        let mut domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        domain.boundary_type = [
            BoundaryType::Periodic,
            BoundaryType::Fixed,
            BoundaryType::Periodic,
        ];

        let err = validate_deform_config(&config, &domain, 1).unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] xy shear requires periodic x and y boundaries; got x=Periodic, y=Fixed"
        );
    }

    #[test]
    fn active_deform_with_mpi_reports_subdomain_validation_error() {
        let config = DeformConfig {
            z: Some(AxisDeformDef {
                style: "erate".to_string(),
                rate: Some(-0.001),
                velocity: None,
                lo: None,
                hi: None,
            }),
            ..DeformConfig::default()
        };
        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        let err = validate_deform_config(&config, &domain, 2).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("soil_deform"));
        assert!(message.contains("MPI rank count 2"));
        assert!(message.contains("unsupported subdomain-bound update"));
        assert!(message.contains("sub_domain_low/sub_domain_high"));
    }

    #[test]
    fn no_deform_with_mpi_is_accepted() {
        let config = DeformConfig::default();
        let domain = make_domain([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        let (axes, shear_xy_rate) = validate_deform_config(&config, &domain, 4).unwrap();
        assert!(axes.iter().all(|axis| axis.is_none()));
        assert!(shear_xy_rate.is_none());
    }

    #[test]
    fn erate_box_collapse_reports_validation_error() {
        let axis = AxisDeform {
            style: DeformStyle::Erate { rate: -1.0 },
            lo_0: 0.0,
            hi_0: 1.0,
        };
        let err = next_axis_bounds(&axis, "z", 1.0, 1, 1).unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] z: box would collapse at step 1 (lo=0.5, hi=0.5); reduce compression rate/velocity or choose final bounds with hi > lo"
        );
    }

    #[test]
    fn vel_box_collapse_reports_validation_error() {
        let axis = AxisDeform {
            style: DeformStyle::Vel { velocity: -1.0 },
            lo_0: 0.0,
            hi_0: 1.0,
        };
        let err = next_axis_bounds(&axis, "x", 1.0, 1, 1).unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] x: box would collapse at step 1 (lo=0.5, hi=0.5); reduce compression rate/velocity or choose final bounds with hi > lo"
        );
    }

    #[test]
    fn final_box_collapse_reports_validation_error() {
        let axis = AxisDeform {
            style: DeformStyle::Final {
                target_lo: 1.0,
                target_hi: 0.0,
            },
            lo_0: 0.0,
            hi_0: 1.0,
        };
        let err = next_axis_bounds(&axis, "y", 1.0, 1, 1).unwrap_err();
        assert_eq!(
            err.to_string(),
            "[deform] y: box would collapse at step 1 (lo=1, hi=0); reduce compression rate/velocity or choose final bounds with hi > lo"
        );
    }
}
