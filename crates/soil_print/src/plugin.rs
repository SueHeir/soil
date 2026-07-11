use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use soil_core::{Config, ParticleSimScheduleSet, ScheduleSetupSet};

use crate::{
    check_stage_end_save,
    dump::{write_dump_binary, write_dump_lammps, write_dump_text},
    dump_atoms, output_virial_to_thermo, print_thermo, print_vtp, read_restart, setup_thermo,
    sync_dump_box, write_restart, DumpConfig, DumpRegistry, RestartConfig, Thermo, ThermoConfig,
    VtpConfig,
};

/// Main output plugin — registers thermo, dump, restart, and VTP systems.
///
/// Add this plugin to the app to enable all output subsystems. Each subsystem
/// is independently configured via its TOML section and only produces output
/// when its interval is non-zero.
pub struct PrintPlugin;

impl Plugin for PrintPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"[dump]
# Dump output interval (0 = disabled)
interval = 0
# Dump format: "text" (CSV), "binary", or "lammps" (.lammpstrj, OVITO-native)
format = "text"
# false: MPI-gather all ranks into one file/step; true: one file per rank
per_rank = false
# include ghost (halo) atoms, flagged with an IsGhost column
ghost = false

[restart]
# Restart file write interval (0 = disabled)
interval = 0
# Restart format: "bincode" or "json"
format = "bincode"
# Whether to read restart files at startup
read = false

[vtp]
# VTP (ParaView) output interval (0 = disabled)
interval = 0"#,
        )
    }

    fn build(&self, app: &mut App) {
        self.try_build(app)
            .unwrap_or_else(|error| panic!("invalid output configuration: {error}"));
    }

    fn try_build(&self, app: &mut App) -> Result<(), AppError> {
        Config::try_load::<DumpConfig>(app, "dump")
            .map_err(|e| AppError::message(e.to_string()))?;
        let restart = Config::try_load::<RestartConfig>(app, "restart")
            .map_err(|e| AppError::message(e.to_string()))?;
        if restart.format != "bincode" && restart.format != "json" {
            return Err(AppError::message(format!(
                "[restart] format must be 'bincode' or 'json', got '{}'",
                restart.format
            )));
        }
        Config::try_load::<VtpConfig>(app, "vtp").map_err(|e| AppError::message(e.to_string()))?;
        Config::try_load::<ThermoConfig>(app, "thermo")
            .map_err(|e| AppError::message(e.to_string()))?;

        // Built-in dump formats, registered through the same public extension
        // point (`register_format`) any plugin uses — core just ships three.
        let dump_reg = DumpRegistry::new();
        dump_reg.register_format("text", write_dump_text);
        dump_reg.register_format("binary", write_dump_binary);
        dump_reg.register_format("lammps", write_dump_lammps);
        app.add_resource(dump_reg);

        app.add_resource(Thermo::new())
            .add_setup_system(setup_thermo, ScheduleSetupSet::PostSetup)
            .add_setup_system(
                read_restart.run_if(first_stage_only()),
                ScheduleSetupSet::PostSetup,
            )
            .add_setup_system(sync_dump_box, ScheduleSetupSet::PostSetup)
            .add_update_system(output_virial_to_thermo, ParticleSimScheduleSet::PostForce)
            .add_update_system(sync_dump_box, ParticleSimScheduleSet::PostForce)
            .add_update_system(print_vtp, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(print_thermo, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(dump_atoms, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(write_restart, ParticleSimScheduleSet::PostFinalIntegration)
            .add_update_system(
                check_stage_end_save.before("update_cycle"),
                ParticleSimScheduleSet::PostFinalIntegration,
            );
        Ok(())
    }
}

// ── Helper: restart directory ───────────────────────────────────────────────
