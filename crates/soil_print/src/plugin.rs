use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use soil_core::{stage_tables, Config, ParticleSimScheduleSet, ScheduleSetupSet};

use crate::thermo::parse_thermo_column;
use crate::{
    check_stage_end_save,
    dump::{write_dump_binary, write_dump_lammps, write_dump_text},
    dump_atoms, output_virial_to_thermo, preflight_restart_input, print_thermo, print_vtp,
    read_restart, setup_thermo, sync_dump_box, write_restart, DumpConfig, DumpRegistry,
    RestartConfig, Thermo, ThermoConfig, VtpConfig,
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
        Config::load::<DumpConfig>(app, "dump");
        Config::load::<RestartConfig>(app, "restart");
        Config::load::<VtpConfig>(app, "vtp");
        Config::load::<ThermoConfig>(app, "thermo");
        install_print(app);
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
        let thermo = Config::try_load::<ThermoConfig>(app, "thermo")
            .map_err(|e| AppError::message(e.to_string()))?;

        if let Some(config) = app.get_resource_ref::<Config>() {
            let groups = stage_tables(&config)
                .map_err(|error| AppError::message(error.to_string()))?
                .into_iter()
                .map(Config::from_table)
                .map(|stage| stage.try_parse_array::<soil_core::GroupDef>("group"))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| AppError::message(error.to_string()))?;
            for column in thermo.columns.as_deref().unwrap_or(&[]) {
                let parsed = parse_thermo_column(column);
                if let Some(group) = parsed.group_name {
                    if group != "all"
                        && !groups
                            .iter()
                            .any(|stage| stage.iter().any(|candidate| candidate.name == group))
                    {
                        return Err(AppError::message(format!(
                            "thermo column '{}': group '{}' is not declared in [[group]]",
                            column, group
                        )));
                    }
                }
            }
        }

        install_print(app);
        Ok(())
    }
}

fn install_print(app: &mut App) {
    // Built-in dump formats, registered through the same public extension
    // point (`register_format`) any plugin uses — core just ships three.
    let dump_reg = DumpRegistry::new();
    dump_reg.register_format("text", write_dump_text);
    dump_reg.register_format("binary", write_dump_binary);
    dump_reg.register_format("lammps", write_dump_lammps);
    app.add_resource(dump_reg);

    // File corruption is ordinary user input, so it must reach `try_start`
    // as AppError before any legacy scheduler setup or update can run.
    app.add_fallible_setup_system(
        "validate restart input",
        |app: &mut App| {
            let restart = app
                .get_resource_ref::<RestartConfig>()
                .map(|config| (*config).clone())
                .unwrap_or_default();
            let Some(comm) = app.get_resource_ref::<soil_core::CommResource>() else {
                return Ok(());
            };
            let Some(input) = app.get_resource_ref::<soil_core::Input>() else {
                return Ok(());
            };
            preflight_restart_input(&restart, &**comm, &input).map_err(AppError::message)
        },
        ScheduleSetupSet::PostSetup,
    );

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
}

#[cfg(test)]
mod fallible_tests {
    use super::*;

    #[test]
    fn invalid_restart_format_is_returned_by_fallible_plugin_registration() {
        let mut app = App::new();
        app.add_resource(Config::from_str("[restart]\nformat = \"yaml\"\n"));

        let error = match app.try_add_plugins(PrintPlugin) {
            Ok(_) => panic!("invalid restart format unexpectedly registered"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("[restart] format"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn unknown_thermo_group_is_returned_by_fallible_plugin_registration() {
        let mut app = App::new();
        app.add_resource(Config::from_str("[thermo]\ncolumns = ['ke/missing']\n"));
        let error = match app.try_add_plugins(PrintPlugin) {
            Ok(_) => panic!("unknown thermo group unexpectedly registered"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains("group 'missing'"),
            "unexpected error: {error}"
        );
    }
}

// ── Helper: restart directory ───────────────────────────────────────────────
