use crate::{
    check_stage_end_save,
    dump::{write_dump_binary, write_dump_lammps, write_dump_text},
    dump_atoms, output_virial_to_thermo, preflight_restart_input, print_thermo, print_vtp,
    read_restart, setup_thermo, sync_dump_box, write_restart, DumpConfig, DumpRegistry,
    RestartConfig, Thermo, ThermoConfig, VtpConfig,
};
use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use soil_core::{Config, ParticleSimScheduleSet, ScheduleSetupSet, UPDATE_CYCLE};

/// Main output plugin — registers thermo, dump, restart, and VTP systems.
///
/// Add this plugin to the app to enable all output subsystems. Each subsystem
/// is independently configured via its TOML section and only produces output
/// when its interval is non-zero.
pub struct PrintPlugin;

impl Plugin for PrintPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(crate::config::DEFAULT_CONFIG)
    }

    fn build(&self, app: &mut App) {
        Config::load::<DumpConfig>(app, "dump");
        Config::load::<RestartConfig>(app, "restart");
        Config::load::<VtpConfig>(app, "vtp");
        Config::load::<ThermoConfig>(app, "thermo");
        install_print(app);
    }

    fn try_build(&self, app: &mut App) -> Result<(), AppError> {
        crate::config::load_and_validate(app)?;
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
            check_stage_end_save.before(UPDATE_CYCLE),
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
