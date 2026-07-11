//! Configuration defaults and validation for [`crate::PrintPlugin`].

use grass_app::prelude::{App, AppError};
use soil_core::{stage_tables, Config};

use crate::{thermo::parse_thermo_column, DumpConfig, RestartConfig, ThermoConfig, VtpConfig};

/// Default declarative configuration installed by the print plugin.
pub(crate) const DEFAULT_CONFIG: &str = r#"[dump]
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
interval = 0"#;

/// Load and validate every output section before scheduling output systems.
pub(crate) fn load_and_validate(app: &mut App) -> Result<(), AppError> {
    Config::try_load::<DumpConfig>(app, "dump").map_err(|e| AppError::message(e.to_string()))?;
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
    Ok(())
}
