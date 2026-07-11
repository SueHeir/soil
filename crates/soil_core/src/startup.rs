//! Fallible startup configuration helpers.

use crate::{deep_merge, Config};
use grass_io::ConfigError;
use serde::Deserialize;

#[derive(Deserialize)]
struct RunStage {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    steps: Option<u32>,
    #[serde(default)]
    dt: Option<f64>,
    #[serde(default)]
    skip: Option<bool>,
    #[serde(default)]
    save_at_end: Option<bool>,
    #[serde(flatten)]
    overrides: toml::Table,
}

/// Build the effective configuration table for each declared run stage.
pub fn stage_tables(config: &Config) -> Result<Vec<toml::Table>, ConfigError> {
    let mut global = config.table.clone();
    let stages = match global.remove("run") {
        None => return Ok(vec![global]),
        Some(toml::Value::Table(table)) => vec![parse_stage(table, None)?],
        Some(toml::Value::Array(values)) => values
            .into_iter()
            .enumerate()
            .map(|(index, value)| match value {
                toml::Value::Table(table) => parse_stage(table, Some(index)),
                value => Err(ConfigError::InvalidArray {
                    key: "run".into(),
                    index: Some(index),
                    message: format!("expected table, got {}", value.type_str()),
                }),
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(value) => {
            return Err(ConfigError::InvalidArray {
                key: "run".into(),
                index: None,
                message: format!("expected [run] or [[run]], got {}", value.type_str()),
            })
        }
    };
    Ok(stages
        .into_iter()
        .map(|stage| {
            let mut table = global.clone();
            deep_merge(&mut table, &stage.overrides);
            table
        })
        .collect())
}

fn parse_stage(table: toml::Table, index: Option<usize>) -> Result<RunStage, ConfigError> {
    table.try_into().map_err(|error| ConfigError::InvalidArray {
        key: "run".into(),
        index,
        message: error.to_string(),
    })
}
