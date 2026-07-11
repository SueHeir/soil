//! Executable compatibility matrix for SOIL's exported typed scheduler labels.

use std::{env, fs, panic};

use grass_scheduler::prelude::*;
use serde::Deserialize;
use soil_core::{ParticleSimScheduleSet, PBC, SORT_ATOMS};

#[derive(Deserialize)]
struct Config {
    case: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    target: String,
    provider: bool,
    expected: String,
}

fn provider() {}
fn downstream() {}
fn borders() {}

fn organizes(case: &Case) -> bool {
    panic::catch_unwind(|| {
        let mut scheduler = Scheduler::default();
        match case.target.as_str() {
            "pbc" => {
                if case.provider {
                    scheduler.add_update_system(
                        provider.label(PBC),
                        ParticleSimScheduleSet::PreExchange,
                    );
                }
                scheduler.add_update_system(
                    downstream.requires(PBC),
                    ParticleSimScheduleSet::PreExchange,
                );
            }
            "sort_atoms" => {
                if case.provider {
                    scheduler.add_update_system(
                        provider.label(SORT_ATOMS),
                        ParticleSimScheduleSet::PreNeighbor,
                    );
                }
                // This is the same required sort -> borders seam registered by
                // CommunicationPlugin, expressed here as a minimal matrix case.
                scheduler.add_update_system(
                    borders.label(soil_core::BORDERS).requires(SORT_ATOMS),
                    ParticleSimScheduleSet::PreNeighbor,
                );
            }
            target => panic!("unknown typed-label target `{target}`"),
        }
        scheduler.organize_systems();
    })
    .is_ok()
}

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/typed_system_labels/config.toml".to_owned());
    let config: Config = toml::from_str(&fs::read_to_string(&path).expect("read config"))
        .expect("parse typed-label config");

    let mut passed = 0;
    for case in &config.case {
        let observed = if organizes(case) { "accept" } else { "reject" };
        let ok = observed == case.expected;
        passed += usize::from(ok);
        println!(
            "CASE {} expected={} observed={} status={}",
            case.name,
            case.expected,
            observed,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    println!("RESULT {passed}/{} contracts passed", config.case.len());
    assert_eq!(passed, config.case.len(), "typed-label contract mismatch");
}
