//! Atom groups: named subsets selected by type and/or spatial region.

use std::{error::Error, fmt};

use grass_app::prelude::*;
use grass_scheduler::prelude::*;
use serde::Deserialize;

use crate::{Atom, Config, ParticleSimScheduleSet, Region, ScheduleSetupSet, StageOverrides};

// ── Config ──────────────────────────────────────────────────────────────────

#[derive(Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
/// Definition of a single atom group from TOML `[[group]]`.
pub struct GroupDef {
    /// Group name (used to reference the group elsewhere in the config).
    pub name: String,
    /// Atom types included in this group; `None` = all types (subject to `region`).
    #[serde(rename = "type", default)]
    pub atom_types: Option<Vec<u32>>,
    /// Spatial region filter (AND-combined with type filter if both present).
    #[serde(default)]
    pub region: Option<Region>,
    /// Whether the group re-evaluates membership every timestep.
    /// Defaults to `true` if a region is set, `false` otherwise.
    #[serde(default)]
    pub dynamic: Option<bool>,
}

impl GroupDef {
    /// Returns whether this group should be rebuilt every timestep.
    /// Smart default: dynamic if a region is set, static otherwise.
    pub fn is_dynamic(&self) -> bool {
        self.dynamic.unwrap_or(self.region.is_some())
    }
}

// ── Group ───────────────────────────────────────────────────────────────────

/// A named group with a boolean mask over local atoms.
#[derive(Debug)]
pub struct Group {
    /// Group name (matches [`GroupDef::name`]).
    pub name: String,
    /// The definition (type/region filters) this group was built from.
    pub def: GroupDef,
    /// Boolean membership mask over local atoms; `mask[i]` is `true` if atom `i` is in the group.
    pub mask: Vec<bool>,
    /// Number of local atoms in the group (count of `true` entries in `mask`).
    pub count: usize,
}

// ── GroupRegistry ───────────────────────────────────────────────────────────

/// Registry of all atom groups. Always contains a built-in "all" group.
pub struct GroupRegistry {
    /// All registered groups, including the built-in "all" group.
    pub groups: Vec<Group>,
}

/// Error returned when a named group lookup fails.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupLookupError {
    /// Requested group name.
    pub name: String,
    /// Group names present in the registry when the lookup was attempted.
    pub available: Vec<String>,
}

impl GroupLookupError {
    fn new(name: &str, available: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            available,
        }
    }
}

impl fmt::Display for GroupLookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "group '{}' not found", self.name)?;
        if self.available.is_empty() {
            write!(f, "; no groups are currently registered")
        } else {
            write!(f, "; available groups: {}", self.available.join(", "))
        }
    }
}

impl Error for GroupLookupError {}

impl GroupRegistry {
    /// Create an empty registry (the "all" group is added during setup).
    pub fn new() -> Self {
        GroupRegistry { groups: Vec::new() }
    }

    /// Look up a group by name.
    pub fn get(&self, name: &str) -> Option<&Group> {
        self.groups.iter().find(|g| g.name == name)
    }

    /// Return the names of all registered groups.
    pub fn available_group_names(&self) -> Vec<String> {
        self.groups.iter().map(|g| g.name.clone()).collect()
    }

    /// Look up a group by name, returning available-group diagnostics if it is missing.
    pub fn try_get(&self, name: &str) -> Result<&Group, GroupLookupError> {
        self.get(name)
            .ok_or_else(|| GroupLookupError::new(name, self.available_group_names()))
    }

    /// Look up a group by name, panicking with a helpful message if not found.
    pub fn expect(&self, name: &str) -> &Group {
        self.try_get(name).unwrap_or_else(|err| panic!("{err}"))
    }

    /// Validate that a group name exists with contextual diagnostics.
    pub fn validate_name(&self, name: &str, context: &str) -> Result<(), GroupLookupError> {
        self.try_get(name)
            .map(|_| ())
            .map_err(|error| GroupLookupError {
                name: format!("{context}: {error}"),
                available: Vec::new(),
            })
    }

    /// Return the mask for a named group, or `None` if no group is specified (meaning "all atoms").
    pub fn mask_for(
        &self,
        group_name: &Option<String>,
    ) -> Result<Option<&[bool]>, GroupLookupError> {
        group_name
            .as_ref()
            .map(|gname| self.try_get(gname).map(|group| group.mask.as_slice()))
            .transpose()
    }
}

impl Default for GroupRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns `true` if atom `i` passes the group filter (or if no mask is active).
#[inline(always)]
pub fn group_includes(mask: Option<&[bool]>, i: usize) -> bool {
    match mask {
        Some(m) => m[i],
        None => true,
    }
}

// ── Membership evaluation ───────────────────────────────────────────────────

/// Evaluate whether atom `i` matches a group definition (AND-combine all criteria).
fn evaluate_membership(def: &GroupDef, atoms: &Atom, i: usize) -> bool {
    if let Some(ref types) = def.atom_types {
        if !types.contains(&atoms.atom_type[i]) {
            return false;
        }
    }
    if let Some(ref r) = def.region {
        // `pos` is `Real` (f32 in mixed/single); regions are defined in f64.
        let p = atoms.pos[i];
        if !r.contains(&[p[0] as f64, p[1] as f64, p[2] as f64]) {
            return false;
        }
    }
    true
}

fn rebuild_group_masks(groups: &mut GroupRegistry, atoms: &Atom) {
    let nlocal = atoms.nlocal as usize;
    for group in groups.groups.iter_mut() {
        // Skip static groups that are already initialized
        if !group.def.is_dynamic() && !group.mask.is_empty() {
            continue;
        }
        group.mask.clear();
        group.mask.resize(nlocal, false);
        group.count = 0;
        if group.name == "all" {
            for m in group.mask.iter_mut() {
                *m = true;
            }
            group.count = nlocal;
        } else {
            for i in 0..nlocal {
                if evaluate_membership(&group.def, atoms, i) {
                    group.mask[i] = true;
                    group.count += 1;
                }
            }
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────────────

/// Registers group setup and per-step rebuild systems.
pub struct GroupPlugin;

impl Plugin for GroupPlugin {
    fn default_config(&self) -> Option<&str> {
        Some(
            r#"# Atom groups — named subsets for selective operations.
# [[group]]
# name = "mobile"
# type = [1, 2]                                              # optional: match atom_type
# region = { type = "block", min = [0, 0, 0], max = [5, 5, 5] }  # optional: spatial region
# dynamic = false                                            # optional: lock membership at setup (default: true if region set, false otherwise)"#,
        )
    }

    fn build(&self, app: &mut App) {
        self.try_build(app)
            .unwrap_or_else(|error| panic!("invalid [[group]] configuration: {error}"));
    }

    fn try_build(&self, app: &mut App) -> Result<(), AppError> {
        if let Some(config) = app.get_resource_ref::<Config>() {
            config
                .try_parse_array::<GroupDef>("group")
                .map_err(|error| AppError::message(error.to_string()))?;
        }
        app.add_resource(GroupRegistry::new())
            .add_setup_system(setup_groups, ScheduleSetupSet::PostSetup)
            .add_update_system(rebuild_groups, ParticleSimScheduleSet::PreForce);
        Ok(())
    }
}

// ── Systems ─────────────────────────────────────────────────────────────────

/// Setup system: build the [`GroupRegistry`] from the config's `[[group]]`
/// definitions (plus the built-in "all" group) and populate initial masks.
pub fn setup_groups(
    config: Res<Config>,
    atoms: Res<Atom>,
    comm: Res<crate::CommResource>,
    mut groups: ResMut<GroupRegistry>,
    stage_overrides: Res<StageOverrides>,
    scheduler_manager: Res<SchedulerManager>,
) {
    let index = scheduler_manager.index;

    if index == 0 {
        // First stage: parse global [[group]] config
        let defs = config.parse_array::<GroupDef>("group");

        // Always start with the built-in "all" group.
        groups.groups.clear();
        groups.groups.push(Group {
            name: "all".to_string(),
            def: GroupDef {
                name: "all".to_string(),
                atom_types: None,
                region: None,
                dynamic: None,
            },
            mask: Vec::new(),
            count: 0,
        });

        for def in defs {
            if def.name == "all" {
                if comm.rank() == 0 {
                    eprintln!("WARNING: Cannot redefine built-in group 'all', skipping.");
                }
                continue;
            }
            if comm.rank() == 0 {
                println!("Group '{}': {:?}", def.name, def);
            }
            groups.groups.push(Group {
                name: def.name.clone(),
                def,
                mask: Vec::new(),
                count: 0,
            });
        }
    } else if stage_overrides.table.contains_key("group") {
        // Later stage with group overrides: merge/overlay by name
        let stage_defs: Vec<GroupDef> = stage_overrides.section("group");
        for def in stage_defs {
            if def.name == "all" {
                if comm.rank() == 0 {
                    eprintln!("WARNING: Cannot redefine built-in group 'all', skipping.");
                }
                continue;
            }
            if let Some(existing) = groups.groups.iter_mut().find(|g| g.name == def.name) {
                // Replace definition, clear mask to force rebuild
                if comm.rank() == 0 {
                    println!(
                        "Stage {}: redefining group '{}': {:?}",
                        index, def.name, def
                    );
                }
                existing.def = def;
                existing.mask.clear();
            } else {
                // Add new group
                if comm.rank() == 0 {
                    println!("Stage {}: adding group '{}': {:?}", index, def.name, def);
                }
                groups.groups.push(Group {
                    name: def.name.clone(),
                    def,
                    mask: Vec::new(),
                    count: 0,
                });
            }
        }
    }
    // else: no group overrides in this stage, keep existing groups as-is

    rebuild_group_masks(&mut groups, &atoms);
}

/// Per-step system: recompute every dynamic group's membership mask against the
/// current atom positions/types (static groups are left unchanged).
pub fn rebuild_groups(atoms: Res<Atom>, mut groups: ResMut<GroupRegistry>) {
    rebuild_group_masks(&mut groups, &atoms);
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_atom(positions: &[(f64, f64, f64)], types: &[u32]) -> Atom {
        let mut atom = Atom::new();
        for (i, (px, py, pz)) in positions.iter().enumerate() {
            atom.push_test_atom(i as u32, [*px, *py, *pz], 0.5, 1.0);
            atom.atom_type[i] = types[i];
        }
        atom.nlocal = positions.len() as u32;
        atom
    }

    fn make_group_def(name: &str, types: Option<Vec<u32>>, region: Option<Region>) -> GroupDef {
        GroupDef {
            name: name.to_string(),
            atom_types: types,
            region,
            dynamic: None,
        }
    }

    #[test]
    fn test_all_group_always_exists() {
        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "all".to_string(),
            def: make_group_def("all", None, None),
            mask: Vec::new(),
            count: 0,
        });
        let atom = make_atom(&[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)], &[0, 1]);
        rebuild_group_masks(&mut registry, &atom);
        let all = registry.expect("all");
        assert_eq!(all.count, 2);
        assert!(all.mask.iter().all(|&m| m));
    }

    #[test]
    fn test_type_filter() {
        let atom = make_atom(
            &[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
            &[0, 1, 0],
        );
        let def = make_group_def("type0", Some(vec![0]), None);
        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "type0".to_string(),
            def,
            mask: Vec::new(),
            count: 0,
        });
        rebuild_group_masks(&mut registry, &atom);
        let g = registry.expect("type0");
        assert_eq!(g.count, 2);
        assert_eq!(g.mask, vec![true, false, true]);
    }

    #[test]
    fn test_region_filter() {
        let atom = make_atom(
            &[(0.0, 0.0, 1.0), (0.0, 0.0, 3.0), (0.0, 0.0, 5.0)],
            &[0, 0, 0],
        );
        let def = make_group_def(
            "bottom",
            None,
            Some(Region::Block {
                min: [f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0],
                max: [f64::INFINITY, f64::INFINITY, 4.0],
            }),
        );
        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "bottom".to_string(),
            def,
            mask: Vec::new(),
            count: 0,
        });
        rebuild_group_masks(&mut registry, &atom);
        let g = registry.expect("bottom");
        assert_eq!(g.count, 2);
        assert_eq!(g.mask, vec![true, true, false]);
    }

    #[test]
    fn test_combined_type_and_region() {
        let atom = make_atom(
            &[(0.0, 0.0, 1.0), (0.0, 0.0, 3.0), (0.0, 0.0, 5.0)],
            &[0, 1, 0],
        );
        let def = make_group_def(
            "combo",
            Some(vec![0]),
            Some(Region::Block {
                min: [f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0],
                max: [f64::INFINITY, f64::INFINITY, 4.0],
            }),
        );
        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "combo".to_string(),
            def,
            mask: Vec::new(),
            count: 0,
        });
        rebuild_group_masks(&mut registry, &atom);
        let g = registry.expect("combo");
        assert_eq!(g.count, 1); // only atom 0 (type 0, z=1.0)
        assert_eq!(g.mask, vec![true, false, false]);
    }

    #[test]
    fn test_empty_group() {
        let atom = make_atom(&[(0.0, 0.0, 1.0), (0.0, 0.0, 3.0)], &[0, 0]);
        let def = make_group_def("empty", Some(vec![99]), None);
        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "empty".to_string(),
            def,
            mask: Vec::new(),
            count: 0,
        });
        rebuild_group_masks(&mut registry, &atom);
        let g = registry.expect("empty");
        assert_eq!(g.count, 0);
        assert!(g.mask.iter().all(|&m| !m));
    }

    #[test]
    fn missing_group_lookup_reports_available_groups() {
        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "mobile".to_string(),
            def: make_group_def("mobile", None, None),
            mask: vec![true, false],
            count: 1,
        });
        registry.groups.push(Group {
            name: "wall".to_string(),
            def: make_group_def("wall", None, None),
            mask: vec![false, true],
            count: 1,
        });

        let err = registry.try_get("moblie").unwrap_err();
        assert_eq!(err.name, "moblie");
        assert_eq!(
            err.available,
            vec!["mobile".to_string(), "wall".to_string()]
        );
        assert_eq!(
            err.to_string(),
            "group 'moblie' not found; available groups: mobile, wall"
        );
    }

    #[test]
    fn mask_for_returns_error_for_misspelled_group() {
        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "mobile".to_string(),
            def: make_group_def("mobile", None, None),
            mask: vec![true, false],
            count: 1,
        });

        assert_eq!(registry.mask_for(&None).unwrap(), None);
        assert_eq!(
            registry.mask_for(&Some("mobile".to_string())).unwrap(),
            Some(&[true, false][..])
        );

        let err = registry.mask_for(&Some("moblie".to_string())).unwrap_err();
        assert_eq!(
            err.to_string(),
            "group 'moblie' not found; available groups: mobile"
        );
    }

    #[test]
    fn test_is_dynamic_defaults() {
        // Type-only group: static by default
        let def = make_group_def("type_only", Some(vec![1]), None);
        assert!(!def.is_dynamic());

        // Region group: dynamic by default
        let def = make_group_def(
            "region",
            None,
            Some(Region::Block {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            }),
        );
        assert!(def.is_dynamic());

        // No filters (like "all"): static by default
        let def = make_group_def("all", None, None);
        assert!(!def.is_dynamic());
    }

    #[test]
    fn test_is_dynamic_explicit() {
        // Explicitly dynamic type-only group
        let mut def = make_group_def("type_dyn", Some(vec![1]), None);
        def.dynamic = Some(true);
        assert!(def.is_dynamic());

        // Explicitly static region group
        let mut def = make_group_def(
            "region_static",
            None,
            Some(Region::Block {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            }),
        );
        def.dynamic = Some(false);
        assert!(!def.is_dynamic());
    }

    #[test]
    fn test_static_group_skips_rebuild() {
        let atom = make_atom(
            &[(0.0, 0.0, 1.0), (0.0, 0.0, 3.0), (0.0, 0.0, 5.0)],
            &[0, 1, 0],
        );

        // Create a static type group (dynamic = None, no region → static)
        let def = make_group_def("type0", Some(vec![0]), None);
        assert!(!def.is_dynamic());

        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "type0".to_string(),
            def,
            mask: Vec::new(),
            count: 0,
        });

        // First rebuild: mask is empty, so it gets built
        rebuild_group_masks(&mut registry, &atom);
        assert_eq!(registry.expect("type0").count, 2);
        assert_eq!(registry.expect("type0").mask, vec![true, false, true]);

        // Manually change the mask to verify it's NOT rebuilt
        registry.groups[0].mask = vec![false, false, false];
        registry.groups[0].count = 0;

        // Second rebuild: mask is non-empty and group is static → skipped
        rebuild_group_masks(&mut registry, &atom);
        assert_eq!(registry.groups[0].mask, vec![false, false, false]);
        assert_eq!(registry.groups[0].count, 0);
    }

    #[test]
    fn test_dynamic_group_rebuilds_every_time() {
        let atom = make_atom(
            &[(0.0, 0.0, 1.0), (0.0, 0.0, 3.0), (0.0, 0.0, 5.0)],
            &[0, 0, 0],
        );

        // Create a dynamic region group
        let def = make_group_def(
            "bottom",
            None,
            Some(Region::Block {
                min: [f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0],
                max: [f64::INFINITY, f64::INFINITY, 4.0],
            }),
        );
        assert!(def.is_dynamic());

        let mut registry = GroupRegistry::new();
        registry.groups.push(Group {
            name: "bottom".to_string(),
            def,
            mask: Vec::new(),
            count: 0,
        });

        // First rebuild
        rebuild_group_masks(&mut registry, &atom);
        assert_eq!(registry.expect("bottom").count, 2);

        // Manually change mask
        registry.groups[0].mask = vec![false, false, false];
        registry.groups[0].count = 0;

        // Second rebuild: dynamic group gets rebuilt
        rebuild_group_masks(&mut registry, &atom);
        assert_eq!(registry.expect("bottom").count, 2);
        assert_eq!(registry.expect("bottom").mask, vec![true, true, false]);
    }

    #[test]
    fn test_dynamic_field_deserializes() {
        let toml_str = r#"
[[group]]
name = "static_type"
type = [1]

[[group]]
name = "dynamic_region"
region = { type = "block", min = [0, 0, 0], max = [1, 1, 1] }

[[group]]
name = "explicit_static_region"
region = { type = "block", min = [0, 0, 0], max = [1, 1, 1] }
dynamic = false

[[group]]
name = "explicit_dynamic_type"
type = [2]
dynamic = true
"#;
        let table: toml::Table = toml_str.parse().unwrap();
        let defs: Vec<GroupDef> = table.get("group").unwrap().clone().try_into().unwrap();

        assert_eq!(defs.len(), 4);
        assert!(!defs[0].is_dynamic()); // type-only, no explicit → false
        assert!(defs[1].is_dynamic()); // region, no explicit → true
        assert!(!defs[2].is_dynamic()); // region, explicit false
        assert!(defs[3].is_dynamic()); // type, explicit true
    }
}
