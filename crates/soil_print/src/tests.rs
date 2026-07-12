use super::*;
use crate::restart::{write_restart_inner, RestartData};
use crate::thermo::{capitalize, default_columns, parse_thermo_column, print_thermo, setup_thermo};
use grass_scheduler::Scheduler;
use soil_core::{
    CommResource, Neighbor, ParticleSimScheduleSet, RunConfig, RunState, ScheduleSetupSet,
    SingleProcessComm,
};
use std::fs::{self, File};

#[test]
fn test_default_columns_backward_compat() {
    let config = ThermoConfig::default();
    assert!(config.columns.is_none());
    let cols = config.columns.unwrap_or_else(default_columns);
    assert_eq!(
        cols,
        vec!["step", "atoms", "ke", "neighbors", "walltime", "stepps"]
    );
}

#[test]
fn test_column_parsing() {
    let col = parse_thermo_column("temp/mobile");
    assert_eq!(col.compute_name, "temp");
    assert_eq!(col.group_name.as_deref(), Some("mobile"));
    assert_eq!(col.header, "Temp/mobile");

    let col2 = parse_thermo_column("step");
    assert_eq!(col2.compute_name, "step");
    assert!(col2.group_name.is_none());
    assert_eq!(col2.header, "Step");
}

#[test]
fn test_user_value_set_and_read() {
    let mut thermo = Thermo::new();
    assert!(thermo.values.get("pe").is_none());
    thermo.set("pe", 42.0);
    assert_eq!(*thermo.values.get("pe").unwrap(), 42.0);
    thermo.set("pe", 99.0);
    assert_eq!(*thermo.values.get("pe").unwrap(), 99.0);
}

#[test]
fn test_capitalize() {
    assert_eq!(capitalize("step"), "Step");
    assert_eq!(capitalize("ke"), "Ke");
    assert_eq!(capitalize(""), "");
    assert_eq!(capitalize("a"), "A");
}

#[test]
fn test_dump_registry_has_callbacks() {
    let reg = DumpRegistry::new();
    assert!(!reg.has_callbacks());
    reg.register_scalar("test", |_atoms, _reg| vec![]);
    assert!(reg.has_callbacks());
}

#[test]
fn test_column_width_minimum() {
    let col = parse_thermo_column("ke");
    // "Ke" is 2 chars, but minimum width is 12
    assert_eq!(col.width, 12);
}

#[test]
fn test_column_width_long_header() {
    let col = parse_thermo_column("virial_xx/long_group_name");
    // "Virial_xx/long_group_name" is 25 chars > 12
    assert!(col.width >= 12);
    assert_eq!(col.width, col.header.len());
}

#[test]
fn thermo_systems_do_not_require_group_registry_without_group_columns() {
    let mut scheduler = Scheduler::default();
    scheduler.add_scheduler_manager();
    scheduler.add_resource(RunConfig::default());
    scheduler.add_resource(ThermoConfig {
        columns: Some(vec!["step".into(), "atoms".into(), "ke".into()]),
    });
    scheduler.add_resource(CommResource(Box::new(SingleProcessComm::new())));
    scheduler.add_resource(RunState::new());
    scheduler.add_resource(Thermo::new());
    scheduler.add_resource(Atom::new());
    scheduler.add_resource(Neighbor::new());

    scheduler.add_setup_system(setup_thermo, ScheduleSetupSet::PostSetup);
    scheduler.add_update_system(print_thermo, ParticleSimScheduleSet::PostFinalIntegration);
    scheduler.organize_systems();
}

// ── Substrate restart + migration round-trip determinism ──────────────────
//
// Trust in DEM results rests on trust in the substrate: a restart written to
// disk and re-read, and an atom migrated across a subdomain boundary, must
// reconstruct the substrate state *bit-for-bit*. These tests exercise both
// persistence paths at the soil tier and stay strictly method-agnostic — they
// register a *generic* `AtomData` column (a scalar + a vec3 with no physical
// meaning), never a force law or contact model, so the guarantee they pin
// down is the substrate's, not any particular particle method's.
//
// They complement `dirt-restart-determinism-bench`, which validates
// end-to-end run continuity at the DEM tier; here we validate the underlying
// (de)serialization primitives in isolation.

use soil_core::{
    apply_permutation_in_place, Accum, Atom, AtomData, AtomDataRegistry, Domain, Input, Real,
};
use std::any::Any;

/// A generic, method-agnostic per-atom extension used only to prove the
/// substrate carries *registered* columns losslessly. It has no physical
/// meaning: a scalar and a vec3, the two shapes any particle method (DEM, MD,
/// SPH, Peridynamics, …) might register. Hand-implemented (like the `Charge`
/// example in `atom.rs`) so the test pulls in no derive machinery.
#[derive(Default, Clone, PartialEq, Debug)]
struct GenericColumn {
    scalar: Vec<f64>,
    vector: Vec<[f64; 3]>,
}

impl AtomData for GenericColumn {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn snapshot(&self) -> Box<dyn AtomData> {
        Box::new(self.clone())
    }
    fn len(&self) -> usize {
        self.scalar.len()
    }
    fn push_default(&mut self) {
        self.scalar.push(0.0);
        self.vector.push([0.0; 3]);
    }
    fn truncate(&mut self, n: usize) {
        self.scalar.truncate(n);
        self.vector.truncate(n);
    }
    fn swap_remove(&mut self, i: usize) {
        self.scalar.swap_remove(i);
        self.vector.swap_remove(i);
    }
    fn pack(&self, i: usize, buf: &mut Vec<f64>) {
        buf.push(self.scalar[i]);
        buf.extend_from_slice(&self.vector[i]);
    }
    fn unpack(&mut self, buf: &[f64]) -> usize {
        self.scalar.push(buf[0]);
        self.vector.push([buf[1], buf[2], buf[3]]);
        4
    }
    fn apply_permutation(&mut self, perm: &[usize], n: usize) {
        let mut visited = Vec::new();
        self.apply_permutation_with_scratch(perm, n, &mut visited);
    }
    fn apply_permutation_with_scratch(
        &mut self,
        perm: &[usize],
        n: usize,
        visited: &mut Vec<bool>,
    ) {
        apply_permutation_in_place(&mut self.scalar, perm, n, visited);
        apply_permutation_in_place(&mut self.vector, perm, n, visited);
    }
}

/// Per-atom fixture values with deliberately awkward bit patterns (negatives,
/// irrationals, subnormal-ish, non-power-of-two fractions) so a bit-identity
/// assertion is meaningful rather than trivially satisfied by round numbers.
/// `(tag, atom_type, pos, vel, force, mass, cutoff, image, col_scalar, col_vec)`
#[allow(clippy::type_complexity)]
fn fixture_atoms() -> (
    Vec<(
        u32,
        u32,
        [f64; 3],
        [f64; 3],
        [f64; 3],
        f64,
        f64,
        [i32; 3],
        f64,
        [f64; 3],
    )>,
) {
    (vec![
        (
            7,
            2,
            [1.0 / 3.0, -2.7182818284590452, 6.0],
            [-0.577350269, 1.4142135623730951, -9.8],
            [123.456789, -0.0009765625, 3.14159265358979],
            1.6180339887498949,
            0.05,
            [-3, 4, -1],
            0.333333333333,
            [1e-7, -2.5e6, 42.0],
        ),
        (
            11,
            0,
            [9.999999, 0.1, -0.1],
            [1e-12, -1e12, 0.25],
            [-273.15, 88.0, -0.5],
            2.5,
            0.1234567,
            [0, 0, 7],
            -1.0 / 7.0,
            [0.0, -0.0, 100.0],
        ),
        (
            42,
            1,
            [7.0, 5.0, 2.5],
            [0.0, 0.0, 0.0],
            [-1.5, 2.5, -3.5],
            0.75,
            0.03125,
            [5, -5, 0],
            987654.321,
            [1.0 / 9.0, 2.0 / 9.0, 4.0 / 9.0],
        ),
    ],)
}

/// Register a `GenericColumn` on a fresh registry.
fn registry_with_column() -> AtomDataRegistry {
    let mut reg = AtomDataRegistry::new();
    reg.try_register(GenericColumn::default(), 0).unwrap();
    reg
}

/// Build a source `Atom` + registry populated from the fixture. Uses `Real`/
/// `Accum` casts exactly as the substrate stores them, so later bit-identity
/// comparisons are against the substrate's true stored representation
/// (relevant when the build precision is f32 — the f64 wire format still
/// round-trips a stored f32 losslessly).
// `as Real`/`as Accum` are no-op casts under precision-double but genuinely
// narrow f64 -> f32 under precision-single/mixed; mirrors production `from_atoms`.
#[allow(clippy::unnecessary_cast)]
fn build_source() -> (Atom, AtomDataRegistry) {
    let (rows,) = fixture_atoms();
    let mut atoms = Atom::new();
    let reg = registry_with_column();
    {
        let mut col = reg.get_mut::<GenericColumn>().unwrap();
        for (tag, atype, pos, vel, force, mass, cutoff, image, cs, cv) in &rows {
            atoms.tag.push(*tag);
            atoms.atom_type.push(*atype);
            atoms.origin_index.push(0);
            atoms.is_ghost.push(false);
            atoms
                .pos
                .push([pos[0] as Real, pos[1] as Real, pos[2] as Real]);
            atoms
                .vel
                .push([vel[0] as Real, vel[1] as Real, vel[2] as Real]);
            atoms
                .force
                .push([force[0] as Accum, force[1] as Accum, force[2] as Accum]);
            atoms.mass.push(*mass as Real);
            atoms.inv_mass.push((1.0 / *mass) as Real);
            atoms.cutoff_radius.push(*cutoff as Real);
            atoms.image.push(*image);
            col.scalar.push(*cs);
            col.vector.push(*cv);
        }
    }
    atoms.nlocal = rows.len() as u32;
    atoms.natoms = rows.len() as u64;
    atoms.nghost = 0;
    atoms.dt = 1.0 / 4096.0; // exactly representable, non-trivial
    (atoms, reg)
}

/// Bit-identical comparison for a stored `Real` scalar.
fn real_bits_eq(a: Real, b: Real) -> bool {
    a.to_bits() == b.to_bits()
}
fn accum_bits_eq(a: Accum, b: Accum) -> bool {
    a.to_bits() == b.to_bits()
}

/// Assert every substrate field that `restart` persists round-trips bit-for-bit.
/// (Restart intentionally does not carry `image`/`origin_index`/`is_ghost`; the
/// migration test below covers `image`. We assert only what each mechanism
/// actually serializes, so the test stays honest.)
fn assert_restart_fields_bit_identical(
    src: &Atom,
    src_reg: &AtomDataRegistry,
    dst: &Atom,
    dst_reg: &AtomDataRegistry,
) {
    let n = src.nlocal as usize;
    assert_eq!(dst.nlocal, src.nlocal, "nlocal mismatch");
    assert_eq!(dst.natoms, src.natoms, "natoms mismatch");
    assert_eq!(dst.dt.to_bits(), src.dt.to_bits(), "dt not bit-identical");
    for i in 0..n {
        assert_eq!(dst.tag[i], src.tag[i], "tag[{i}]");
        assert_eq!(dst.atom_type[i], src.atom_type[i], "atom_type[{i}]");
        assert!(real_bits_eq(dst.mass[i], src.mass[i]), "mass[{i}]");
        assert!(
            real_bits_eq(dst.cutoff_radius[i], src.cutoff_radius[i]),
            "cutoff_radius[{i}]"
        );
        for d in 0..3 {
            assert!(real_bits_eq(dst.pos[i][d], src.pos[i][d]), "pos[{i}][{d}]");
            assert!(real_bits_eq(dst.vel[i][d], src.vel[i][d]), "vel[{i}][{d}]");
            assert!(
                accum_bits_eq(dst.force[i][d], src.force[i][d]),
                "force[{i}][{d}]"
            );
        }
    }
    let sc = src_reg.get::<GenericColumn>().unwrap();
    let dc = dst_reg.get::<GenericColumn>().unwrap();
    assert_eq!(dc.scalar.len(), n, "restored column length");
    assert_eq!(dc.vector.len(), n, "restored column length");
    for i in 0..n {
        assert_eq!(
            dc.scalar[i].to_bits(),
            sc.scalar[i].to_bits(),
            "column scalar[{i}]"
        );
        for d in 0..3 {
            assert_eq!(
                dc.vector[i][d].to_bits(),
                sc.vector[i][d].to_bits(),
                "column vector[{i}][{d}]"
            );
        }
    }
}

/// Restore an `Atom` + registry from a `RestartData`, mirroring `read_restart`.
/// The destination registry's store is pre-sized to `nlocal` with sentinel
/// (`NaN`) records to reproduce the post-setup state in which restarts are
/// read — this pins the invariant fixed in soil PR #2 (push-based unpack must
/// start from an empty store, else restored columns land at the wrong indices).
#[allow(clippy::unnecessary_cast)] // precision-conditional casts; see build_source
fn restore_from_restart(data: &RestartData) -> (Atom, AtomDataRegistry) {
    let n = data.tag.len();
    let mut atoms = Atom::new();
    atoms.natoms = data.natoms;
    atoms.nlocal = n as u32;
    atoms.nghost = 0;
    atoms.dt = data.dt;
    atoms.tag = data.tag.clone();
    atoms.atom_type = data.atom_type.clone();
    atoms.origin_index = vec![0; n];
    atoms.is_ghost = vec![false; n];
    atoms.pos = data
        .pos_x
        .iter()
        .zip(&data.pos_y)
        .zip(&data.pos_z)
        .map(|((&x, &y), &z)| [x as Real, y as Real, z as Real])
        .collect();
    atoms.vel = data
        .vel_x
        .iter()
        .zip(&data.vel_y)
        .zip(&data.vel_z)
        .map(|((&x, &y), &z)| [x as Real, y as Real, z as Real])
        .collect();
    atoms.force = data
        .force_x
        .iter()
        .zip(&data.force_y)
        .zip(&data.force_z)
        .map(|((&x, &y), &z)| [x as Accum, y as Accum, z as Accum])
        .collect();
    atoms.mass = data.mass.iter().map(|&m| m as Real).collect();
    atoms.inv_mass = atoms.mass.iter().map(|&m| 1.0 / m).collect();
    atoms.cutoff_radius = data.cutoff_radius.iter().map(|&r| r as Real).collect();

    let reg = registry_with_column();
    // Reproduce the post-setup store state: sized to nlocal with sentinels.
    {
        let mut col = reg.get_mut::<GenericColumn>().unwrap();
        for _ in 0..n {
            col.scalar.push(f64::NAN);
            col.vector.push([f64::NAN; 3]);
        }
    }
    if !data.atom_data_buffers.is_empty() {
        reg.unpack_all_from_restart(&data.atom_data_buffers);
    }
    (atoms, reg)
}

/// Write a restart to a real file, read it back, and assert bit-identity.
fn restart_file_roundtrip_case(format: &str) {
    let (src, src_reg) = build_source();
    let step = 128usize;
    let rank = 0i32;

    // Unique per invocation: tests run in parallel and each cleans up its own
    // dir, so a shared name would race (one test deletes another's file).
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let uniq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let base_dir = std::env::temp_dir()
        .join(format!(
            "soil_rrt_{}_{}_{}_{}",
            std::process::id(),
            format,
            step,
            uniq
        ))
        .to_string_lossy()
        .into_owned();
    let _ = fs::remove_dir_all(&base_dir);
    fs::create_dir_all(&base_dir).unwrap();

    let cfg = RestartConfig {
        interval: 1,
        format: format.to_string(),
        read: true,
    };
    let data = RestartData::from_atoms(&src, &src_reg, step);
    write_restart_inner(&data, &base_dir, step, rank, &cfg).expect("restart write");

    // Re-read exactly as `read_restart` selects and parses the file.
    let ext = if format == "json" { "json" } else { "bin" };
    let filename = format!("{}/restart_{}_rank{}.{}", base_dir, step, rank, ext);
    let file = File::open(&filename).expect("reopen restart");
    let read_back: RestartData = if ext == "json" {
        serde_json::from_reader(std::io::BufReader::new(file)).expect("parse json")
    } else {
        bincode::deserialize_from(std::io::BufReader::new(file)).expect("parse bincode")
    };
    assert_eq!(read_back.total_cycle, step, "step counter round-trip");

    let (dst, dst_reg) = restore_from_restart(&read_back);
    assert_restart_fields_bit_identical(&src, &src_reg, &dst, &dst_reg);

    let _ = fs::remove_dir_all(&base_dir);
}

#[test]
fn substrate_restart_roundtrip_bincode_bit_identical() {
    restart_file_roundtrip_case("bincode");
}

#[test]
fn substrate_restart_roundtrip_json_bit_identical() {
    restart_file_roundtrip_case("json");
}

/// Exercise the setup system itself, rather than rebuilding its replacement
/// atom in this test module.  In particular, periodic image counters are not
/// persisted by the restart wire format, but `read_restart` must still provide
/// zero-filled image rows to `ParticleStore::replace_from_restart`.
#[test]
fn production_read_restart_constructs_complete_zero_image_rows() {
    let (src, src_reg) = build_source();
    assert!(src.image.iter().any(|&image| image != [0; 3]));
    let step = 317usize;
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let uniq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let output_dir = std::env::temp_dir().join(format!(
        "soil_production_restart_{}_{}",
        std::process::id(),
        uniq
    ));
    let restart_dir = output_dir.join("restart");
    fs::create_dir_all(&restart_dir).unwrap();

    let config = RestartConfig {
        interval: 1,
        format: "bincode".to_string(),
        read: true,
    };
    let data = RestartData::from_atoms(&src, &src_reg, step);
    write_restart_inner(&data, restart_dir.to_str().unwrap(), step, 0, &config)
        .expect("restart write");

    // Run the actual production setup system.  A plain Scheduler is enough to
    // supply its real Res/ResMut parameters without duplicating read_restart.
    let mut scheduler = Scheduler::default();
    scheduler.add_resource(config);
    scheduler.add_resource(CommResource(Box::new(SingleProcessComm::new())));
    scheduler.add_resource(Input {
        filename: "production-restart-test.toml".to_string(),
        output_dir: Some(output_dir.to_string_lossy().into_owned()),
    });
    scheduler.add_resource(Atom::new());
    scheduler.add_resource(registry_with_column());
    scheduler.add_resource(RunState::new());
    scheduler.add_setup_system(read_restart, ScheduleSetupSet::PostSetup);
    scheduler.organize_systems();
    scheduler.setup();

    let restored = scheduler.get_resource_ref::<Atom>().unwrap();
    let registry = scheduler.get_resource_ref::<AtomDataRegistry>().unwrap();
    let n = src.nlocal as usize;
    assert!(restored.validate_rows(), "all Atom core rows are aligned");
    assert!(registry.validate_rows(n), "extension rows are aligned");
    assert_eq!(restored.image, vec![[0; 3]; n]);
    assert_restart_fields_bit_identical(&src, &src_reg, &restored, &registry);
    assert_eq!(
        scheduler
            .get_resource_ref::<RunState>()
            .unwrap()
            .total_cycle,
        step
    );

    let _ = fs::remove_dir_all(&output_dir);
}

/// Force an atom migration across a subdomain boundary and assert the base
/// `Atom` fields + the registered column reconstruct bit-for-bit on the
/// destination rank. This exercises the exact wire the `exchange` system uses
/// for a forward migrant: `pack_exchange` + `registry.pack_all` on the source,
/// then `unpack_atom` + `registry.unpack_all` on the destination (the body of
/// `comm::unpack_exchanged`). MPI transport is a byte-for-byte copy, so a
/// single-process test reproduces the migrated state faithfully.
#[test]
#[allow(clippy::unnecessary_cast)] // precision-conditional casts; see build_source
fn substrate_migration_across_domain_boundary_bit_identical() {
    // Two subdomains split at x = 5 in a box [0,10)^3: source owns [0,5),
    // destination owns [5,10). The migrant sits at x = 7, in the destination
    // subdomain and past the source's high face — exactly the condition under
    // which `exchange` ships it to the +x neighbour.
    let mut domain = Domain::new();
    domain.boundaries_low = [0.0; 3];
    domain.boundaries_high = [10.0; 3];
    domain.sub_domain_low = [0.0, 0.0, 0.0];
    domain.sub_domain_high = [5.0, 10.0, 10.0];
    domain.update_derived();

    let (mut src, src_reg) = build_source();

    // Pick the migrant (tag 42, at x = 7) and confirm it has genuinely crossed
    // the source subdomain's high face — i.e. this is a real boundary crossing,
    // not an arbitrary pack/unpack.
    let migrant = src
        .tag
        .iter()
        .position(|&t| t == 42)
        .expect("migrant present");
    assert!(
        src.pos[migrant][0] as f64 >= domain.sub_domain_high[0]
            && (src.pos[migrant][0] as f64) < domain.boundaries_high[0],
        "migrant must lie in the destination subdomain, across the boundary",
    );

    // Snapshot the migrant's stored (Real/Accum) representation before it is
    // removed from the source, so we compare against the substrate's own bits.
    let exp_tag = src.tag[migrant];
    let exp_type = src.atom_type[migrant];
    let exp_pos = src.pos[migrant];
    let exp_vel = src.vel[migrant];
    let exp_force = src.force[migrant];
    let exp_mass = src.mass[migrant];
    let exp_cut = src.cutoff_radius[migrant];
    let exp_image = src.image[migrant];
    let (exp_cs, exp_cv) = {
        let c = src_reg.get::<GenericColumn>().unwrap();
        (c.scalar[migrant], c.vector[migrant])
    };

    // ── Source rank: pack the migrant onto the exchange wire, then evict it ──
    let mut buf: Vec<f64> = Vec::new();
    src.pack_exchange(migrant, &mut buf);
    src_reg.pack_all(migrant, &mut buf);
    src.swap_remove(migrant);
    src_reg.swap_remove_all(migrant);
    src.nlocal -= 1;
    // Source no longer owns it.
    assert!(
        !src.tag.contains(&exp_tag),
        "migrant still on source after eviction"
    );
    assert_eq!(
        src_reg.get::<GenericColumn>().unwrap().scalar.len(),
        src.nlocal as usize
    );

    // ── Destination rank: unpack the migrant (mirrors comm::unpack_exchanged) ─
    let (mut dst, dst_reg) = build_source(); // pre-existing atoms on the dest rank
    let dst_before = dst.nlocal as usize;
    let mut pos = 0;
    pos += dst.unpack_atom(&buf[pos..], /* is_ghost = */ false);
    pos += dst_reg.unpack_all(&buf[pos..]);
    assert_eq!(pos, buf.len(), "exchange payload fully consumed");
    dst.nlocal += 1;

    let j = dst.len() - 1; // migrated atom is appended last
    assert!(
        !dst.is_ghost[j],
        "migrated atom must be a real local atom, not a ghost"
    );

    // Base Atom fields — migration carries these (incl. image, unlike restart).
    assert_eq!(dst.tag[j], exp_tag, "tag");
    assert_eq!(dst.atom_type[j], exp_type, "atom_type");
    assert_eq!(dst.image[j], exp_image, "image flags");
    assert!(real_bits_eq(dst.mass[j], exp_mass), "mass");
    assert!(real_bits_eq(dst.cutoff_radius[j], exp_cut), "cutoff_radius");
    // inv_mass is recomputed from mass on unpack; verify it is consistent.
    assert!(
        real_bits_eq(dst.inv_mass[j], (1.0 / exp_mass as f64) as Real),
        "inv_mass"
    );
    for d in 0..3 {
        assert!(real_bits_eq(dst.pos[j][d], exp_pos[d]), "pos[{d}]");
        assert!(real_bits_eq(dst.vel[j][d], exp_vel[d]), "vel[{d}]");
        assert!(accum_bits_eq(dst.force[j][d], exp_force[d]), "force[{d}]");
    }

    // Registered column — bit-identical, and correctly index-aligned with the
    // atom it belongs to (appended after the dest rank's pre-existing atoms).
    let dc = dst_reg.get::<GenericColumn>().unwrap();
    assert_eq!(
        dc.scalar.len(),
        dst.nlocal as usize,
        "column stays sized to nlocal"
    );
    assert_eq!(
        dc.scalar.len(),
        dst_before + 1,
        "exactly one atom migrated in"
    );
    assert_eq!(
        dc.scalar[j].to_bits(),
        exp_cs.to_bits(),
        "column scalar bit-identity"
    );
    for d in 0..3 {
        assert_eq!(
            dc.vector[j][d].to_bits(),
            exp_cv[d].to_bits(),
            "column vector[{d}] bit-identity"
        );
    }
}

/// End-to-end: the acceptance scenario in one test — register a column, write
/// and re-read a restart, then force a migration across a domain boundary,
/// asserting the substrate round-trips bit-identically through *both* paths.
#[test]
fn substrate_restart_and_migration_roundtrip() {
    // Restart path (bincode, the production default).
    restart_file_roundtrip_case("bincode");
    // Migration path.
    substrate_migration_across_domain_boundary_bit_identical();
}
