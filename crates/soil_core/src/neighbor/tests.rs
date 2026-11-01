//! Neighbor-list unit and regression tests.

use super::*;
use crate::{Atom, CommState, ParticleSimScheduleSet, ScheduleSetupSet};
use grass_app::prelude::*;
use grass_scheduler::prelude::*;

#[test]
fn classify_boundary_atoms_flags_ghost_neighbours() {
    // 3 local atoms (0,1,2) + 1 ghost (3). CSR neighbour lists:
    //   atom 0 -> {1}        interior
    //   atom 1 -> {0, 3}     boundary (3 is a ghost)
    //   atom 2 -> {}         interior (no neighbours)
    let mut nb = Neighbor::new();
    nb.neighbor_offsets = vec![0, 1, 3, 3];
    nb.neighbor_indices = vec![1, 0, 3];
    let mask = classify_boundary_atoms(&nb, 3);
    assert_eq!(mask, vec![false, true, false]);
}

fn push_atom(atom: &mut Atom, tag: u32, pos: [f64; 3], radius: f64) {
    atom.push_test_atom(tag, pos, radius, 1.0);
}

#[test]
fn bin_neighbor_list_finds_close_pair() {
    let mut app = App::new();
    let mut atom = Atom::new();
    push_atom(&mut atom, 0, [0.5, 0.5, 0.5], 0.5);
    push_atom(&mut atom, 1, [1.0, 0.5, 0.5], 0.5);
    atom.nlocal = 2;
    atom.natoms = 2;

    let mut domain = crate::Domain::new();
    domain.sub_domain_low = [0.0, 0.0, 0.0];
    domain.sub_domain_high = [2.0, 2.0, 2.0];
    domain.sub_length = [2.0, 2.0, 2.0];

    let mut neighbor = Neighbor::new();
    neighbor.skin_fraction = 1.0;
    neighbor.bin_min_size = 1.0;

    app.add_resource(atom);
    app.add_resource(neighbor);
    app.add_resource(domain);
    app.add_resource(NeighborConfig::default());
    app.add_resource(CurrentState(CommState::FullRebuild));
    app.add_resource(crate::CommResource(Box::new(
        crate::SingleProcessComm::new(),
    )));
    app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
    app.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
    app.organize_systems();
    app.setup();
    app.run();

    let n = app.get_resource_ref::<Neighbor>().unwrap();
    assert!(
        n.neighbor_indices.len() >= 1,
        "bin neighbor list should find the close pair"
    );
    // Check CSR: atom 0's neighbors should contain 1
    let start = n.neighbor_offsets[0] as usize;
    let end = n.neighbor_offsets[1] as usize;
    let has_pair = n.neighbor_indices[start..end].contains(&1u32);
    assert!(has_pair, "pair (0,1) should be in CSR neighbors");
}

#[test]
fn bins_only_mode_bins_atoms_without_pairs() {
    let mut app = App::new();
    let mut atom = Atom::new();
    push_atom(&mut atom, 0, [0.5, 0.5, 0.5], 0.5);
    push_atom(&mut atom, 1, [1.0, 0.5, 0.5], 0.5);
    push_atom(&mut atom, 2, [1.5, 1.5, 1.5], 0.5);
    atom.nlocal = 3;
    atom.natoms = 3;

    let mut domain = crate::Domain::new();
    domain.sub_domain_low = [0.0, 0.0, 0.0];
    domain.sub_domain_high = [2.0, 2.0, 2.0];
    domain.sub_length = [2.0, 2.0, 2.0];

    let mut neighbor = Neighbor::new();
    neighbor.skin_fraction = 1.0;
    neighbor.bin_min_size = 1.0;
    neighbor.build_pairs = false;

    app.add_resource(atom);
    app.add_resource(neighbor);
    app.add_resource(domain);
    app.add_resource(NeighborConfig {
        pairs: false,
        ..NeighborConfig::default()
    });
    app.add_resource(CurrentState(CommState::FullRebuild));
    app.add_resource(crate::CommResource(Box::new(
        crate::SingleProcessComm::new(),
    )));
    app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
    app.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
    app.organize_systems();
    app.setup();
    app.run();

    let n = app.get_resource_ref::<Neighbor>().unwrap();
    assert!(
        n.neighbor_indices.is_empty(),
        "pairs = false builds no pairs"
    );
    assert_eq!(n.neighbor_offsets, vec![0; 4], "empty but well-formed CSR");
    assert_eq!(n.pairs(3).count(), 0);
    // Every local atom is in exactly one bin, and that bin contains its position.
    let mut seen = vec![0usize; 3];
    for c in 0..n.bin_total_cells {
        for m in n.bin_start[c] as usize..n.bin_start[c + 1] as usize {
            let i = n.bin_sorted_atoms[m] as usize;
            seen[i] += 1;
            assert_eq!(n.bin_atom_cell[i] as usize, c);
        }
    }
    assert_eq!(seen, vec![1, 1, 1]);
    assert_ne!(n.bin_atom_cell[0], n.bin_atom_cell[2]);
}

#[test]
fn pair_iter_matches_manual() {
    let mut neighbor = Neighbor::new();
    // 3 local atoms: atom 0 -> [1, 2], atom 1 -> [2], atom 2 -> []
    neighbor.neighbor_offsets = vec![0, 2, 3, 3];
    neighbor.neighbor_indices = vec![1, 2, 2];

    let pairs: Vec<(usize, usize)> = neighbor.pairs(3).collect();
    assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 2)]);

    // Row access yields the same pairs.
    assert_eq!(neighbor.neighbors_of(0), &[1, 2]);
    assert_eq!(neighbor.neighbors_of(1), &[2]);
    assert!(neighbor.neighbors_of(2).is_empty());
    assert!(neighbor.neighbors_of(3).is_empty());
    assert!(Neighbor::new().neighbors_of(0).is_empty());
}

#[test]
fn pair_iter_empty() {
    let mut neighbor = Neighbor::new();
    neighbor.neighbor_offsets = vec![0, 0, 0];
    neighbor.neighbor_indices = vec![];

    let pairs: Vec<(usize, usize)> = neighbor.pairs(2).collect();
    assert!(pairs.is_empty());

    // Also test zero local atoms
    let pairs2: Vec<(usize, usize)> = neighbor.pairs(0).collect();
    assert!(pairs2.is_empty());
}

// ── Reference vs bin-based neighbor list comparison ──────────────────

/// Reference O(N²) all-pairs neighbor finder that doesn't use ghost atoms
/// or any accelerated algorithm. Pure distance-based cutoff check.
fn reference_all_pairs(
    positions: &[[f64; 3]],
    cutoff_radii: &[f64],
    skin_fraction: f64,
    nlocal: usize,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..nlocal {
        for j in (i + 1)..positions.len() {
            let dx = positions[j][0] - positions[i][0];
            let dy = positions[j][1] - positions[i][1];
            let dz = positions[j][2] - positions[i][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let cutoff = (cutoff_radii[i] + cutoff_radii[j]) * skin_fraction;
            if dist < cutoff {
                pairs.push((i, j));
            }
        }
    }
    pairs.sort();
    pairs
}

#[test]
fn bin_neighbor_list_matches_reference() {
    // Create a system and verify that bin-based neighbor list finds
    // exactly the same pairs as the reference all-pairs calculation.
    let n = 30;
    let skin_fraction = 1.0;
    let radius = 0.3;

    let mut positions = Vec::new();
    for i in 0..n {
        let x = (i as f64 * 0.31).sin() * 1.5 + 2.0;
        let y = (i as f64 * 0.67).cos() * 1.5 + 2.0;
        let z = (i as f64 * 0.97).sin() * 1.5 + 2.0;
        positions.push([x, y, z]);
    }

    // Reference all-pairs
    let cutoffs: Vec<f64> = vec![radius; n];
    let ref_pairs = reference_all_pairs(&positions, &cutoffs, skin_fraction, n);

    // Bin-based neighbor list
    let mut app_bin = App::new();
    let mut atom_bin = Atom::new();
    for i in 0..n {
        push_atom(&mut atom_bin, i as u32, positions[i], radius);
    }
    atom_bin.nlocal = n as u32;
    atom_bin.natoms = n as u64;

    let mut domain = crate::Domain::new();
    domain.sub_domain_low = [0.0, 0.0, 0.0];
    domain.sub_domain_high = [4.0, 4.0, 4.0];
    domain.sub_length = [4.0, 4.0, 4.0];

    let mut neighbor_bin = Neighbor::new();
    neighbor_bin.skin_fraction = skin_fraction;
    neighbor_bin.bin_min_size = 0.5;

    app_bin.add_resource(atom_bin);
    app_bin.add_resource(neighbor_bin);
    app_bin.add_resource(domain);
    app_bin.add_resource(NeighborConfig::default());
    app_bin.add_resource(CurrentState(CommState::FullRebuild));
    app_bin.add_resource(crate::CommResource(Box::new(
        crate::SingleProcessComm::new(),
    )));
    app_bin.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
    app_bin.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
    app_bin.organize_systems();
    app_bin.setup();
    app_bin.run();

    let neigh_bin = app_bin.get_resource_ref::<Neighbor>().unwrap();
    // Extract pairs from CSR format
    let nlocal = n;
    let mut bin_pairs = Vec::new();
    for i in 0..nlocal {
        if i + 1 >= neigh_bin.neighbor_offsets.len() {
            break;
        }
        let start = neigh_bin.neighbor_offsets[i] as usize;
        let end = neigh_bin.neighbor_offsets[i + 1] as usize;
        for k in start..end {
            let j = neigh_bin.neighbor_indices[k] as usize;
            let pair = if i < j { (i, j) } else { (j, i) };
            bin_pairs.push(pair);
        }
    }
    bin_pairs.sort();
    bin_pairs.dedup();

    // Verify: bin-based should find all pairs that reference finds
    for pair in &ref_pairs {
        assert!(
            bin_pairs.contains(pair),
            "Bin neighbor list missed pair {:?} found by reference",
            pair
        );
    }
}

#[test]
fn full_bin_neighbor_list_scans_self_cell_without_self_pairs() {
    // Dense same-cell case for the newton=false path: every atom is inside
    // the self-cell cutoff, so the full neighbor list should contain both
    // directions of each pair and no (i, i) self entries.
    let mut app = App::new();
    let mut atom = Atom::new();
    push_atom(&mut atom, 0, [0.50, 0.50, 0.50], 0.5);
    push_atom(&mut atom, 1, [0.75, 0.50, 0.50], 0.5);
    push_atom(&mut atom, 2, [0.50, 0.75, 0.50], 0.5);
    atom.nlocal = 3;
    atom.natoms = 3;

    let mut domain = crate::Domain::new();
    domain.sub_domain_low = [0.0, 0.0, 0.0];
    domain.sub_domain_high = [2.0, 2.0, 2.0];
    domain.sub_length = [2.0, 2.0, 2.0];

    let mut neighbor = Neighbor::new();
    neighbor.skin_fraction = 1.0;
    neighbor.bin_min_size = 1.0;
    neighbor.newton = false;

    app.add_resource(atom);
    app.add_resource(neighbor);
    app.add_resource(domain);
    app.add_resource(NeighborConfig {
        newton: false,
        ..NeighborConfig::default()
    });
    app.add_resource(CurrentState(CommState::FullRebuild));
    app.add_resource(crate::CommResource(Box::new(
        crate::SingleProcessComm::new(),
    )));
    app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
    app.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
    app.organize_systems();
    app.setup();
    app.run();

    let neigh = app.get_resource_ref::<Neighbor>().unwrap();
    let mut directed_pairs = Vec::new();
    for i in 0..3 {
        let start = neigh.neighbor_offsets[i] as usize;
        let end = neigh.neighbor_offsets[i + 1] as usize;
        for &j in &neigh.neighbor_indices[start..end] {
            directed_pairs.push((i, j as usize));
        }
    }
    directed_pairs.sort();
    assert_eq!(
        directed_pairs,
        vec![(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    );
}

fn full_neighbor_directed_pairs(neighbor: &Neighbor, nlocal: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..nlocal {
        let start = neighbor.neighbor_offsets[i] as usize;
        let end = neighbor.neighbor_offsets[i + 1] as usize;
        for &j in &neighbor.neighbor_indices[start..end] {
            pairs.push((i, j as usize));
        }
    }
    pairs.sort();
    pairs
}

fn legacy_full_neighbor_directed_pairs(
    atom: &Atom,
    neighbor: &Neighbor,
    nlocal: usize,
    cutoffs: &[f64],
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let skin_fraction_sq = neighbor.skin_fraction * neighbor.skin_fraction;
    for i in 0..nlocal {
        let my_cell = neighbor.bin_atom_cell[i] as usize;
        let pi = atom.pos[i];
        let pi = [pi[0] as f64, pi[1] as f64, pi[2] as f64];
        for &offset in &neighbor.bin_stencil {
            let c = (my_cell as i32 + offset) as usize;
            let start = neighbor.bin_start[c] as usize;
            let end = neighbor.bin_start[c + 1] as usize;
            for m in start..end {
                let j = neighbor.bin_sorted_atoms[m] as usize;
                if offset == 0 && j == i {
                    continue;
                }
                let pj = neighbor.bin_sorted_pos[m];
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let dz = pj[2] - pi[2];
                let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                let sum_skin = cutoffs[i] + cutoffs[j];
                if r2 < sum_skin * sum_skin * skin_fraction_sq {
                    pairs.push((i, j));
                }
            }
        }
    }
    pairs.sort();
    pairs
}

#[test]
fn full_bin_neighbor_list_matches_legacy_self_cell_scan() {
    // Regression for the newton=false full-list path: production code scans
    // the self cell separately and uses a stencil with offset 0 removed. This
    // must produce the same directed pair set as the legacy full-stencil scan
    // with hot-loop `offset == 0 && j == i` filtering.
    let n = 12;
    let skin_fraction = 1.0;
    let positions = [
        [0.20, 0.20, 0.20],
        [0.55, 0.22, 0.20],
        [0.20, 0.58, 0.20],
        [0.62, 0.60, 0.20],
        [1.05, 0.24, 0.20],
        [1.38, 0.22, 0.20],
        [0.22, 1.05, 0.20],
        [0.58, 1.35, 0.20],
        [1.08, 1.08, 0.20],
        [1.42, 1.38, 0.20],
        [2.20, 0.20, 0.20],
        [2.55, 0.22, 0.20],
    ];
    let cutoffs = [
        0.30, 0.36, 0.28, 0.34, 0.32, 0.27, 0.35, 0.31, 0.29, 0.37, 0.30, 0.33,
    ];

    let mut app = App::new();
    let mut atom = Atom::new();
    for i in 0..n {
        push_atom(&mut atom, i as u32, positions[i], cutoffs[i]);
    }
    atom.nlocal = n as u32;
    atom.natoms = n as u64;

    let mut domain = crate::Domain::new();
    domain.sub_domain_low = [0.0, 0.0, 0.0];
    domain.sub_domain_high = [3.0, 2.0, 2.0];
    domain.sub_length = [3.0, 2.0, 2.0];

    let mut neighbor = Neighbor::new();
    neighbor.skin_fraction = skin_fraction;
    neighbor.bin_min_size = 0.5;
    neighbor.newton = false;

    app.add_resource(atom);
    app.add_resource(neighbor);
    app.add_resource(domain);
    app.add_resource(NeighborConfig {
        newton: false,
        ..NeighborConfig::default()
    });
    app.add_resource(CurrentState(CommState::FullRebuild));
    app.add_resource(crate::CommResource(Box::new(
        crate::SingleProcessComm::new(),
    )));
    app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
    app.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
    app.organize_systems();
    app.setup();
    app.run();

    let atom = app.get_resource_ref::<Atom>().unwrap();
    let neighbor = app.get_resource_ref::<Neighbor>().unwrap();
    let optimized_pairs = full_neighbor_directed_pairs(&neighbor, n);
    let legacy_pairs = legacy_full_neighbor_directed_pairs(&atom, &neighbor, n, &cutoffs);

    assert_eq!(optimized_pairs, legacy_pairs);
    assert!(!optimized_pairs.iter().any(|(i, j)| i == j));
    assert!(
        optimized_pairs.iter().any(|(i, j)| i > j) && optimized_pairs.iter().any(|(i, j)| i < j),
        "newton=false full list should contain both pair directions"
    );
}

#[test]
fn triclinic_bin_neighbor_list_matches_reference() {
    // Same cross-check as above but with a TILTED (triclinic) box. Atoms are
    // kept in the interior so no periodic/ghost pairs are needed — this isolates
    // the lamda binning + triclinic stencil: it must still find every real-space
    // pair within cutoff (no misses from a too-narrow sheared stencil).
    let n = 40;
    let skin_fraction = 1.0;
    let radius = 0.3;
    let xy = 1.0; // box tilt (0.25*Lx, within the box-flip bound)

    let mut positions = Vec::new();
    for i in 0..n {
        // Spread through the interior [0.6, 3.4]^3 so no wrap is required.
        let x = (i as f64 * 0.31).sin() * 1.4 + 2.0;
        let y = (i as f64 * 0.67).cos() * 1.4 + 2.0;
        let z = (i as f64 * 0.97).sin() * 1.4 + 2.0;
        positions.push([x, y, z]);
    }

    let cutoffs: Vec<f64> = vec![radius; n];
    let ref_pairs = reference_all_pairs(&positions, &cutoffs, skin_fraction, n);

    let mut app = App::new();
    let mut atom = Atom::new();
    for i in 0..n {
        push_atom(&mut atom, i as u32, positions[i], radius);
    }
    atom.nlocal = n as u32;
    atom.natoms = n as u64;

    let mut domain = crate::Domain::new();
    domain.boundaries_low = [0.0, 0.0, 0.0];
    domain.boundaries_high = [4.0, 4.0, 4.0];
    domain.sub_domain_low = [0.0, 0.0, 0.0];
    domain.sub_domain_high = [4.0, 4.0, 4.0];
    domain.size = [4.0, 4.0, 4.0];
    domain.sub_length = [4.0, 4.0, 4.0];
    domain.sub_lamda_low = [0.0, 0.0, 0.0];
    domain.sub_lamda_high = [1.0, 1.0, 1.0];
    domain.triclinic = true;
    domain.tilt = [xy, 0.0, 0.0];

    let mut neighbor = Neighbor::new();
    neighbor.skin_fraction = skin_fraction;
    neighbor.bin_min_size = 0.5;

    app.add_resource(atom);
    app.add_resource(neighbor);
    app.add_resource(domain);
    app.add_resource(NeighborConfig::default());
    app.add_resource(CurrentState(CommState::FullRebuild));
    app.add_resource(crate::CommResource(Box::new(
        crate::SingleProcessComm::new(),
    )));
    app.add_setup_system(neighbor_setup, ScheduleSetupSet::PostSetup);
    app.add_update_system(bin_neighbor_list, ParticleSimScheduleSet::Neighbor);
    app.organize_systems();
    app.setup();
    app.run();

    let neigh = app.get_resource_ref::<Neighbor>().unwrap();
    let mut bin_pairs = Vec::new();
    for i in 0..n {
        if i + 1 >= neigh.neighbor_offsets.len() {
            break;
        }
        let start = neigh.neighbor_offsets[i] as usize;
        let end = neigh.neighbor_offsets[i + 1] as usize;
        for k in start..end {
            let j = neigh.neighbor_indices[k] as usize;
            bin_pairs.push(if i < j { (i, j) } else { (j, i) });
        }
    }
    bin_pairs.sort();
    bin_pairs.dedup();

    for pair in &ref_pairs {
        assert!(
            bin_pairs.contains(pair),
            "Triclinic bin neighbor list missed pair {:?} found by reference",
            pair
        );
    }
}
