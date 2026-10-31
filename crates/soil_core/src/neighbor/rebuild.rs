//! Neighbor-list rebuild policy and displacement tracking.

use grass_scheduler::prelude::*;

use super::Neighbor;
use crate::{Atom, CommResource, CommState};

/// Helper: save current positions for displacement-based rebuild check.
/// Saves local atom positions and all atom tags (including ghosts) for identity tracking.
pub(super) fn save_build_positions(atoms: &Atom, neighbor: &mut Neighbor) {
    let nlocal = atoms.nlocal as usize;
    neighbor.last_build_pos.resize(nlocal, [0.0; 3]);
    neighbor.last_build_radius.resize(nlocal, 0.0);
    // atoms.pos is `Real` (f32 under mixed/single); snapshot is kept in f64 for the
    // displacement check, so widen on copy rather than bulk copy_from_slice.
    for i in 0..nlocal {
        let p = atoms.pos[i];
        neighbor.last_build_pos[i] = [p[0] as f64, p[1] as f64, p[2] as f64];
        neighbor.last_build_radius[i] = atoms.cutoff_radius[i] as f64;
    }
    neighbor.last_build_total = atoms.len();
    neighbor.steps_since_build = 0;
    let (min_skin, max_skin) = atoms.cutoff_radius[..nlocal]
        .iter()
        .fold((f64::MAX, f64::MIN), |(mn, mx), &s| {
            (mn.min(s as f64), mx.max(s as f64))
        });
    neighbor.cached_min_skin = min_skin;
    if (max_skin - min_skin).abs() < 1e-15 {
        let cutoff = 2.0 * min_skin * neighbor.skin_fraction;
        neighbor.cached_uniform_cutoff_sq = Some(cutoff * cutoff);
    } else {
        neighbor.cached_uniform_cutoff_sq = None;
    }
}

/// Helper: check displacement threshold
fn displacement_exceeded(atoms: &Atom, neighbor: &Neighbor) -> bool {
    let budget_fraction = (neighbor.skin_fraction - 1.0).max(0.0);
    // Minimum-image convention for periodic axes: when an atom wraps across a
    // periodic boundary, the raw displacement is ~box_size, but the physical
    // displacement is small. forward_comm recomputes per-atom periodic offsets
    // so ghost positions stay correct after PBC wrapping.
    let [bx, by, bz] = neighbor.pbc_box;
    let [px, py, pz] = neighbor.pbc_flags;
    let n = neighbor.last_build_pos.len();

    if px && py && pz {
        // Fully-periodic fast path (the common case). A physical displacement
        // between rebuilds is far below half a box, so the nearest-image fold is
        // a rarely-taken compare. This deliberately avoids `f64::round` and
        // `mul_add`: on the baseline x86-64 target (no SSE4.1/FMA) both lower to
        // libm calls, which made this per-step scan ~10x slower than the rest of
        // the loop body.
        let (hbx, hby, hbz) = (0.5 * bx, 0.5 * by, 0.5 * bz);
        let fold = |d: f64, half: f64, full: f64| -> f64 {
            if d > half {
                d - full
            } else if d < -half {
                d + full
            } else {
                d
            }
        };
        let pos = &atoms.pos[..n];
        let radii = &atoms.cutoff_radius[..n];
        let saved = &neighbor.last_build_pos[..n];
        let saved_radius = &neighbor.last_build_radius[..n];
        let mut exceeded = false;
        for idx in 0..n {
            let p = pos[idx];
            let q = saved[idx];
            let dx = fold(p[0] as f64 - q[0], hbx, bx);
            let dy = fold(p[1] as f64 - q[1], hby, by);
            let dz = fold(p[2] as f64 - q[2], hbz, bz);
            let radius = saved_radius[idx];
            let budget = budget_fraction * radius;
            // Non-short-circuiting `|` keeps the body branch-free so the scan
            // vectorizes; the exact rebuild decision is unchanged.
            exceeded |= ((radii[idx] as f64 - radius).abs() > 1e-15)
                | (dx * dx + dy * dy + dz * dz > budget * budget);
        }
        return exceeded;
    }

    // General path: mixed / non-periodic axes, per-axis single-image correction.
    let hbx = bx * 0.5;
    let hby = by * 0.5;
    let hbz = bz * 0.5;
    for idx in 0..n {
        let mut dx = atoms.pos[idx][0] as f64 - neighbor.last_build_pos[idx][0];
        let mut dy = atoms.pos[idx][1] as f64 - neighbor.last_build_pos[idx][1];
        let mut dz = atoms.pos[idx][2] as f64 - neighbor.last_build_pos[idx][2];
        if px {
            if dx > hbx {
                dx -= bx;
            } else if dx < -hbx {
                dx += bx;
            }
        }
        if py {
            if dy > hby {
                dy -= by;
            } else if dy < -hby {
                dy += by;
            }
        }
        if pz {
            if dz > hbz {
                dz -= bz;
            } else if dz < -hbz {
                dz += bz;
            }
        }
        let radius = neighbor.last_build_radius[idx];
        if (atoms.cutoff_radius[idx] as f64 - radius).abs() > 1e-15 {
            return true;
        }
        let budget = budget_fraction * radius;
        if dx * dx + dy * dy + dz * dz > budget * budget {
            return true;
        }
    }
    false
}

/// Helper: check if a rebuild is needed based on atom count change,
/// step count, and displacement since last build.
fn needs_rebuild(atoms: &Atom, neighbor: &Neighbor, comm_state: &CurrentState<CommState>) -> bool {
    let nlocal = atoms.nlocal as usize;
    // Always rebuild on first call or local atom count change
    if neighbor.last_build_pos.len() != nlocal
        || neighbor.last_build_radius.len() != nlocal
        || nlocal == 0
    {
        return true;
    }
    // If state is FullRebuild, a full rebuild was requested (first step,
    // or decide_rebuild detected displacement exceeded threshold).
    if comm_state.0 == CommState::FullRebuild {
        return true;
    }

    if neighbor.every == 0 {
        // Displacement-based only
        displacement_exceeded(atoms, neighbor)
    } else if neighbor.check {
        // Every N steps OR displacement exceeded
        neighbor.steps_since_build >= neighbor.every || displacement_exceeded(atoms, neighbor)
    } else {
        // Every N steps only (no displacement check)
        neighbor.steps_since_build >= neighbor.every
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Real;

    fn atom_with_radii(radii: &[f64]) -> Atom {
        let mut atoms = Atom::new();
        for (i, &r) in radii.iter().enumerate() {
            atoms.push_test_atom(i as u32, [i as f64, 0.0, 0.0], r, 1.0);
        }
        atoms.nlocal = radii.len() as u32;
        atoms.natoms = radii.len() as u64;
        atoms
    }

    #[test]
    fn polydisperse_displacement_budget_is_per_atom_radius() {
        let mut atoms = atom_with_radii(&[0.1, 1.0]);
        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = 1.5;
        save_build_positions(&atoms, &mut neighbor);

        atoms.pos[1][0] += 0.20 as Real;
        assert!(
            !displacement_exceeded(&atoms, &neighbor),
            "large atom should not rebuild at the small atom's displacement budget"
        );

        atoms.pos[0][0] += 0.051 as Real;
        assert!(
            displacement_exceeded(&atoms, &neighbor),
            "small atom should still rebuild when its own budget is exceeded"
        );
    }

    #[test]
    fn radius_change_forces_rebuild() {
        let mut atoms = atom_with_radii(&[0.5]);
        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = 1.2;
        save_build_positions(&atoms, &mut neighbor);

        atoms.cutoff_radius[0] = 0.6 as Real;
        assert!(displacement_exceeded(&atoms, &neighbor));
    }

    #[test]
    fn rebuild_policy_keeps_large_atom_valid_without_early_rebuild() {
        let mut atoms = atom_with_radii(&[0.1, 1.0]);
        let mut neighbor = Neighbor::new();
        neighbor.skin_fraction = 1.5;
        save_build_positions(&atoms, &mut neighbor);
        atoms.pos[1][0] += 0.20 as Real;

        let state = CurrentState(CommState::CommunicateOnly);
        assert!(!needs_rebuild(&atoms, &neighbor, &state));
    }
}

/// Runs at PostInitialIntegration before remove_ghost_atoms.
/// Gated by `run_if(in_state(CommState::CommunicateOnly))` — only checks when
/// the neighbor list is still considered valid.
///
/// Checks displacement to decide if a full rebuild is needed this step.
/// If so, sets CommState to FullRebuild so that remove_ghost_atoms / exchange /
/// full borders all run.
///
/// Uses all_reduce to ensure ALL ranks agree — if any rank needs a rebuild,
/// all ranks do the full rebuild (required for MPI send/recv pattern matching).
pub fn decide_rebuild(
    atoms: Res<Atom>,
    mut neighbor: ResMut<Neighbor>,
    comm: Res<CommResource>,
    mut comm_state: ResMut<CurrentState<CommState>>,
) {
    let local_needs = if needs_rebuild(&atoms, &neighbor, &comm_state) {
        1.0
    } else {
        0.0
    };
    // NOTE: we intentionally do NOT force a full rebuild just because an atom
    // drifted outside the global box (about to wrap a periodic boundary) or
    // outside its subdomain (about to migrate). The neighbor skin exists exactly
    // so atoms can drift a little out of bounds between rebuilds. Coordinate
    // wrapping is now done cheaply every step by `pbc` (runs in both
    // CommunicateOnly and FullRebuild), and ghost positions for such drifted /
    // wrapped atoms are kept correct by `forward_comm_borders` (it re-sends each
    // ghost's owner position plus the swap's FIXED `periodic_offset` from the
    // last rebuild — see `pack_forward_into`; the offset is deliberately NOT
    // recomputed per atom, so a wrapped owner's ghost stays continuous). MPI
    // migration (`exchange`) still only happens on a scheduled FullRebuild; a
    // wrapped-but-not-yet-migrated atom stays within skin of the boundary and is
    // served by the ghost mechanism until then — standard DEM. Forcing a rebuild
    // here made binning fire every step in a dense periodic box (the perf bug).
    // Any rank needing rebuild forces all ranks to rebuild
    let global_needs = comm.all_reduce_sum_f64(local_needs);
    if global_needs > 0.0 {
        comm_state.0 = CommState::FullRebuild;
    } else {
        neighbor.steps_since_build += 1;
    }
}
