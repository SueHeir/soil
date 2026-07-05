//! Spatial binning, stencil construction, atom sorting, and CSR list construction.

use grass_scheduler::prelude::*;

use super::rebuild::save_build_positions;
use super::Neighbor;
use crate::{Atom, AtomDataRegistry, CommResource, CommState, Domain};

/// Compute bin grid parameters (count, size, origin, stencil) from domain bounds and cutoff.
///
/// Shared helper used by both [`neighbor_setup`] (initial setup) and [`recompute_bins`]
/// (after shrink-wrap domain changes). Updates `bin_count`, `bin_size`, `bin_origin`,
/// `bin_total_cells`, stencil arrays, and PBC box dimensions on `neighbor`.
///
/// # Bin grid layout
///
/// The bin grid extends beyond the sub-domain by `sx`/`sy`/`sz` ghost layers on each
/// side, where `s = ceil(cutoff / bin_size)`. This ensures ghost atoms (which extend
/// up to `cutoff` beyond the sub-domain) are binned into valid cells.
///
/// Cell indexing is row-major: `cell = cx * ny * nz + cy * nz + cz`.
///
/// # Stencil construction
///
/// The stencil lists cell offsets `(dx, dy, dz)` whose **minimum possible distance**
/// to the origin cell is less than the cutoff. The minimum distance between two cells
/// offset by `d` bins is `max(0, |d| - 1) * bin_size` (since atoms can be anywhere
/// within their cell). Only cells passing this spherical distance test are included.
pub(super) fn compute_bin_grid(neighbor: &mut Neighbor, domain: &Domain, comm_size: i32) {
    let max_cutoff = neighbor.cached_max_cutoff;

    // Multi-process: bins must be at most cutoff/2 so each sub-domain has enough bins.
    // Single-process: bins >= cutoff gives stencil range = 1 (fewer cells to check).
    let required_bin = if comm_size > 1 {
        max_cutoff * 0.5
    } else {
        max_cutoff
    };
    let min_bin = neighbor.bin_min_size.max(required_bin);

    if domain.triclinic {
        compute_bin_grid_triclinic(neighbor, domain, max_cutoff, min_bin);
        return;
    }

    // Compute number of interior bins per axis (at least 1), then actual bin sizes.
    let xi = (domain.sub_length[0] / min_bin).floor().max(1.0) as i32;
    let yi = (domain.sub_length[1] / min_bin).floor().max(1.0) as i32;
    let zi = (domain.sub_length[2] / min_bin).floor().max(1.0) as i32;

    let actual_bin_size = [
        domain.sub_length[0] / xi as f64,
        domain.sub_length[1] / yi as f64,
        domain.sub_length[2] / zi as f64,
    ];

    // Ghost layers per side: enough bins to cover the ghost zone + stencil reach.
    // When ghost_cutoff > pair cutoff (e.g., dirt_clump extends it for sub-sphere
    // offsets), local atoms can end up in ghost bins. The stencil from those bins
    // must not exceed the grid, so we add stencil range on top of ghost layers.
    let ghost_cut = if domain.ghost_cutoff > max_cutoff {
        domain.ghost_cutoff
    } else {
        max_cutoff
    };
    let stencil_range_x = (max_cutoff / actual_bin_size[0]).ceil() as i32;
    let stencil_range_y = (max_cutoff / actual_bin_size[1]).ceil() as i32;
    let stencil_range_z = (max_cutoff / actual_bin_size[2]).ceil() as i32;
    let sx = (ghost_cut / actual_bin_size[0]).ceil() as i32 + stencil_range_x;
    let sy = (ghost_cut / actual_bin_size[1]).ceil() as i32 + stencil_range_y;
    let sz = (ghost_cut / actual_bin_size[2]).ceil() as i32 + stencil_range_z;

    // Total bins = interior + 2 * ghost layers per axis.
    let nx = xi + 2 * sx;
    let ny = yi + 2 * sy;
    let nz = zi + 2 * sz;
    neighbor.bin_count = [nx, ny, nz];
    neighbor.bin_size = actual_bin_size;

    let total_cells = (nx * ny * nz) as usize;
    neighbor.bin_total_cells = total_cells;

    // Bin origin = sub-domain corner shifted left by ghost layers.
    neighbor.bin_origin = [
        domain.sub_domain_low[0] - actual_bin_size[0] * sx as f64,
        domain.sub_domain_low[1] - actual_bin_size[1] * sy as f64,
        domain.sub_domain_low[2] - actual_bin_size[2] * sz as f64,
    ];

    // Precompute stencil offsets — only include cells whose minimum distance < cutoff.
    // Minimum distance between cells offset by d bins = max(0, |d|-1) * bin_size,
    // because atoms within adjacent cells (|d|=1) can be arbitrarily close.
    let cutoff_sq = max_cutoff * max_cutoff;
    neighbor.bin_stencil.clear();
    neighbor.bin_stencil_forward.clear();
    neighbor.bin_stencil_self = false;
    for dx in -sx..=sx {
        for dy in -sy..=sy {
            for dz in -sz..=sz {
                let min_dx = (dx.unsigned_abs().saturating_sub(1)) as f64 * actual_bin_size[0];
                let min_dy = (dy.unsigned_abs().saturating_sub(1)) as f64 * actual_bin_size[1];
                let min_dz = (dz.unsigned_abs().saturating_sub(1)) as f64 * actual_bin_size[2];
                if min_dx * min_dx + min_dy * min_dy + min_dz * min_dz < cutoff_sq {
                    // Row-major cell offset for 3D -> 1D indexing.
                    let offset = dx * ny * nz + dy * nz + dz;
                    neighbor.bin_stencil.push(offset);
                    if offset > 0 {
                        neighbor.bin_stencil_forward.push(offset);
                    } else if offset == 0 {
                        neighbor.bin_stencil_self = true;
                    }
                }
            }
        }
    }

    // Cache full box dimensions for minimum-image displacement checks.
    neighbor.pbc_box = [
        domain.boundaries_high[0] - domain.boundaries_low[0],
        domain.boundaries_high[1] - domain.boundaries_low[1],
        domain.boundaries_high[2] - domain.boundaries_low[2],
    ];
}

/// Triclinic bin grid: bins live in fractional (lamda) coordinates where the
/// tilted box is the unit cube, so `bin_origin`/`bin_size` are lamda quantities and
/// the binning code must map positions through [`Domain::x2lamda`] before flooring.
///
/// The grid (origin/size/count) is **tilt-independent** — only the stencil depends
/// on the tilt, because a step in λ₁ (the gradient) shifts real x by `xy·Δλ₁`, so
/// y-neighbours appear x-displaced. The stencil is rebuilt for the *current* tilt
/// each rebuild (the deform forces a rebuild every step anyway). The min-distance
/// test is deliberately **conservative** (it lets the x term be cancelled and the
/// y/z separations reach their per-cell minimum independently) so no real-space
/// neighbour within `max_cutoff` is ever excluded; the real-space distance filter
/// in `bin_neighbor_list` removes the resulting false positives.
fn compute_bin_grid_triclinic(
    neighbor: &mut Neighbor,
    domain: &Domain,
    max_cutoff: f64,
    min_bin: f64,
) {
    let l = domain.size; // real edge lengths [Lx, Ly, Lz]
    let xy = domain.tilt[0];
    // Lamda sub-domain lengths (uniform fractions = 1/nproc per dim).
    let sub_l = [
        domain.sub_lamda_high[0] - domain.sub_lamda_low[0],
        domain.sub_lamda_high[1] - domain.sub_lamda_low[1],
        domain.sub_lamda_high[2] - domain.sub_lamda_low[2],
    ];

    // Interior bins per dim, sized so each bin's real extent >= min_bin.
    let ni = [
        (sub_l[0] * l[0] / min_bin).floor().max(1.0) as i32,
        (sub_l[1] * l[1] / min_bin).floor().max(1.0) as i32,
        (sub_l[2] * l[2] / min_bin).floor().max(1.0) as i32,
    ];
    // Lamda bin sizes and their approximate real extents along each lamda axis.
    let bl = [
        sub_l[0] / ni[0] as f64,
        sub_l[1] / ni[1] as f64,
        sub_l[2] / ni[2] as f64,
    ];
    let real_bin = [bl[0] * l[0], bl[1] * l[1], bl[2] * l[2]];

    let ghost_cut = if domain.ghost_cutoff > max_cutoff {
        domain.ghost_cutoff
    } else {
        max_cutoff
    };
    let stencil_range = [
        (max_cutoff / real_bin[0]).ceil() as i32,
        (max_cutoff / real_bin[1]).ceil() as i32,
        (max_cutoff / real_bin[2]).ceil() as i32,
    ];
    // Box-flip bounds |xy| <= 0.5*Lx, so the x stencil/ghost reach must cover that
    // tilt skew in addition to the cutoff (a y-neighbour is shifted up to xy in x).
    let skew_x_bins = (xy.abs() / real_bin[0]).ceil() as i32;
    let sx = (ghost_cut / real_bin[0]).ceil() as i32 + stencil_range[0] + skew_x_bins;
    let sy = (ghost_cut / real_bin[1]).ceil() as i32 + stencil_range[1];
    let sz = (ghost_cut / real_bin[2]).ceil() as i32 + stencil_range[2];

    let nx = ni[0] + 2 * sx;
    let ny = ni[1] + 2 * sy;
    let nz = ni[2] + 2 * sz;
    neighbor.bin_count = [nx, ny, nz];
    neighbor.bin_size = bl;
    neighbor.bin_total_cells = (nx * ny * nz) as usize;
    neighbor.bin_origin = [
        domain.sub_lamda_low[0] - bl[0] * sx as f64,
        domain.sub_lamda_low[1] - bl[1] * sy as f64,
        domain.sub_lamda_low[2] - bl[2] * sz as f64,
    ];

    // Stencil: conservative triclinic min-distance test (see fn doc).
    let cutoff_sq = max_cutoff * max_cutoff;
    neighbor.bin_stencil.clear();
    neighbor.bin_stencil_forward.clear();
    neighbor.bin_stencil_self = false;
    for dx in -sx..=sx {
        for dy in -sy..=sy {
            for dz in -sz..=sz {
                // y/z contribution at their per-cell minimum separation.
                let m1 = (dy.unsigned_abs().saturating_sub(1)) as f64 * bl[1];
                let m2 = (dz.unsigned_abs().saturating_sub(1)) as f64 * bl[2];
                let dyz_sq = (l[1] * m1) * (l[1] * m1) + (l[2] * m2) * (l[2] * m2);
                if dyz_sq >= cutoff_sq {
                    continue;
                }
                let budget = (cutoff_sq - dyz_sq).sqrt();
                // Achievable real-x displacement interval, letting Δλ0 and Δλ1 each
                // range over their full per-cell boxes (generous → never misses).
                let a_lo = l[0] * (dx as f64 - 1.0) * bl[0];
                let a_hi = l[0] * (dx as f64 + 1.0) * bl[0];
                let b1 = xy * (dy as f64 - 1.0) * bl[1];
                let b2 = xy * (dy as f64 + 1.0) * bl[1];
                let (b_lo, b_hi) = if b1 <= b2 { (b1, b2) } else { (b2, b1) };
                let lo = a_lo + b_lo;
                let hi = a_hi + b_hi;
                if hi >= -budget && lo <= budget {
                    let offset = dx * ny * nz + dy * nz + dz;
                    neighbor.bin_stencil.push(offset);
                    if offset > 0 {
                        neighbor.bin_stencil_forward.push(offset);
                    } else if offset == 0 {
                        neighbor.bin_stencil_self = true;
                    }
                }
            }
        }
    }

    neighbor.pbc_box = [l[0], l[1], l[2]];
}

/// Recompute bin grid parameters after domain bounds change (e.g., shrink-wrap).
fn recompute_bins(neighbor: &mut Neighbor, domain: &Domain, comm_size: i32) {
    if neighbor.cached_max_cutoff <= 0.0 {
        return; // not yet initialized
    }
    compute_bin_grid(neighbor, domain, comm_size);
}

/// Reorder local atoms by spatial bin for improved cache locality.
///
/// Runs at [`ParticleSimScheduleSet::PreNeighbor`] (before ghost communication). Atoms are
/// sorted by their bin cell index so that spatially nearby atoms are contiguous
/// in memory, improving cache hit rates during neighbor list construction and
/// force computation.
///
/// Sorting is triggered either periodically (every `sort_every` steps) or when
/// a neighbor rebuild is needed. The permutation is applied to all atom arrays
/// (via [`AtomDataRegistry`]) and ghost `origin_index` values are updated.
pub fn sort_atoms_by_bin(
    mut atoms: ResMut<Atom>,
    mut neighbor: ResMut<Neighbor>,
    comm: Res<CommResource>,
    registry: Res<AtomDataRegistry>,
    mut domain: ResMut<Domain>,
    mut comm_state: ResMut<CurrentState<CommState>>,
) {
    // Recompute bin grid if domain bounds changed (e.g., shrink-wrap).
    // Must happen before sorting since sorting depends on bin parameters.
    if domain.bounds_changed {
        recompute_bins(&mut neighbor, &domain, comm.size());
        domain.bounds_changed = false;
        comm_state.0 = CommState::FullRebuild;
    }
    // Increment sort_counter first so it stays synchronized across all MPI ranks,
    // even if some ranks skip sorting due to nlocal == 0 or other conditions.
    neighbor.sort_counter += 1;
    let periodic_sort = neighbor.sort_every > 0 && neighbor.sort_counter >= neighbor.sort_every;
    if periodic_sort {
        neighbor.sort_counter = 0;
    }

    let nlocal = atoms.nlocal as usize;
    // N2: invalidate the cached local cells by default; only the full sort below
    // republishes them. Any early return here leaves the build to bin all atoms.
    neighbor.local_cells_valid = false;
    if nlocal == 0 || neighbor.bin_total_cells == 0 || nlocal > atoms.pos.len() {
        // Even with nothing to sort locally, must participate in the all_reduce
        // when periodic_sort triggers so all MPI ranks stay synchronized.
        if periodic_sort {
            let global_did_sort = comm.all_reduce_sum_f64(0.0);
            if global_did_sort > 0.0 {
                neighbor.last_build_pos.clear();
                comm_state.0 = CommState::FullRebuild;
            }
        }
        return;
    }

    // Only reorder local atoms when a full borders rebuild is going to run this
    // step (or when a periodic sort is due, which itself forces a rebuild below).
    //
    // Reordering locals invalidates the saved ghost sendlists (`swap_data`): their
    // `send_indices`, `recv_start`, and the `origin_index` echoed back during the
    // reverse force comm all reference pre-permutation local indices. Those are
    // only regenerated by `borders`, which runs in `CommState::FullRebuild`. On a
    // `CommunicateOnly` step `borders` does NOT run, so permuting there would leave
    // `reverse_send_force` accumulating forces onto the wrong owner atoms.
    //
    // Tying the permute to the *globally consistent* `FullRebuild` state (set
    // collectively by `decide_rebuild`) also keeps every MPI rank on the same code
    // path: the collective in the periodic-sort branch below can never deadlock by
    // some ranks permuting while others early-return.
    let full_rebuild = matches!(comm_state.0, CommState::FullRebuild);
    if !periodic_sort && !full_rebuild {
        return;
    }

    let inv_bsx = 1.0 / neighbor.bin_size[0];
    let inv_bsy = 1.0 / neighbor.bin_size[1];
    let inv_bsz = 1.0 / neighbor.bin_size[2];
    let nx = neighbor.bin_count[0];
    let ny = neighbor.bin_count[1];
    let nz = neighbor.bin_count[2];
    let total_cells = (nx * ny * nz) as usize;
    let [ox, oy, oz] = neighbor.bin_origin;

    // O(N) stable counting sort by bin cell (replaces an O(N log N) comparison sort
    // + per-rebuild allocations). Within a bin, atoms keep ascending original index.
    // Note the *neighbor build* re-sorts independently with its own stable counting
    // sort, so this spatial sort only affects cache locality, never the pair set.
    // Scratch buffers are reused across rebuilds.
    let mut cell = std::mem::take(&mut neighbor.sort_cell);
    let mut counts = std::mem::take(&mut neighbor.sort_counts);
    let mut perm = std::mem::take(&mut neighbor.sort_perm);
    let mut pos_scratch = std::mem::take(&mut neighbor.sort_pos_scratch);
    let mut sorted_local_cell = std::mem::take(&mut neighbor.sorted_local_cell);
    cell.clear();
    cell.resize(nlocal, 0u32);
    counts.clear();
    counts.resize(total_cells + 1, 0u32);
    perm.clear();
    perm.resize(nlocal, 0usize);

    // 1. assign each atom to a cell and histogram (counts[c+1] = #atoms in cell c).
    // Triclinic: bin in fractional (lamda) coords where the box is the unit cube.
    let triclinic = domain.triclinic;
    for i in 0..nlocal {
        let pr = atoms.pos[i];
        let pf = [pr[0] as f64, pr[1] as f64, pr[2] as f64];
        let p = if triclinic { domain.x2lamda(pf) } else { pf };
        let cx = (((p[0] - ox) * inv_bsx).floor() as i32).clamp(0, nx - 1);
        let cy = (((p[1] - oy) * inv_bsy).floor() as i32).clamp(0, ny - 1);
        let cz = (((p[2] - oz) * inv_bsz).floor() as i32).clamp(0, nz - 1);
        let c = (cx * ny * nz + cy * nz + cz) as u32;
        cell[i] = c;
        counts[c as usize + 1] += 1;
    }
    // 2. prefix sum: counts[c] becomes the start cursor of cell c
    for c in 0..total_cells {
        counts[c + 1] += counts[c];
    }
    // 3. stable scatter: perm[cursor]=i in ascending i, advancing the per-cell cursor
    for i in 0..nlocal {
        let c = cell[i] as usize;
        let slot = counts[c];
        perm[slot as usize] = i;
        counts[c] = slot + 1;
    }

    // Skip permutation if atoms are already in order
    let already_sorted = perm.iter().enumerate().all(|(i, &p)| p == i);

    // N2: publish the POST-permute local cell indices for the neighbor build to
    // reuse (the new atom at index k is the old atom perm[k], whose cell is
    // cell[perm[k]]). Identical formula to the build, so the resulting cell array
    // — hence the neighbor list — is bit-for-bit the same as computing it there.
    sorted_local_cell.clear();
    sorted_local_cell.resize(nlocal, 0u32);
    for k in 0..nlocal {
        sorted_local_cell[k] = cell[perm[k]];
    }
    neighbor.local_cells_valid = true;

    if !already_sorted {
        atoms.apply_permutation(&perm, nlocal);
        registry.apply_permutation_all(&perm, nlocal);

        // Apply the same permutation to last_build_pos so displacement checks
        // compare the correct atom's current position against its saved position.
        if neighbor.last_build_pos.len() >= nlocal {
            pos_scratch.clear();
            pos_scratch.extend_from_slice(&neighbor.last_build_pos[..nlocal]);
            for (new_i, &old_i) in perm.iter().enumerate() {
                neighbor.last_build_pos[new_i] = pos_scratch[old_i];
            }
        }

        // No ghost `origin_index` fix-up is attempted here. A permutation only ever
        // runs in lock-step with a `FullRebuild` (see the guard above), and `borders`
        // discards every ghost and rebuilds the sendlists from scratch immediately
        // afterwards, so any existing ghost indices are about to be thrown away.
        // (Remapping them here would in fact be wrong under MPI: a ghost's
        // `origin_index` refers to a *local index on the owning rank*, which has no
        // relation to this rank's local permutation.)
    }

    // Return scratch buffers for reuse next rebuild.
    neighbor.sort_cell = cell;
    neighbor.sort_counts = counts;
    neighbor.sort_perm = perm;
    neighbor.sort_pos_scratch = pos_scratch;
    neighbor.sorted_local_cell = sorted_local_cell;

    // After permutation, swap_data.send_indices are stale (they reference pre-sort
    // local indices). Force a full borders rebuild so new swap_data is generated.
    // When periodic_sort triggered, all ranks entered this function (they share the
    // same sort_counter). Use all_reduce to synchronize: if ANY rank permuted, ALL
    // ranks must do a full borders rebuild (borders uses collective MPI communication
    // that requires all ranks to follow the same code path).
    // The all_reduce MUST happen after all early returns so every rank participates.
    if periodic_sort {
        let local_did_sort = if already_sorted { 0.0 } else { 1.0 };
        let global_did_sort = comm.all_reduce_sum_f64(local_did_sort);
        if global_did_sort > 0.0 {
            neighbor.last_build_pos.clear();
            comm_state.0 = CommState::FullRebuild;
        }
    }
}

/// Bin-based neighbor list builder. O(N) expected time for uniform particle distributions.
///
/// Uses a spatial bin grid with a precomputed stencil of neighbor cells. The algorithm:
///
/// 1. **Assign** each atom to a bin cell based on its position (counting sort).
/// 2. **Build** CSR bin offsets so `bin_start[c]..bin_start[c+1]` gives atoms in cell `c`.
/// 3. **Scan** each local atom's self-cell (j > i only) and forward stencil cells to find
///    neighbors within cutoff, producing a half neighbor list in CSR format.
///
/// Two code paths are used: a **fast path** when all atoms share the same cutoff radius
/// (skips per-pair cutoff computation), and a **slow path** for polydisperse systems.
///
/// All inner loops use `unsafe` unchecked indexing for performance; safety invariants
/// are documented inline and validated by the CSR construction logic.
pub fn bin_neighbor_list(
    atoms: Res<Atom>,
    mut neighbor: ResMut<Neighbor>,
    mut comm_state: ResMut<CurrentState<CommState>>,
    domain: Res<Domain>,
) {
    let nlocal = atoms.nlocal as usize;
    let total = atoms.len();
    let triclinic = domain.triclinic;

    comm_state.0 = CommState::CommunicateOnly;
    save_build_positions(&atoms, &mut neighbor);

    let ny = neighbor.bin_count[1];
    let nz = neighbor.bin_count[2];
    let nx = neighbor.bin_count[0];
    let total_cells = neighbor.bin_total_cells;
    let inv_bsx = 1.0 / neighbor.bin_size[0];
    let inv_bsy = 1.0 / neighbor.bin_size[1];
    let inv_bsz = 1.0 / neighbor.bin_size[2];
    let bin_ox = neighbor.bin_origin[0];
    let bin_oy = neighbor.bin_origin[1];
    let bin_oz = neighbor.bin_origin[2];

    // Step 1: Assign each atom to a bin cell via floor((pos - origin) / bin_size).
    // Reuse persistent arrays (taken via mem::take, returned at end) to avoid allocation.
    let mut atom_cell = std::mem::take(&mut neighbor.bin_atom_cell);
    atom_cell.clear();
    atom_cell.resize(total, 0u32);
    let mut bin_count_arr = std::mem::take(&mut neighbor.bin_count_arr);
    bin_count_arr.clear();
    bin_count_arr.resize(total_cells, 0u32);

    // N2: if `sort_atoms_by_bin` already computed the local atoms' cells this step
    // (same grid, same formula), reuse them and bin only the boundary ghosts;
    // otherwise bin every atom. Either way `bin_count_arr` ends up identical.
    let reuse_local = neighbor.local_cells_valid && neighbor.sorted_local_cell.len() == nlocal;
    let sorted_local_cell = std::mem::take(&mut neighbor.sorted_local_cell);
    let ghost_start = if reuse_local {
        // SAFETY: nlocal <= total; each cached cell is < total_cells (clamped at sort time).
        for i in 0..nlocal {
            let c = unsafe { *sorted_local_cell.get_unchecked(i) };
            unsafe {
                *atom_cell.get_unchecked_mut(i) = c;
                *bin_count_arr.get_unchecked_mut(c as usize) += 1;
            }
        }
        nlocal
    } else {
        0
    };
    // SAFETY: i < total = atoms.len(), cell is clamped to 0..total_cells by clamp on cx/cy/cz.
    for i in ghost_start..total {
        let pi_real = unsafe { *atoms.pos.get_unchecked(i) };
        // Triclinic: bin in fractional (lamda) coords (matches sort_atoms_by_bin).
        let pi_f = [pi_real[0] as f64, pi_real[1] as f64, pi_real[2] as f64];
        let pi = if triclinic {
            domain.x2lamda(pi_f)
        } else {
            pi_f
        };
        let cx = ((pi[0] - bin_ox) * inv_bsx).floor() as i32;
        let cy = ((pi[1] - bin_oy) * inv_bsy).floor() as i32;
        let cz = ((pi[2] - bin_oz) * inv_bsz).floor() as i32;
        let cx = cx.clamp(0, nx - 1);
        let cy = cy.clamp(0, ny - 1);
        let cz = cz.clamp(0, nz - 1);
        let cell = (cx * ny * nz + cy * nz + cz) as u32;
        unsafe {
            *atom_cell.get_unchecked_mut(i) = cell;
            *bin_count_arr.get_unchecked_mut(cell as usize) += 1;
        }
    }
    neighbor.sorted_local_cell = sorted_local_cell;
    neighbor.local_cells_valid = false;

    // Step 2: Build CSR bin offsets (reuse persistent array)
    let mut bin_start = std::mem::take(&mut neighbor.bin_start);
    bin_start.clear();
    bin_start.resize(total_cells + 1, 0u32);
    for c in 0..total_cells {
        bin_start[c + 1] = bin_start[c] + bin_count_arr[c];
    }

    // Place atoms into sorted order by bin — reuse bin_count_arr as write cursor
    // Also record each atom's position in sorted_atoms for self-cell skip optimization
    let mut sorted_atoms = std::mem::take(&mut neighbor.bin_sorted_atoms);
    sorted_atoms.clear();
    sorted_atoms.resize(total, 0u32);
    let mut atom_sorted_idx = std::mem::take(&mut neighbor.bin_atom_sorted_idx);
    atom_sorted_idx.clear();
    atom_sorted_idx.resize(total, 0u32);
    bin_count_arr[..total_cells].copy_from_slice(&bin_start[..total_cells]);
    for i in 0..total {
        let c = atom_cell[i] as usize;
        let pos = bin_count_arr[c];
        sorted_atoms[pos as usize] = i as u32;
        atom_sorted_idx[i] = pos;
        bin_count_arr[c] = pos + 1;
    }

    // Step 2b: Build sorted position cache for cache-friendly inner loop access
    let mut sorted_pos = std::mem::take(&mut neighbor.bin_sorted_pos);
    sorted_pos.clear();
    sorted_pos.resize(total, [0.0; 3]);
    for m in 0..total {
        // SAFETY: sorted_atoms[m] was populated from 0..total, so < atoms.pos.len().
        let p = unsafe {
            *atoms
                .pos
                .get_unchecked(*sorted_atoms.get_unchecked(m) as usize)
        };
        sorted_pos[m] = [p[0] as f64, p[1] as f64, p[2] as f64];
    }

    // Step 3: Build CSR neighbor lists using forward stencil.
    // Each pair is found exactly once: self cell uses j > i dedup,
    // forward cells have positive offset so each pair appears once.
    let has_self = neighbor.bin_stencil_self;
    let skin_fraction = neighbor.skin_fraction;
    let uniform_cutoff_sq = neighbor.cached_uniform_cutoff_sq;
    let newton = neighbor.newton;
    // Take stencil to avoid borrow conflict with neighbor_indices
    let stencil_forward = std::mem::take(&mut neighbor.bin_stencil_forward);
    let stencil_full = std::mem::take(&mut neighbor.bin_stencil);

    let prev_count = neighbor.neighbor_indices.len();
    // Work with local vecs to avoid borrow conflicts through ResMut
    let mut offsets = std::mem::take(&mut neighbor.neighbor_offsets);
    let mut indices = std::mem::take(&mut neighbor.neighbor_indices);
    offsets.clear();
    indices.clear();
    offsets.reserve(nlocal + 1);
    // Pre-allocate indices buffer generously; use unchecked writes with a counter
    let indices_cap = (prev_count + prev_count / 4).max(nlocal * 40);
    indices.reserve(indices_cap);
    let mut nidx: usize = 0;

    // Cache the buffer pointer + capacity so the hot inner loop's push does no
    // per-element field reads. They're refreshed only on the (rare) grow path;
    // `indices` is never read/written through its Vec API inside the loops, so the
    // cached `iptr` stays valid until the next reserve (which re-derives it).
    let mut cap = indices.capacity();
    let mut iptr = indices.as_mut_ptr();

    // Macro: bounds-check against cached capacity, write unchecked via cached ptr.
    macro_rules! push_index {
        ($val:expr) => {
            if nidx >= cap {
                // SAFETY: nidx == current logical length
                unsafe { indices.set_len(nidx) };
                indices.reserve(nidx / 2 + 256);
                cap = indices.capacity();
                iptr = indices.as_mut_ptr();
            }
            // SAFETY: nidx < cap after potential growth
            unsafe { *iptr.add(nidx) = $val };
            nidx += 1;
        };
    }

    // SAFETY invariants for all inner loops below:
    // - m ranges bin_start[c]..bin_start[c+1], both < total (prefix sum of counts summing to total)
    // - sorted_atoms[m] < total (populated from 0..total)
    // - sorted_pos[m] valid for same range as sorted_atoms
    // - c = my_cell + offset, where my_cell is clamped to valid bin range and offset is from stencil
    // - bin_start has total_cells + 1 entries; c and c+1 are within bounds by stencil construction
    // - Self-cell: start from atom_sorted_idx[i]+1 to skip all atoms at or before i in the bin.
    //   Within the same bin, local atoms appear in index order (counting sort preserves insertion
    //   order, and sort_atoms_by_bin pre-sorts locals by bin). Ghosts have index >= nlocal > i.
    if let Some(cutoff_sq) = uniform_cutoff_sq {
        // Fast path: all atoms have the same skin — skip per-pair skin load
        // Select stencil based on newton flag
        let stencil = if newton {
            &stencil_forward
        } else {
            &stencil_full
        };
        for i in 0..nlocal {
            offsets.push(nidx as u32);
            let my_cell = unsafe { *atom_cell.get_unchecked(i) } as usize;
            let pi = unsafe { *atoms.pos.get_unchecked(i) };
            let pi = [pi[0] as f64, pi[1] as f64, pi[2] as f64];

            if has_self {
                let cell_start = unsafe { *bin_start.get_unchecked(my_cell) } as usize;
                let end = unsafe { *bin_start.get_unchecked(my_cell + 1) } as usize;
                if newton {
                    // Half list: start after atom i's position — all subsequent atoms
                    // in this bin have j > i (locals in order, ghosts have index >= nlocal > i).
                    let self_start = unsafe { *atom_sorted_idx.get_unchecked(i) } as usize + 1;
                    for m in self_start..end {
                        let pj = unsafe { *sorted_pos.get_unchecked(m) };
                        let dx = pj[0] - pi[0];
                        let dy = pj[1] - pi[1];
                        let dz = pj[2] - pi[2];
                        let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                        if r2 < cutoff_sq {
                            let j = unsafe { *sorted_atoms.get_unchecked(m) };
                            push_index!(j);
                        }
                    }
                } else {
                    // Full list: scan all atoms in self-cell except i itself
                    for m in cell_start..end {
                        let j = unsafe { *sorted_atoms.get_unchecked(m) };
                        if j as usize == i {
                            continue;
                        }
                        let pj = unsafe { *sorted_pos.get_unchecked(m) };
                        let dx = pj[0] - pi[0];
                        let dy = pj[1] - pi[1];
                        let dz = pj[2] - pi[2];
                        let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                        if r2 < cutoff_sq {
                            push_index!(j);
                        }
                    }
                }
            }

            for &offset in stencil.iter() {
                // newton=true: stencil_forward has only offset > 0 (skip self cell)
                // newton=false: stencil_full includes offset <= 0 but self cell (0) handled above
                if !newton && offset == 0 {
                    continue;
                }
                let c = (my_cell as i32 + offset) as usize;
                let start = unsafe { *bin_start.get_unchecked(c) } as usize;
                let end = unsafe { *bin_start.get_unchecked(c + 1) } as usize;
                for m in start..end {
                    let pj = unsafe { *sorted_pos.get_unchecked(m) };
                    let dx = pj[0] - pi[0];
                    let dy = pj[1] - pi[1];
                    let dz = pj[2] - pi[2];
                    let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                    if r2 < cutoff_sq {
                        let j = unsafe { *sorted_atoms.get_unchecked(m) };
                        push_index!(j);
                    }
                }
            }
        }
    } else {
        // Slow path: per-pair skin computation
        let skin_fraction_sq = skin_fraction * skin_fraction;
        let stencil = if newton {
            &stencil_forward
        } else {
            &stencil_full
        };
        for i in 0..nlocal {
            offsets.push(nidx as u32);
            let my_cell = unsafe { *atom_cell.get_unchecked(i) } as usize;
            let pi = unsafe { *atoms.pos.get_unchecked(i) };
            let pi = [pi[0] as f64, pi[1] as f64, pi[2] as f64];
            let si = unsafe { *atoms.cutoff_radius.get_unchecked(i) };

            if has_self {
                let cell_start = unsafe { *bin_start.get_unchecked(my_cell) } as usize;
                let end = unsafe { *bin_start.get_unchecked(my_cell + 1) } as usize;
                if newton {
                    let self_start = unsafe { *atom_sorted_idx.get_unchecked(i) } as usize + 1;
                    for m in self_start..end {
                        let j = unsafe { *sorted_atoms.get_unchecked(m) } as usize;
                        let pj = unsafe { *sorted_pos.get_unchecked(m) };
                        let dx = pj[0] - pi[0];
                        let dy = pj[1] - pi[1];
                        let dz = pj[2] - pi[2];
                        let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                        let sum_skin =
                            si as f64 + unsafe { *atoms.cutoff_radius.get_unchecked(j) } as f64;
                        if r2 < sum_skin * sum_skin * skin_fraction_sq {
                            push_index!(j as u32);
                        }
                    }
                } else {
                    for m in cell_start..end {
                        let j = unsafe { *sorted_atoms.get_unchecked(m) } as usize;
                        if j == i {
                            continue;
                        }
                        let pj = unsafe { *sorted_pos.get_unchecked(m) };
                        let dx = pj[0] - pi[0];
                        let dy = pj[1] - pi[1];
                        let dz = pj[2] - pi[2];
                        let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                        let sum_skin =
                            si as f64 + unsafe { *atoms.cutoff_radius.get_unchecked(j) } as f64;
                        if r2 < sum_skin * sum_skin * skin_fraction_sq {
                            push_index!(j as u32);
                        }
                    }
                }
            }

            for &offset in stencil.iter() {
                if !newton && offset == 0 {
                    continue;
                }
                let c = (my_cell as i32 + offset) as usize;
                let start = unsafe { *bin_start.get_unchecked(c) } as usize;
                let end = unsafe { *bin_start.get_unchecked(c + 1) } as usize;
                for m in start..end {
                    let j = unsafe { *sorted_atoms.get_unchecked(m) } as usize;
                    let pj = unsafe { *sorted_pos.get_unchecked(m) };
                    let dx = pj[0] - pi[0];
                    let dy = pj[1] - pi[1];
                    let dz = pj[2] - pi[2];
                    let r2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                    let sum_skin =
                        si as f64 + unsafe { *atoms.cutoff_radius.get_unchecked(j) } as f64;
                    if r2 < sum_skin * sum_skin * skin_fraction_sq {
                        push_index!(j as u32);
                    }
                }
            }
        }
    }
    // SAFETY: nidx elements were written via raw pointer; all < capacity.
    unsafe { indices.set_len(nidx) };
    offsets.push(nidx as u32);

    neighbor.neighbor_offsets = offsets;
    neighbor.neighbor_indices = indices;
    neighbor.bin_stencil_forward = stencil_forward;
    neighbor.bin_stencil = stencil_full;
    neighbor.bin_atom_cell = atom_cell;
    neighbor.bin_count_arr = bin_count_arr;
    neighbor.bin_start = bin_start;
    neighbor.bin_sorted_atoms = sorted_atoms;
    neighbor.bin_atom_sorted_idx = atom_sorted_idx;
    neighbor.bin_sorted_pos = sorted_pos;
}
