//! Bin-grid and stencil construction.

use super::Neighbor;
use crate::Domain;

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
    // When ghost_cutoff > pair cutoff (e.g., a method extends communication beyond
    // the pair distance), local atoms can end up in ghost bins. The stencil from
    // those bins must not exceed the grid, so we add stencil range on top of ghost
    // layers.
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
    neighbor.bin_stencil_nonself.clear();
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
                        neighbor.bin_stencil_nonself.push(offset);
                    } else if offset == 0 {
                        neighbor.bin_stencil_self = true;
                    } else {
                        neighbor.bin_stencil_nonself.push(offset);
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
    neighbor.bin_stencil_nonself.clear();
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
                        neighbor.bin_stencil_nonself.push(offset);
                    } else if offset == 0 {
                        neighbor.bin_stencil_self = true;
                    } else {
                        neighbor.bin_stencil_nonself.push(offset);
                    }
                }
            }
        }
    }

    neighbor.pbc_box = [l[0], l[1], l[2]];
}

/// Recompute bin grid parameters after domain bounds change (e.g., shrink-wrap).
pub(super) fn recompute_bins(neighbor: &mut Neighbor, domain: &Domain, comm_size: i32) {
    if neighbor.cached_max_cutoff <= 0.0 {
        return; // not yet initialized
    }
    compute_bin_grid(neighbor, domain, comm_size);
}
