//! Bin-based CSR neighbor-list construction and distance filters.

use grass_scheduler::prelude::*;

use super::rebuild::save_build_positions;
use super::Neighbor;
use crate::{Atom, CommState, Domain};

struct NeighborIndexWriter {
    ptr: *mut u32,
    cap: usize,
    len: usize,
}

impl NeighborIndexWriter {
    fn new(ptr: *mut u32, cap: usize) -> Self {
        Self { ptr, cap, len: 0 }
    }

    #[inline(always)]
    unsafe fn push(&mut self, value: u32) {
        debug_assert!(self.len < self.cap);
        *self.ptr.add(self.len) = value;
        self.len += 1;
    }
}

#[inline(always)]
unsafe fn append_uniform_range_scalar(
    start: usize,
    end: usize,
    pi: [f64; 3],
    sorted_pos: &[[f64; 3]],
    sorted_atoms: &[u32],
    cutoff_sq: f64,
    writer: &mut NeighborIndexWriter,
) {
    for m in start..end {
        let pj = *sorted_pos.get_unchecked(m);
        let dx = pj[0] - pi[0];
        let dy = pj[1] - pi[1];
        let dz = pj[2] - pi[2];
        // Plain multiply-add (not `mul_add`): matches the AVX2 kernel bit for
        // bit and avoids a libm `fma` call per candidate on non-FMA targets.
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 < cutoff_sq {
            writer.push(*sorted_atoms.get_unchecked(m));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn append_uniform_range_avx2(
    start: usize,
    end: usize,
    pi: [f64; 3],
    sorted_pos: &[[f64; 3]],
    sorted_atoms: &[u32],
    cutoff_sq: f64,
    writer: &mut NeighborIndexWriter,
) {
    use std::arch::x86_64::*;

    let px = _mm256_set1_pd(pi[0]);
    let py = _mm256_set1_pd(pi[1]);
    let pz = _mm256_set1_pd(pi[2]);
    let cutoff = _mm256_set1_pd(cutoff_sq);
    let mut m = start;
    while m + 4 <= end {
        let p0 = *sorted_pos.get_unchecked(m);
        let p1 = *sorted_pos.get_unchecked(m + 1);
        let p2 = *sorted_pos.get_unchecked(m + 2);
        let p3 = *sorted_pos.get_unchecked(m + 3);
        let x = _mm256_set_pd(p3[0], p2[0], p1[0], p0[0]);
        let y = _mm256_set_pd(p3[1], p2[1], p1[1], p0[1]);
        let z = _mm256_set_pd(p3[2], p2[2], p1[2], p0[2]);
        let dx = _mm256_sub_pd(x, px);
        let dy = _mm256_sub_pd(y, py);
        let dz = _mm256_sub_pd(z, pz);
        let r2 = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
            _mm256_mul_pd(dz, dz),
        );
        let mask = _mm256_movemask_pd(_mm256_cmp_pd(r2, cutoff, _CMP_LT_OQ));
        for lane in 0..4 {
            if (mask & (1 << lane)) != 0 {
                writer.push(*sorted_atoms.get_unchecked(m + lane));
            }
        }
        m += 4;
    }
    append_uniform_range_scalar(m, end, pi, sorted_pos, sorted_atoms, cutoff_sq, writer);
}

#[inline(always)]
unsafe fn append_uniform_range(
    start: usize,
    end: usize,
    pi: [f64; 3],
    sorted_pos: &[[f64; 3]],
    sorted_atoms: &[u32],
    cutoff_sq: f64,
    use_simd: bool,
    writer: &mut NeighborIndexWriter,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd {
            append_uniform_range_avx2(start, end, pi, sorted_pos, sorted_atoms, cutoff_sq, writer);
            return;
        }
    }
    let _ = use_simd;
    append_uniform_range_scalar(start, end, pi, sorted_pos, sorted_atoms, cutoff_sq, writer);
}

#[inline(always)]
unsafe fn append_polydisperse_range_scalar(
    start: usize,
    end: usize,
    pi: [f64; 3],
    si: f64,
    skin_fraction_sq: f64,
    sorted_pos: &[[f64; 3]],
    sorted_atoms: &[u32],
    cutoff_radius: &[crate::precision::Real],
    writer: &mut NeighborIndexWriter,
) {
    for m in start..end {
        let j = *sorted_atoms.get_unchecked(m) as usize;
        let pj = *sorted_pos.get_unchecked(m);
        let dx = pj[0] - pi[0];
        let dy = pj[1] - pi[1];
        let dz = pj[2] - pi[2];
        // Plain multiply-add (not `mul_add`): matches the AVX2 kernel bit for
        // bit and avoids a libm `fma` call per candidate on non-FMA targets.
        let r2 = dx * dx + dy * dy + dz * dz;
        let sum_skin = si + *cutoff_radius.get_unchecked(j) as f64;
        if r2 < sum_skin * sum_skin * skin_fraction_sq {
            writer.push(j as u32);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn append_polydisperse_range_avx2(
    start: usize,
    end: usize,
    pi: [f64; 3],
    si: f64,
    skin_fraction_sq: f64,
    sorted_pos: &[[f64; 3]],
    sorted_atoms: &[u32],
    cutoff_radius: &[crate::precision::Real],
    writer: &mut NeighborIndexWriter,
) {
    use std::arch::x86_64::*;

    let px = _mm256_set1_pd(pi[0]);
    let py = _mm256_set1_pd(pi[1]);
    let pz = _mm256_set1_pd(pi[2]);
    let si_scalar = si;
    let si = _mm256_set1_pd(si_scalar);
    let skin = _mm256_set1_pd(skin_fraction_sq);
    let mut m = start;
    while m + 4 <= end {
        let p0 = *sorted_pos.get_unchecked(m);
        let p1 = *sorted_pos.get_unchecked(m + 1);
        let p2 = *sorted_pos.get_unchecked(m + 2);
        let p3 = *sorted_pos.get_unchecked(m + 3);
        let j0 = *sorted_atoms.get_unchecked(m) as usize;
        let j1 = *sorted_atoms.get_unchecked(m + 1) as usize;
        let j2 = *sorted_atoms.get_unchecked(m + 2) as usize;
        let j3 = *sorted_atoms.get_unchecked(m + 3) as usize;
        let x = _mm256_set_pd(p3[0], p2[0], p1[0], p0[0]);
        let y = _mm256_set_pd(p3[1], p2[1], p1[1], p0[1]);
        let z = _mm256_set_pd(p3[2], p2[2], p1[2], p0[2]);
        let sj = _mm256_set_pd(
            *cutoff_radius.get_unchecked(j3) as f64,
            *cutoff_radius.get_unchecked(j2) as f64,
            *cutoff_radius.get_unchecked(j1) as f64,
            *cutoff_radius.get_unchecked(j0) as f64,
        );
        let dx = _mm256_sub_pd(x, px);
        let dy = _mm256_sub_pd(y, py);
        let dz = _mm256_sub_pd(z, pz);
        let r2 = _mm256_add_pd(
            _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy)),
            _mm256_mul_pd(dz, dz),
        );
        let sum = _mm256_add_pd(si, sj);
        let cutoff = _mm256_mul_pd(_mm256_mul_pd(sum, sum), skin);
        let mask = _mm256_movemask_pd(_mm256_cmp_pd(r2, cutoff, _CMP_LT_OQ));
        for lane in 0..4 {
            if (mask & (1 << lane)) != 0 {
                writer.push(*sorted_atoms.get_unchecked(m + lane));
            }
        }
        m += 4;
    }
    append_polydisperse_range_scalar(
        m,
        end,
        pi,
        si_scalar,
        skin_fraction_sq,
        sorted_pos,
        sorted_atoms,
        cutoff_radius,
        writer,
    );
}

#[inline(always)]
unsafe fn append_polydisperse_range(
    start: usize,
    end: usize,
    pi: [f64; 3],
    si: f64,
    skin_fraction_sq: f64,
    sorted_pos: &[[f64; 3]],
    sorted_atoms: &[u32],
    cutoff_radius: &[crate::precision::Real],
    use_simd: bool,
    writer: &mut NeighborIndexWriter,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if use_simd {
            append_polydisperse_range_avx2(
                start,
                end,
                pi,
                si,
                skin_fraction_sq,
                sorted_pos,
                sorted_atoms,
                cutoff_radius,
                writer,
            );
            return;
        }
    }
    let _ = use_simd;
    append_polydisperse_range_scalar(
        start,
        end,
        pi,
        si,
        skin_fraction_sq,
        sorted_pos,
        sorted_atoms,
        cutoff_radius,
        writer,
    );
}

/// Compute bin grid parameters (count, size, origin, stencil) from domain bounds and cutoff.
///
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
        let cx = super::floor_to_i32((pi[0] - bin_ox) * inv_bsx);
        let cy = super::floor_to_i32((pi[1] - bin_oy) * inv_bsy);
        let cz = super::floor_to_i32((pi[2] - bin_oz) * inv_bsz);
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

    // Bins-only mode (`[neighbor] pairs = false`): the per-bin CSR is the
    // product. Leave an empty, well-formed pair list and return.
    if !neighbor.build_pairs {
        let mut offsets = std::mem::take(&mut neighbor.neighbor_offsets);
        offsets.clear();
        offsets.resize(nlocal + 1, 0);
        neighbor.neighbor_offsets = offsets;
        neighbor.neighbor_indices.clear();
        neighbor.bin_atom_cell = atom_cell;
        neighbor.bin_count_arr = bin_count_arr;
        neighbor.bin_start = bin_start;
        neighbor.bin_sorted_atoms = sorted_atoms;
        neighbor.bin_atom_sorted_idx = atom_sorted_idx;
        return;
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
    let stencil_nonself = std::mem::take(&mut neighbor.bin_stencil_nonself);

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
    // Cache the buffer pointer + capacity so the hot inner loop's push does no
    // per-element field reads. They're refreshed only on the (rare) grow path;
    // `indices` is never read/written through its Vec API inside the loops, so the
    // cached `iptr` stays valid until the next reserve (which re-derives it).
    let mut cap = indices.capacity();

    // Reserve for all possible candidates once before distance filtering. This
    // keeps Vec growth checks out of accepted-neighbor loops; the actual accepted
    // pair count is always <= this conservative candidate count.
    let reserve_stencil = if newton {
        &stencil_forward
    } else {
        &stencil_nonself
    };
    let mut candidate_capacity = 0usize;
    for i in 0..nlocal {
        let my_cell = unsafe { *atom_cell.get_unchecked(i) } as usize;
        if has_self {
            let cell_start = unsafe { *bin_start.get_unchecked(my_cell) } as usize;
            let end = unsafe { *bin_start.get_unchecked(my_cell + 1) } as usize;
            if newton {
                let self_start = unsafe { *atom_sorted_idx.get_unchecked(i) } as usize + 1;
                candidate_capacity += end.saturating_sub(self_start);
            } else {
                candidate_capacity += end - cell_start;
            }
        }
        for &offset in reserve_stencil.iter() {
            let c = (my_cell as i32 + offset) as usize;
            let start = unsafe { *bin_start.get_unchecked(c) } as usize;
            let end = unsafe { *bin_start.get_unchecked(c + 1) } as usize;
            candidate_capacity += end - start;
        }
    }
    if candidate_capacity > cap {
        indices.reserve(candidate_capacity);
        cap = indices.capacity();
    }

    let mut writer = NeighborIndexWriter::new(indices.as_mut_ptr(), cap);
    let use_simd = neighbor.distance_kernel.simd_available();

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
            &stencil_nonself
        };
        for i in 0..nlocal {
            offsets.push(writer.len as u32);
            let my_cell = unsafe { *atom_cell.get_unchecked(i) } as usize;
            let self_pos = unsafe { *atom_sorted_idx.get_unchecked(i) } as usize;
            let pi = unsafe { *sorted_pos.get_unchecked(self_pos) };

            if has_self {
                let cell_start = unsafe { *bin_start.get_unchecked(my_cell) } as usize;
                let end = unsafe { *bin_start.get_unchecked(my_cell + 1) } as usize;
                if newton {
                    // Half list: start after atom i's position — all subsequent atoms
                    // in this bin have j > i (locals in order, ghosts have index >= nlocal > i).
                    let self_start = unsafe { *atom_sorted_idx.get_unchecked(i) } as usize + 1;
                    unsafe {
                        append_uniform_range(
                            self_start,
                            end,
                            pi,
                            &sorted_pos,
                            &sorted_atoms,
                            cutoff_sq,
                            use_simd,
                            &mut writer,
                        );
                    }
                } else {
                    // Full list: scan both sides of i's sorted position. This
                    // produces the same self-cell candidates without testing
                    // `j == i` for every atom in the cell.
                    unsafe {
                        append_uniform_range(
                            cell_start,
                            self_pos,
                            pi,
                            &sorted_pos,
                            &sorted_atoms,
                            cutoff_sq,
                            use_simd,
                            &mut writer,
                        );
                        append_uniform_range(
                            self_pos + 1,
                            end,
                            pi,
                            &sorted_pos,
                            &sorted_atoms,
                            cutoff_sq,
                            use_simd,
                            &mut writer,
                        );
                    }
                }
            }

            for &offset in stencil.iter() {
                let c = (my_cell as i32 + offset) as usize;
                let start = unsafe { *bin_start.get_unchecked(c) } as usize;
                let end = unsafe { *bin_start.get_unchecked(c + 1) } as usize;
                unsafe {
                    append_uniform_range(
                        start,
                        end,
                        pi,
                        &sorted_pos,
                        &sorted_atoms,
                        cutoff_sq,
                        use_simd,
                        &mut writer,
                    );
                }
            }
        }
    } else {
        // Slow path: per-pair skin computation
        let skin_fraction_sq = skin_fraction * skin_fraction;
        let stencil = if newton {
            &stencil_forward
        } else {
            &stencil_nonself
        };
        for i in 0..nlocal {
            offsets.push(writer.len as u32);
            let my_cell = unsafe { *atom_cell.get_unchecked(i) } as usize;
            let self_pos = unsafe { *atom_sorted_idx.get_unchecked(i) } as usize;
            let pi = unsafe { *sorted_pos.get_unchecked(self_pos) };
            let si = unsafe { *atoms.cutoff_radius.get_unchecked(i) } as f64;

            if has_self {
                let cell_start = unsafe { *bin_start.get_unchecked(my_cell) } as usize;
                let end = unsafe { *bin_start.get_unchecked(my_cell + 1) } as usize;
                if newton {
                    let self_start = unsafe { *atom_sorted_idx.get_unchecked(i) } as usize + 1;
                    unsafe {
                        append_polydisperse_range(
                            self_start,
                            end,
                            pi,
                            si,
                            skin_fraction_sq,
                            &sorted_pos,
                            &sorted_atoms,
                            &atoms.cutoff_radius,
                            use_simd,
                            &mut writer,
                        );
                    }
                } else {
                    unsafe {
                        append_polydisperse_range(
                            cell_start,
                            self_pos,
                            pi,
                            si,
                            skin_fraction_sq,
                            &sorted_pos,
                            &sorted_atoms,
                            &atoms.cutoff_radius,
                            use_simd,
                            &mut writer,
                        );
                        append_polydisperse_range(
                            self_pos + 1,
                            end,
                            pi,
                            si,
                            skin_fraction_sq,
                            &sorted_pos,
                            &sorted_atoms,
                            &atoms.cutoff_radius,
                            use_simd,
                            &mut writer,
                        );
                    }
                }
            }

            for &offset in stencil.iter() {
                let c = (my_cell as i32 + offset) as usize;
                let start = unsafe { *bin_start.get_unchecked(c) } as usize;
                let end = unsafe { *bin_start.get_unchecked(c + 1) } as usize;
                unsafe {
                    append_polydisperse_range(
                        start,
                        end,
                        pi,
                        si,
                        skin_fraction_sq,
                        &sorted_pos,
                        &sorted_atoms,
                        &atoms.cutoff_radius,
                        use_simd,
                        &mut writer,
                    );
                }
            }
        }
    }
    // SAFETY: nidx elements were written via raw pointer; all < capacity.
    let nidx = writer.len;
    unsafe { indices.set_len(nidx) };
    offsets.push(nidx as u32);

    neighbor.neighbor_offsets = offsets;
    neighbor.neighbor_indices = indices;
    neighbor.bin_stencil_forward = stencil_forward;
    neighbor.bin_stencil = stencil_full;
    neighbor.bin_stencil_nonself = stencil_nonself;
    neighbor.bin_atom_cell = atom_cell;
    neighbor.bin_count_arr = bin_count_arr;
    neighbor.bin_start = bin_start;
    neighbor.bin_sorted_atoms = sorted_atoms;
    neighbor.bin_atom_sorted_idx = atom_sorted_idx;
    neighbor.bin_sorted_pos = sorted_pos;
}
