//! Generic GPU cell-list neighbor build (substrate; no physics).
//!
//! Builds an O(N) cell list on the device — bin → atomic count → exclusive
//! prefix-sum → scatter — exposing `cell_start` (CSR offsets), `sorted_atoms`,
//! and `atom_cell` as device buffers a downstream force/bond kernel can bind.
//! Reusable by any particle method (DEM, MD, peridynamics).

use bytemuck::{Pod, Zeroable};

use crate::GpuContext;

/// Uniform-grid parameters, computed on the host from the particle AABB.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub n: [i32; 3],
    pub origin: [f32; 3],
    pub bin_size: f32,
    pub total_cells: usize,
}

impl Grid {
    /// Grid whose cell size equals the interaction cutoff, so any interacting
    /// pair lies within a ±1 cell stencil. One ghost cell of padding per side.
    pub fn from_positions(pos: &[[f32; 3]], cutoff: f32) -> Self {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for p in pos {
            for d in 0..3 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        let bin_size = cutoff.max(f32::MIN_POSITIVE);
        let mut n = [1i32; 3];
        let mut origin = [0.0f32; 3];
        for d in 0..3 {
            let o = lo[d] - bin_size;
            let cells = (((hi[d] + bin_size) - o) / bin_size).ceil() as i32;
            n[d] = cells.max(1);
            origin[d] = o;
        }
        let total_cells = (n[0] * n[1] * n[2]) as usize;
        Grid { n, origin, bin_size, total_cells }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    total_cells: u32,
    _p0: u32,
    _p1: u32,
    nx: i32,
    ny: i32,
    nz: i32,
    _p2: i32,
    ox: f32,
    oy: f32,
    oz: f32,
    _p3: f32,
    inv_bx: f32,
    inv_by: f32,
    inv_bz: f32,
    _p4: f32,
}

/// GPU cell-list builder. Holds the position input + cell-list output buffers.
/// `pos`, `cell_start`, `sorted_atoms`, `atom_cell` are exposed so a consumer
/// kernel (force/bond) can bind them in its own bind group.
#[allow(dead_code)]
pub struct CellList {
    ctx: GpuContext,
    n: usize,
    total_cells: usize,
    pos: wgpu::Buffer,
    atom_cell: wgpu::Buffer,
    cell_count: wgpu::Buffer,
    cell_start: wgpu::Buffer,
    cursor: wgpu::Buffer,
    sorted_atoms: wgpu::Buffer,
    params: wgpu::Buffer,
    staging_u32: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    p_clear: wgpu::ComputePipeline,
    p_assign: wgpu::ComputePipeline,
    p_prefix: wgpu::ComputePipeline,
    p_scatter: wgpu::ComputePipeline,
    p_sort: wgpu::ComputePipeline,
}

const WG: u32 = 64;

impl CellList {
    pub fn new(ctx: GpuContext, n: usize, total_cells: usize) -> Self {
        let device = &ctx.device;
        let nz = n.max(1);
        let tcz = total_cells.max(1);
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let mk = |label: &str, size: u64, usage| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size, usage, mapped_at_creation: false,
        });
        // pos is rw (COPY_SRC) so a resident owner can integrate it in place and
        // read it back; the cell-list kernels only read it (read_only binding).
        let pos = mk("cl_pos", (nz * 3 * 4) as u64, rw);
        let atom_cell = mk("cl_atom_cell", (nz * 4) as u64, rw);
        let cell_count = mk("cl_cell_count", (tcz * 4) as u64, rw);
        let cell_start = mk("cl_cell_start", ((tcz + 1) * 4) as u64, rw);
        let cursor = mk("cl_cursor", (tcz * 4) as u64, rw);
        let sorted_atoms = mk("cl_sorted_atoms", (nz * 4) as u64, rw);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cl_params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_u32 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cl_staging_u32"), size: ((tcz + 1).max(nz) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cell_list"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cell_list.wgsl").into()),
        });
        let buf = |binding: u32, read_only: bool, uniform: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if uniform { wgpu::BufferBindingType::Uniform }
                    else { wgpu::BufferBindingType::Storage { read_only } },
                has_dynamic_offset: false, min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cl bgl"),
            entries: &[
                buf(0, true, false), buf(1, false, true), buf(2, false, false),
                buf(3, false, false), buf(4, false, false), buf(5, false, false),
                buf(6, false, false),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cl bg"), layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pos.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: atom_cell.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cell_count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: cell_start.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: cursor.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: sorted_atoms.as_entire_binding() },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cl pl"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0,
        });
        let mk_pipe = |entry: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry), layout: Some(&layout), module: &shader,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });
        Self {
            p_clear: mk_pipe("clear_cells"),
            p_assign: mk_pipe("assign_cells"),
            p_prefix: mk_pipe("prefix_sum"),
            p_scatter: mk_pipe("scatter"),
            p_sort: mk_pipe("sort_cells"),
            ctx, n, total_cells, pos, atom_cell, cell_count, cell_start, cursor,
            sorted_atoms, params, staging_u32, bind_group,
        }
    }

    /// Upload positions + grid params (the position buffer is also exposed for a
    /// resident consumer that writes positions directly).
    pub fn upload_positions(&self, pos: &[[f32; 3]], grid: Grid) {
        assert_eq!(pos.len(), self.n);
        assert_eq!(grid.total_cells, self.total_cells);
        let q = &self.ctx.queue;
        q.write_buffer(&self.pos, 0, bytemuck::cast_slice(pos));
        let inv_b = 1.0 / grid.bin_size;
        let p = Params {
            n: self.n as u32, total_cells: self.total_cells as u32, _p0: 0, _p1: 0,
            nx: grid.n[0], ny: grid.n[1], nz: grid.n[2], _p2: 0,
            ox: grid.origin[0], oy: grid.origin[1], oz: grid.origin[2], _p3: 0.0,
            inv_bx: inv_b, inv_by: inv_b, inv_bz: inv_b, _p4: 0.0,
        };
        q.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
    }

    /// Set the grid params only (no position upload). For a resident owner that
    /// integrates `pos` on-device: set the (fixed) grid once, then `record` each
    /// step rebins the device-resident positions without any host transfer.
    pub fn set_grid(&self, grid: Grid) {
        assert_eq!(grid.total_cells, self.total_cells);
        let inv_b = 1.0 / grid.bin_size;
        let p = Params {
            n: self.n as u32, total_cells: self.total_cells as u32, _p0: 0, _p1: 0,
            nx: grid.n[0], ny: grid.n[1], nz: grid.n[2], _p2: 0,
            ox: grid.origin[0], oy: grid.origin[1], oz: grid.origin[2], _p3: 0.0,
            inv_bx: inv_b, inv_by: inv_b, inv_bz: inv_b, _p4: 0.0,
        };
        self.ctx.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
    }

    /// Record the build (clear → assign → prefix → scatter) into a compute pass.
    /// A resident loop calls this each step before its force kernel.
    pub fn record(&self, pass: &mut wgpu::ComputePass) {
        pass.set_bind_group(0, &self.bind_group, &[]);
        let cells = (self.total_cells as u32).div_ceil(WG).max(1);
        let atoms = (self.n as u32).div_ceil(WG).max(1);
        pass.set_pipeline(&self.p_clear);
        pass.dispatch_workgroups(cells, 1, 1);
        pass.set_pipeline(&self.p_assign);
        pass.dispatch_workgroups(atoms, 1, 1);
        pass.set_pipeline(&self.p_prefix);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.p_scatter);
        pass.dispatch_workgroups(atoms, 1, 1);
        // Deterministic within-cell ordering (one thread per cell) so the neighbour
        // traversal is reproducible — removes the atomic-scatter race that made
        // windowed/per-step runs diverge from a single window.
        pass.set_pipeline(&self.p_sort);
        pass.dispatch_workgroups(cells, 1, 1);
    }

    /// Convenience: upload, build (one submit), block. For one-shot / testing.
    pub fn build(&self, pos: &[[f32; 3]], grid: Grid) {
        self.upload_positions(pos, grid);
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cell_list build"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cell_list build"), timestamp_writes: None,
            });
            self.record(&mut pass);
        }
        self.ctx.queue.submit(Some(enc.finish()));
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    }

    /// Exposed buffer handles for a downstream consumer kernel to bind.
    pub fn pos_buffer(&self) -> &wgpu::Buffer { &self.pos }
    pub fn cell_start_buffer(&self) -> &wgpu::Buffer { &self.cell_start }
    pub fn sorted_atoms_buffer(&self) -> &wgpu::Buffer { &self.sorted_atoms }
    pub fn atom_cell_buffer(&self) -> &wgpu::Buffer { &self.atom_cell }

    pub fn download_sorted_atoms(&self) -> Vec<u32> { self.download_u32(&self.sorted_atoms, self.n) }
    pub fn download_cell_start(&self) -> Vec<u32> { self.download_u32(&self.cell_start, self.total_cells + 1) }
    pub fn download_atom_cell(&self) -> Vec<u32> { self.download_u32(&self.atom_cell, self.n) }

    fn download_u32(&self, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
        let bytes = (count * 4) as u64;
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cl dl"),
        });
        enc.copy_buffer_to_buffer(buf, 0, &self.staging_u32, 0, bytes);
        self.ctx.queue.submit(Some(enc.finish()));
        let slice = self.staging_u32.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range();
        let v: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_u32.unmap();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_list_build_is_valid_permutation_and_binning() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let n = 512usize;
        let r = 0.5f32;
        let cutoff = 2.0 * r;
        let spacing = 0.9f32;
        let side = (n as f64).cbrt().ceil() as usize;
        let mut pos = Vec::with_capacity(n);
        for k in 0..n {
            let (ix, iy, iz) = (k % side, (k / side) % side, k / (side * side));
            let f = k as f64;
            pos.push([
                ix as f32 * spacing + (0.13 * f).sin() as f32 * 0.05,
                iy as f32 * spacing + (0.27 * f).cos() as f32 * 0.05,
                iz as f32 * spacing + (0.41 * f).sin() as f32 * 0.05,
            ]);
        }
        let grid = Grid::from_positions(&pos, cutoff);
        let cl = CellList::new(ctx, n, grid.total_cells);
        cl.build(&pos, grid);

        let sorted = cl.download_sorted_atoms();
        let cell_start = cl.download_cell_start();
        let atom_cell = cl.download_atom_cell();

        // cell_start ends at n; sorted_atoms is a permutation of 0..n.
        assert_eq!(cell_start[grid.total_cells] as usize, n);
        let mut seen = vec![false; n];
        for &a in &sorted {
            assert!((a as usize) < n);
            assert!(!seen[a as usize], "atom {a} placed twice");
            seen[a as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));

        // Every atom sits in the cell its position maps to (CPU recompute).
        // Recompute in f32 to match the GPU's f32 binning exactly (f64 here would
        // disagree by one cell for atoms sitting on a cell boundary).
        let inv_b = 1.0f32 / grid.bin_size;
        let o = grid.origin;
        let (nx, ny, nz) = (grid.n[0] as i64, grid.n[1] as i64, grid.n[2] as i64);
        for i in 0..n {
            let cx = (((pos[i][0] - o[0]) * inv_b).floor() as i64).clamp(0, nx - 1);
            let cy = (((pos[i][1] - o[1]) * inv_b).floor() as i64).clamp(0, ny - 1);
            let cz = (((pos[i][2] - o[2]) * inv_b).floor() as i64).clamp(0, nz - 1);
            let expect = ((cx * ny + cy) * nz + cz) as u32;
            assert_eq!(atom_cell[i], expect, "atom {i} cell mismatch");
        }
        // sorted_atoms grouped by cell: each atom lands within its cell's range.
        for c in 0..grid.total_cells {
            for m in cell_start[c]..cell_start[c + 1] {
                assert_eq!(atom_cell[sorted[m as usize] as usize] as usize, c);
            }
        }
        eprintln!("cell_list: {n} atoms, {} cells — valid permutation + binning", grid.total_cells);
    }
}
