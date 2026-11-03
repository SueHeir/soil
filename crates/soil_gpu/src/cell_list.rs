//! Cell-list neighbor build on the device (substrate; no physics).
//!
//! bin -> atomic count -> exclusive prefix sum -> scatter -> per-cell sort, exposing
//! `cell_start` (CSR offsets), `sorted_atoms`, and `atom_cell` for a downstream force or
//! bond kernel to read. Reusable by any particle method.

use cubecl::prelude::*;
use cubecl::server::Handle;
use grass_gpu::cubecl;
use grass_gpu::{GpuDevice, GpuReal};

/// Uniform-grid parameters, computed on the host from the particle bounds.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub n: [i32; 3],
    pub origin: [GpuReal; 3],
    /// Cell width on each axis.
    pub bin_size: [GpuReal; 3],
    pub total_cells: usize,
}

impl Grid {
    /// Grid whose cell size equals the interaction cutoff, so any interacting pair lies
    /// within a +/-1 cell stencil, with one cell of padding per side. `pos` is flat xyz.
    pub fn from_positions(pos: &[GpuReal], cutoff: GpuReal) -> Self {
        let mut lo = [GpuReal::MAX; 3];
        let mut hi = [GpuReal::MIN; 3];
        for p in pos.chunks_exact(3) {
            for d in 0..3 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        let bin_size = cutoff.max(GpuReal::MIN_POSITIVE);
        let mut n = [1i32; 3];
        let mut origin = [0.0; 3];
        for d in 0..3 {
            let o = lo[d] - bin_size;
            let cells = (((hi[d] + bin_size) - o) / bin_size).ceil() as i32;
            n[d] = cells.max(1);
            origin[d] = o;
        }
        let total_cells = (n[0] * n[1] * n[2]) as usize;
        Grid {
            n,
            origin,
            bin_size: [bin_size; 3],
            total_cells,
        }
    }

    /// [`from_positions`](Self::from_positions) with at most `max_cells` cells. A wider cell
    /// still finds every pair within `cutoff`; it only checks more candidates. This keeps
    /// memory bounded when one far-flung particle stretches the bounding box.
    pub fn from_positions_capped(pos: &[GpuReal], cutoff: GpuReal, max_cells: usize) -> Self {
        let mut cell = cutoff;
        loop {
            let grid = Self::from_positions(pos, cell);
            if grid.total_cells <= max_cells {
                return grid;
            }
            cell *= 1.25;
        }
    }

    /// A grid for a box where some axes wrap. On a periodic axis the cells tile `lo..hi`
    /// exactly, so the stencil's wrap lands on the box period. Other axes cover `pos`,
    /// padded by one cell. Cells are at least `cutoff` wide, wider if `max_cells` needs it.
    ///
    /// Panics if a periodic axis is under three cells wide, since the stencil would then
    /// visit one cell twice and an atom would interact with its own image.
    pub fn for_box(
        lo: [GpuReal; 3],
        hi: [GpuReal; 3],
        periodic: [bool; 3],
        pos: &[GpuReal],
        cutoff: GpuReal,
        max_cells: usize,
    ) -> Self {
        let mut cell = cutoff.max(GpuReal::MIN_POSITIVE);
        loop {
            let mut grid = Self::from_positions(pos, cell);
            for d in 0..3 {
                if periodic[d] {
                    let length = hi[d] - lo[d];
                    let cells = (length / cell).floor() as i32;
                    assert!(
                        cells >= 3 || cell > cutoff,
                        "soil_gpu: periodic axis {d} is {length} long, under three contact cutoffs ({cutoff})"
                    );
                    grid.n[d] = cells.max(3);
                    grid.origin[d] = lo[d];
                    grid.bin_size[d] = length / grid.n[d] as GpuReal;
                }
            }
            grid.total_cells = (grid.n[0] * grid.n[1] * grid.n[2]) as usize;
            if grid.total_cells <= max_cells || (0..3).all(|d| periodic[d] && grid.n[d] == 3) {
                return grid;
            }
            cell *= 1.25;
        }
    }
}

const WG: u32 = 64;
/// Chunks the prefix sum splits cells into, each summed by one invocation.
const CHUNKS: u32 = 256;

/// `[n, total_cells, nx, ny, nz]`
const N_ATOMS: usize = 0;
const N_CELLS: usize = 1;
const NX: usize = 2;
const NY: usize = 3;
const NZ: usize = 4;

/// Each atom's cell, clamped to the grid. Clamping keeps stencil neighbors within one cell
/// of each other, so an atom that drifts past the grid still finds its contacts.
#[cube(launch_unchecked)]
fn assign_cells<F: Float>(
    pos: &[F],
    grid: &[F],
    sizes: &[i32],
    atom_cell: &mut [u32],
    cell_count: &mut [Atomic<u32>],
) {
    let i = ABSOLUTE_POS;
    if (i as i32) < sizes[N_ATOMS] {
        let (nx, ny, nz) = (sizes[NX], sizes[NY], sizes[NZ]);
        let cx = i32::cast_from(F::floor((pos[3 * i] - grid[0]) * grid[3]));
        let cy = i32::cast_from(F::floor((pos[3 * i + 1] - grid[1]) * grid[4]));
        let cz = i32::cast_from(F::floor((pos[3 * i + 2] - grid[2]) * grid[5]));
        let cx = cx.clamp(0i32, nx - 1i32);
        let cy = cy.clamp(0i32, ny - 1i32);
        let cz = cz.clamp(0i32, nz - 1i32);
        let c = u32::cast_from((cx * ny + cy) * nz + cz);
        atom_cell[i] = c;
        cell_count[c as usize].fetch_add(1u32);
    }
}

#[cube(launch_unchecked)]
fn clear_cells(sizes: &[i32], cell_count: &mut [Atomic<u32>]) {
    if (ABSOLUTE_POS as i32) < sizes[N_CELLS] {
        cell_count[ABSOLUTE_POS].store(0u32);
    }
}

/// Sums one chunk of cell counts.
#[cube(launch_unchecked)]
fn chunk_sums(sizes: &[i32], cell_count: &mut [Atomic<u32>], sums: &mut [u32]) {
    let t = ABSOLUTE_POS as u32;
    if t < comptime!(CHUNKS) {
        let total = u32::cast_from(sizes[N_CELLS]);
        let chunk = (total + comptime!(CHUNKS) - 1u32) / comptime!(CHUNKS);
        let begin = t * chunk;
        let end = u32::min(begin + chunk, total);
        let mut s = 0u32;
        let mut c = begin;
        while c < end {
            s += cell_count[c as usize].load();
            c += 1u32;
        }
        sums[t as usize] = s;
    }
}

/// Turns chunk sums into each chunk's starting offset, and writes the total.
#[cube(launch_unchecked)]
fn chunk_offsets(sizes: &[i32], sums: &mut [u32], cell_start: &mut [u32]) {
    let mut acc = 0u32;
    let mut t = 0u32;
    while t < comptime!(CHUNKS) {
        let v = sums[t as usize];
        sums[t as usize] = acc;
        acc += v;
        t += 1u32;
    }
    cell_start[sizes[N_CELLS] as usize] = acc;
}

/// Fills one chunk of `cell_start`, seeding the scatter's cursors.
#[cube(launch_unchecked)]
fn fill_starts(
    sizes: &[i32],
    sums: &[u32],
    cell_count: &mut [Atomic<u32>],
    cell_start: &mut [u32],
    cursor: &mut [Atomic<u32>],
) {
    let t = ABSOLUTE_POS as u32;
    if t < comptime!(CHUNKS) {
        let total = u32::cast_from(sizes[N_CELLS]);
        let chunk = (total + comptime!(CHUNKS) - 1u32) / comptime!(CHUNKS);
        let begin = t * chunk;
        let end = u32::min(begin + chunk, total);
        let mut acc = sums[t as usize];
        let mut c = begin;
        while c < end {
            cell_start[c as usize] = acc;
            cursor[c as usize].store(acc);
            acc += cell_count[c as usize].load();
            c += 1u32;
        }
    }
}

#[cube(launch_unchecked)]
fn scatter(sizes: &[i32], atom_cell: &[u32], cursor: &mut [Atomic<u32>], sorted_atoms: &mut [u32]) {
    let i = ABSOLUTE_POS;
    if (i as i32) < sizes[N_ATOMS] {
        let slot = cursor[atom_cell[i] as usize].fetch_add(1u32);
        sorted_atoms[slot as usize] = i as u32;
    }
}

/// Makes the within-cell order deterministic. The atomic scatter fills each cell in race
/// order, and a chaotic trajectory amplifies the resulting summation-order rounding. Cells
/// hold a handful of atoms, so an insertion sort per cell is cheap.
#[cube(launch_unchecked)]
fn sort_cells(sizes: &[i32], cell_start: &[u32], sorted_atoms: &mut [u32]) {
    let c = ABSOLUTE_POS;
    if (c as i32) < sizes[N_CELLS] {
        let lo = cell_start[c];
        let hi = cell_start[c + 1];
        let mut a = lo + 1u32;
        while a < hi {
            let key = sorted_atoms[a as usize];
            let mut b = a;
            while b > lo {
                let prev = sorted_atoms[(b - 1u32) as usize];
                if prev <= key {
                    break;
                }
                sorted_atoms[b as usize] = prev;
                b -= 1u32;
            }
            sorted_atoms[b as usize] = key;
            a += 1u32;
        }
    }
}

/// A buffer only kernels touch.
#[derive(Clone)]
pub struct DeviceColumn {
    handle: Handle,
    len: usize,
}

impl DeviceColumn {
    /// A device-only column of `len` elements of `size` bytes.
    pub fn empty(gpu: &GpuDevice, len: usize, size: usize) -> Self {
        let len = len.max(1);
        Self {
            handle: gpu.client.empty(len * size),
            len,
        }
    }

    /// The kernel argument for this column.
    pub fn arg(&self) -> BufferArg {
        // SAFETY: the handle was allocated for this many elements.
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.len) }
    }

    /// A column holding `data`.
    pub fn from_slice<T: CubeElement>(gpu: &GpuDevice, data: &[T]) -> Self {
        Self {
            handle: gpu.client.create_from_slice(T::as_bytes(data)),
            len: data.len(),
        }
    }

    /// The first `n` elements, waiting for kernels already launched.
    pub fn read<T: CubeElement + Copy>(&self, gpu: &GpuDevice, n: usize) -> Vec<T> {
        let bytes = gpu
            .client
            .read_one(self.handle.clone())
            .expect("soil_gpu: read failed");
        T::from_bytes(&bytes)[..n].to_vec()
    }
}

/// GPU cell-list builder.
pub struct CellList {
    gpu: GpuDevice,
    n: usize,
    total_cells: usize,
    atom_cell: DeviceColumn,
    cell_count: DeviceColumn,
    cell_start: DeviceColumn,
    cursor: DeviceColumn,
    sorted_atoms: DeviceColumn,
    sums: DeviceColumn,
    grid_params: DeviceColumn,
    sizes: DeviceColumn,
}

impl CellList {
    /// A cell list binning the first `n` atoms on `grid`.
    pub fn new(gpu: GpuDevice, n: usize, grid: Grid) -> Self {
        let mut list = Self {
            n: 0,
            total_cells: 0,
            atom_cell: DeviceColumn::empty(&gpu, 1, 4),
            cell_count: DeviceColumn::empty(&gpu, 1, 4),
            cell_start: DeviceColumn::empty(&gpu, 2, 4),
            cursor: DeviceColumn::empty(&gpu, 1, 4),
            sorted_atoms: DeviceColumn::empty(&gpu, 1, 4),
            sums: DeviceColumn::empty(&gpu, CHUNKS as usize, 4),
            grid_params: DeviceColumn::empty(&gpu, 6, std::mem::size_of::<GpuReal>()),
            sizes: DeviceColumn::empty(&gpu, 5, 4),
            gpu,
        };
        list.set_active(n, grid);
        list
    }

    /// Bin the first `n` atoms on `grid`, reallocating only what changed size.
    pub fn set_active(&mut self, n: usize, grid: Grid) {
        let gpu = &self.gpu;
        if n != self.n {
            self.atom_cell = DeviceColumn::empty(gpu, n, 4);
            self.sorted_atoms = DeviceColumn::empty(gpu, n, 4);
            self.n = n;
        }
        if grid.total_cells != self.total_cells {
            self.cell_count = DeviceColumn::empty(gpu, grid.total_cells, 4);
            self.cursor = DeviceColumn::empty(gpu, grid.total_cells, 4);
            self.cell_start = DeviceColumn::empty(gpu, grid.total_cells + 1, 4);
            self.total_cells = grid.total_cells;
        }
        self.set_grid(grid);
    }

    /// Set the grid only. Its cell count must match, or use [`set_active`](Self::set_active).
    pub fn set_grid(&mut self, grid: Grid) {
        if grid.total_cells != self.total_cells {
            return self.set_active(self.n, grid);
        }
        let inv = grid.bin_size.map(|b| 1.0 / b);
        self.grid_params = DeviceColumn::from_slice(
            &self.gpu,
            &[
                grid.origin[0],
                grid.origin[1],
                grid.origin[2],
                inv[0],
                inv[1],
                inv[2],
            ],
        );
        self.sizes = DeviceColumn::from_slice(
            &self.gpu,
            &[
                self.n as i32,
                self.total_cells as i32,
                grid.n[0],
                grid.n[1],
                grid.n[2],
            ],
        );
    }

    /// Launch the build over a flat xyz position column.
    pub fn launch(&self, pos: BufferArg) {
        let client = &self.gpu.client;
        let dim = CubeDim::new_1d(WG);
        let cells = CubeCount::Static((self.total_cells as u32).div_ceil(WG).max(1), 1, 1);
        let atoms = CubeCount::Static((self.n as u32).div_ceil(WG).max(1), 1, 1);
        let chunks = CubeCount::Static(CHUNKS.div_ceil(WG), 1, 1);
        // SAFETY: every index a kernel forms is bounded by `sizes`, which matches the
        // allocations above, or by the cell total the prefix sum computed.
        unsafe {
            clear_cells::launch_unchecked(
                client,
                cells.clone(),
                dim,
                self.sizes.arg(),
                self.cell_count.arg(),
            );
            assign_cells::launch_unchecked::<GpuReal>(
                client,
                atoms.clone(),
                dim,
                pos,
                self.grid_params.arg(),
                self.sizes.arg(),
                self.atom_cell.arg(),
                self.cell_count.arg(),
            );
            chunk_sums::launch_unchecked(
                client,
                chunks.clone(),
                dim,
                self.sizes.arg(),
                self.cell_count.arg(),
                self.sums.arg(),
            );
            chunk_offsets::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                self.sizes.arg(),
                self.sums.arg(),
                self.cell_start.arg(),
            );
            fill_starts::launch_unchecked(
                client,
                chunks,
                dim,
                self.sizes.arg(),
                self.sums.arg(),
                self.cell_count.arg(),
                self.cell_start.arg(),
                self.cursor.arg(),
            );
            scatter::launch_unchecked(
                client,
                atoms,
                dim,
                self.sizes.arg(),
                self.atom_cell.arg(),
                self.cursor.arg(),
                self.sorted_atoms.arg(),
            );
            sort_cells::launch_unchecked(
                client,
                cells,
                dim,
                self.sizes.arg(),
                self.cell_start.arg(),
                self.sorted_atoms.arg(),
            );
        }
    }

    /// Output for a force kernel: each atom's cell.
    pub fn atom_cell(&self) -> BufferArg {
        self.atom_cell.arg()
    }
    /// Output for a force kernel: CSR offsets into `sorted_atoms`.
    pub fn cell_start(&self) -> BufferArg {
        self.cell_start.arg()
    }
    /// Output for a force kernel: atom indices grouped by cell.
    pub fn sorted_atoms(&self) -> BufferArg {
        self.sorted_atoms.arg()
    }

    /// Reads the outputs back, for tests.
    pub fn download(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        (
            self.atom_cell.read(&self.gpu, self.n),
            self.cell_start.read(&self.gpu, self.total_cells + 1),
            self.sorted_atoms.read(&self.gpu, self.n),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grass_gpu::DualBuffer;

    #[test]
    fn build_is_a_sorted_permutation_with_correct_bins() {
        let Ok(gpu) = GpuDevice::default_device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let n = 3000usize;
        let mut pos = Vec::with_capacity(3 * n);
        for i in 0..n {
            let f = i as GpuReal;
            pos.extend_from_slice(&[
                (f * 0.137).fract() * 4.0,
                (f * 0.371).fract() * 4.0,
                (f * 0.713).fract() * 4.0,
            ]);
        }
        let grid = Grid::from_positions(&pos, 0.5);
        let mut column = DualBuffer::new(&gpu, pos.clone());
        let list = CellList::new(gpu.clone(), n, grid);
        list.launch(column.gpu_read());
        let (atom_cell, cell_start, sorted) = list.download();

        assert_eq!(cell_start[grid.total_cells] as usize, n);
        let mut seen = vec![false; n];
        for c in 0..grid.total_cells {
            let slice = &sorted[cell_start[c] as usize..cell_start[c + 1] as usize];
            assert!(slice.windows(2).all(|w| w[0] < w[1]), "cell {c} not sorted");
            for &a in slice {
                assert_eq!(atom_cell[a as usize] as usize, c);
                assert!(!std::mem::replace(&mut seen[a as usize], true));
            }
        }
        assert!(seen.iter().all(|&s| s));
        for (i, p) in pos.chunks_exact(3).enumerate() {
            let cell = |d: usize| ((p[d] - grid.origin[d]) / grid.bin_size[d]).floor() as i32;
            let expected = (cell(0) * grid.n[1] + cell(1)) * grid.n[2] + cell(2);
            assert_eq!(atom_cell[i] as i32, expected, "atom {i}");
        }
    }
}
