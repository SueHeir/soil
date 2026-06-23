// Generic GPU cell-list neighbor build (f32). No physics — substrate only.
//
// bin -> count (atomic) -> exclusive prefix-sum (256-way chunked, single
// workgroup) -> scatter. Produces `cell_start` (CSR offsets, len total_cells+1)
// and `sorted_atoms` (atom indices grouped by cell), plus `atom_cell` (each
// atom's cell). A consumer (force kernel, bond kernel, ...) reads these to scan
// each atom's neighbourhood. binsize is chosen by the host = cutoff, so a +/-1
// cell stencil captures every interacting pair.

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
};

@group(0) @binding(0) var<storage, read>        pos: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;
@group(0) @binding(2) var<storage, read_write>  atom_cell: array<u32>;
@group(0) @binding(3) var<storage, read_write>  cell_count: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write>  cell_start: array<u32>;
@group(0) @binding(5) var<storage, read_write>  cursor: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write>  sorted_atoms: array<u32>;

fn cell_of(i: u32) -> u32 {
    let b = 3u * i;
    var cx = i32(floor((pos[b]      - params.ox) * params.inv_bx));
    var cy = i32(floor((pos[b + 1u] - params.oy) * params.inv_by));
    var cz = i32(floor((pos[b + 2u] - params.oz) * params.inv_bz));
    cx = clamp(cx, 0, params.nx - 1);
    cy = clamp(cy, 0, params.ny - 1);
    cz = clamp(cz, 0, params.nz - 1);
    return u32((cx * params.ny + cy) * params.nz + cz);
}

@compute @workgroup_size(64)
fn clear_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.total_cells) { return; }
    atomicStore(&cell_count[c], 0u);
}

@compute @workgroup_size(64)
fn assign_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let c = cell_of(i);
    atom_cell[i] = c;
    atomicAdd(&cell_count[c], 1u);
}

// Exclusive prefix sum of cell_count -> cell_start, seeding cursors. 256-way
// chunked scan in one workgroup (serial work per thread ~ cells/256).
const SCAN_THREADS: u32 = 256u;
var<workgroup> sh: array<u32, 256>;

@compute @workgroup_size(256)
fn prefix_sum(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let tc = params.total_cells;
    let chunk = (tc + SCAN_THREADS - 1u) / SCAN_THREADS;
    let begin = t * chunk;
    var end = begin + chunk;
    if (end > tc) { end = tc; }

    var s: u32 = 0u;
    var c = begin;
    loop {
        if (c >= end) { break; }
        s = s + atomicLoad(&cell_count[c]);
        c = c + 1u;
    }
    sh[t] = s;
    workgroupBarrier();

    if (t == 0u) {
        var acc: u32 = 0u;
        for (var i: u32 = 0u; i < SCAN_THREADS; i = i + 1u) {
            let v = sh[i];
            sh[i] = acc;
            acc = acc + v;
        }
        cell_start[tc] = acc;
    }
    workgroupBarrier();

    var acc2 = sh[t];
    var c2 = begin;
    loop {
        if (c2 >= end) { break; }
        cell_start[c2] = acc2;
        atomicStore(&cursor[c2], acc2);
        acc2 = acc2 + atomicLoad(&cell_count[c2]);
        c2 = c2 + 1u;
    }
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let c = atom_cell[i];
    let slot = atomicAdd(&cursor[c], 1u);
    sorted_atoms[slot] = i;
}

// Make the within-cell order DETERMINISTIC. The atomic scatter above fills each
// cell's slice of `sorted_atoms` in non-deterministic (race) order; a chaotic
// granular trajectory amplifies the resulting f32 summation-order noise. One
// thread per cell insertion-sorts its slice by ascending atom index, so the
// neighbour traversal — and thus the whole simulation — is reproducible and
// window-boundary independent (the residency / MPI-halo correctness gate). Cells
// hold only a handful of atoms at the contact-cutoff grid spacing, so this is cheap.
@compute @workgroup_size(64)
fn sort_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= params.total_cells) { return; }
    let lo = cell_start[c];
    let hi = cell_start[c + 1u];
    var a = lo + 1u;
    loop {
        if (a >= hi) { break; }
        let key = sorted_atoms[a];
        var b = a;
        loop {
            if (b == lo) { break; }
            let prev = sorted_atoms[b - 1u];
            if (prev <= key) { break; }
            sorted_atoms[b] = prev;
            b = b - 1u;
        }
        sorted_atoms[b] = key;
        a = a + 1u;
    }
}
