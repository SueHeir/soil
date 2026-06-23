//! Persistent per-neighbor state slots (substrate; no physics).
//!
//! Each atom owns `MAX_CONTACTS` fixed slots holding `(partner_index, payload)`,
//! double-buffered (ping-pong): a kernel reads OLD slots and writes NEW ones each
//! step; the host swaps. Contacts not re-written are pruned for free; new contacts
//! start at zero. Keyed by partner atom index (valid on a single device — no atom
//! permutation/migration). This is the generic structure DEM uses for tangential
//! contact springs and peridynamics would use for bond state.
//!
//! The slots live in **bind group 1** (a consumer kernel keeps its own buffers in
//! group 0), via the WGSL snippet [`SLOTS_WGSL`] which a consumer concatenates
//! into its shader. The snippet provides `slot_lookup`, `slot_write`,
//! `slot_clear_from`. Payload is fixed at 3 floats/slot in v1 (DEM tangential
//! spring); widen later by templating the snippet.

use crate::GpuContext;

/// Fixed per-atom slot capacity (must match `SLOT_MAX` in [`SLOTS_WGSL`]).
pub const MAX_CONTACTS: usize = 32;
/// Payload floats per slot (must match `SLOT_P` in [`SLOTS_WGSL`]).
pub const PAYLOAD: usize = 3;
/// Empty-slot marker (must match `SLOT_NONE`).
pub const SENTINEL: u32 = 0xFFFF_FFFF;

/// WGSL to concatenate into a consumer shader. Declares the slot buffers in
/// `@group(1)` and the lookup/write helpers. The consumer's pipeline layout must
/// be `[consumer_group0_layout, NeighborSlots::layout()]`, and it must
/// `set_bind_group(1, slots.current_bind_group(), &[])` (swapping each step).
pub const SLOTS_WGSL: &str = r#"
const SLOT_MAX: u32 = 32u;
const SLOT_P: u32 = 3u;
const SLOT_NONE: u32 = 0xFFFFFFFFu;
@group(1) @binding(0) var<storage, read>       slot_cp_old: array<u32>;
@group(1) @binding(1) var<storage, read>       slot_cs_old: array<f32>;
@group(1) @binding(2) var<storage, read_write> slot_cp_new: array<u32>;
@group(1) @binding(3) var<storage, read_write> slot_cs_new: array<f32>;

// Read atom i's OLD payload for `partner`; zero vec if not present.
fn slot_lookup(i: u32, partner: u32) -> vec3<f32> {
    let base = i * SLOT_MAX;
    for (var k: u32 = 0u; k < SLOT_MAX; k = k + 1u) {
        let p = slot_cp_old[base + k];
        if (p == SLOT_NONE) { continue; }
        if (p == partner) {
            let b = SLOT_P * (base + k);
            return vec3<f32>(slot_cs_old[b], slot_cs_old[b + 1u], slot_cs_old[b + 2u]);
        }
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Write atom i's NEW slot at index `idx` (caller's running count, < SLOT_MAX).
fn slot_write(i: u32, idx: u32, partner: u32, payload: vec3<f32>) {
    let base = i * SLOT_MAX;
    slot_cp_new[base + idx] = partner;
    let b = SLOT_P * (base + idx);
    slot_cs_new[b] = payload.x;
    slot_cs_new[b + 1u] = payload.y;
    slot_cs_new[b + 2u] = payload.z;
}

// Mark atom i's remaining NEW slots empty (call after writing `start` contacts).
fn slot_clear_from(i: u32, start: u32) {
    let base = i * SLOT_MAX;
    for (var k: u32 = start; k < SLOT_MAX; k = k + 1u) {
        slot_cp_new[base + k] = SLOT_NONE;
    }
}
"#;

/// Double-buffered per-atom neighbor-state slots, bound in group 1.
#[allow(dead_code)]
pub struct NeighborSlots {
    ctx: GpuContext,
    n: usize,
    partner_a: wgpu::Buffer,
    partner_b: wgpu::Buffer,
    payload_a: wgpu::Buffer,
    payload_b: wgpu::Buffer,
    staging: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bg_ab: wgpu::BindGroup, // A=old(read), B=new(write)
    bg_ba: wgpu::BindGroup, // B=old(read), A=new(write)
    ping: std::cell::Cell<bool>,
}

impl NeighborSlots {
    pub fn new(ctx: GpuContext, n: usize) -> Self {
        let device = &ctx.device;
        let nz = n.max(1);
        let cp_bytes = (nz * MAX_CONTACTS * 4) as u64;
        let cs_bytes = (nz * MAX_CONTACTS * PAYLOAD * 4) as u64;
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let mk = |label: &str, size: u64| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label), size, usage: st, mapped_at_creation: false,
        });
        let partner_a = mk("slot_partner_a", cp_bytes);
        let partner_b = mk("slot_partner_b", cp_bytes);
        let payload_a = mk("slot_payload_a", cs_bytes);
        let payload_b = mk("slot_payload_b", cs_bytes);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slot_staging"), size: cs_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false, min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("slots bgl(group1)"),
            entries: &[entry(0, true), entry(1, true), entry(2, false), entry(3, false)],
        });
        let bg = |label, cp_o: &wgpu::Buffer, cs_o: &wgpu::Buffer, cp_n: &wgpu::Buffer, cs_n: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label), layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: cp_o.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: cs_o.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: cp_n.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: cs_n.as_entire_binding() },
                ],
            })
        };
        let bg_ab = bg("slots ab", &partner_a, &payload_a, &partner_b, &payload_b);
        let bg_ba = bg("slots ba", &partner_b, &payload_b, &partner_a, &payload_a);

        let s = Self {
            ctx, n, partner_a, partner_b, payload_a, payload_b, staging,
            layout, bg_ab, bg_ba, ping: std::cell::Cell::new(false),
        };
        s.clear();
        s
    }

    /// Initialise both buffers to empty (all SENTINEL partners, zero payload).
    pub fn clear(&self) {
        let sentinels = vec![SENTINEL; self.n * MAX_CONTACTS];
        let zeros = vec![0.0f32; self.n * MAX_CONTACTS * PAYLOAD];
        let q = &self.ctx.queue;
        q.write_buffer(&self.partner_a, 0, bytemuck::cast_slice(&sentinels));
        q.write_buffer(&self.partner_b, 0, bytemuck::cast_slice(&sentinels));
        q.write_buffer(&self.payload_a, 0, bytemuck::cast_slice(&zeros));
        q.write_buffer(&self.payload_b, 0, bytemuck::cast_slice(&zeros));
    }

    /// Group-1 bind-group layout — put this second in a consumer pipeline layout.
    pub fn layout(&self) -> &wgpu::BindGroupLayout { &self.layout }

    /// The current step's bind group (old→new). Bind at group 1.
    pub fn current_bind_group(&self) -> &wgpu::BindGroup {
        if self.ping.get() { &self.bg_ba } else { &self.bg_ab }
    }

    /// Swap old/new for the next step (call after each step's dispatch).
    pub fn swap(&self) { self.ping.set(!self.ping.get()); }

    /// Download the most-recently-written payload buffer (for validation). After a
    /// dispatch+swap, the just-written NEW becomes the current mapping's OLD:
    /// ping=true → current old is B; ping=false → current old is A.
    pub fn download_payload(&self) -> Vec<f32> {
        let new_buf = if self.ping.get() { &self.payload_b } else { &self.payload_a };
        let bytes = (self.n * MAX_CONTACTS * PAYLOAD * 4) as u64;
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("slot dl"),
        });
        enc.copy_buffer_to_buffer(new_buf, 0, &self.staging, 0, bytes);
        self.ctx.queue.submit(Some(enc.finish()));
        let slice = self.staging.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging.unmap();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct TParams { n: u32, _p: [u32; 3] }

    // Dogfood: a tiny kernel using SLOTS_WGSL accumulates a per-(atom,self) payload
    // across ping-pong steps. After K steps, payload.x must equal K — proving
    // lookup + write + persistence across the swap.
    const TEST_WGSL: &str = r#"
struct TParams { n: u32, a: u32, b: u32, c: u32 };
@group(0) @binding(0) var<uniform> tp: TParams;
@compute @workgroup_size(64)
fn slot_accum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= tp.n) { return; }
    let old = slot_lookup(i, i);         // self-partner contact
    slot_write(i, 0u, i, vec3<f32>(old.x + 1.0, 0.0, 0.0));
    slot_clear_from(i, 1u);
}
"#;

    #[test]
    fn neighbor_slots_persist_across_pingpong() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let n = 100usize;
        let slots = NeighborSlots::new(ctx.clone(), n);

        // Build the test pipeline: group0 = params, group1 = slots.
        let device = &ctx.device;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tp"), size: std::mem::size_of::<TParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&params, 0, bytemuck::bytes_of(&TParams { n: n as u32, _p: [0; 3] }));
        let g0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tg0"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let g0_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tg0 bg"), layout: &g0,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() }],
        });
        let src = format!("{SLOTS_WGSL}\n{TEST_WGSL}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("slot test"), source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("slot test pl"),
            bind_group_layouts: &[Some(&g0), Some(slots.layout())],
            immediate_size: 0,
        });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("slot_accum"), layout: Some(&pl), module: &shader,
            entry_point: Some("slot_accum"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let k = 5;
        for _ in 0..k {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("slot step") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("slot step"), timestamp_writes: None });
                pass.set_pipeline(&pipe);
                pass.set_bind_group(0, &g0_bg, &[]);
                pass.set_bind_group(1, slots.current_bind_group(), &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(64).max(1), 1, 1);
            }
            ctx.queue.submit(Some(enc.finish()));
            slots.swap();
        }
        // download_payload reads the last-written buffer (current mapping's OLD).
        let payload = slots.download_payload();
        for i in 0..n {
            let x = payload[i * MAX_CONTACTS * PAYLOAD]; // slot 0, component x
            assert!((x - k as f32).abs() < 1e-6, "atom {i}: accumulated {x}, expected {k}");
        }
        eprintln!("NeighborSlots: persisted accumulation to {k} across ping-pong, {n} atoms");
    }
}
