//! Persistent per-neighbor state slots (substrate; no physics).
//!
//! Each atom owns `MAX_CONTACTS` fixed slots holding `(partner, payload)`, double-buffered:
//! a kernel reads the old slots and writes the new ones each step, and the host swaps.
//! Contacts not rewritten are pruned for free, and new ones start at zero. This is what DEM
//! uses for tangential springs and peridynamics would use for bond state. Partner ids are
//! whatever the caller keys by, tags if atoms can reorder.
//!
//! A kernel takes `old_partner`, `old_payload`, `new_partner`, `new_payload` from
//! [`args`](NeighborSlots::args), and uses [`slot_lookup`], [`slot_write`] and
//! [`slot_clear_from`].

use std::cell::Cell;

use cubecl::prelude::*;
use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use grass_gpu::cubecl;
use grass_gpu::{GpuDevice, GpuReal};

/// Fixed per-atom slot capacity.
pub const MAX_CONTACTS: usize = 32;
/// Payload floats per slot.
pub const PAYLOAD: usize = 3;
/// Empty-slot marker.
pub const SENTINEL: u32 = 0xFFFF_FFFF;

/// Atom `i`'s old payload for `partner`, or zero if there is none.
#[cube]
pub fn slot_lookup<F: Float>(
    old_partner: &[u32],
    old_payload: &[F],
    i: u32,
    partner: u32,
) -> (F, F, F) {
    let base = i * comptime!(MAX_CONTACTS as u32);
    let mut x = F::new(0.0f32);
    let mut y = F::new(0.0f32);
    let mut z = F::new(0.0f32);
    let mut k = 0u32;
    while k < comptime!(MAX_CONTACTS as u32) {
        if old_partner[(base + k) as usize] == partner {
            let b = (comptime!(PAYLOAD as u32) * (base + k)) as usize;
            x = old_payload[b];
            y = old_payload[b + 1];
            z = old_payload[b + 2];
            break;
        }
        k += 1u32;
    }
    (x, y, z)
}

/// Writes atom `i`'s new slot `index`.
#[cube]
pub fn slot_write<F: Float>(
    new_partner: &mut [u32],
    new_payload: &mut [F],
    i: u32,
    index: u32,
    partner: u32,
    payload: (F, F, F),
) {
    let slot = i * comptime!(MAX_CONTACTS as u32) + index;
    new_partner[slot as usize] = partner;
    let b = (comptime!(PAYLOAD as u32) * slot) as usize;
    new_payload[b] = payload.0;
    new_payload[b + 1] = payload.1;
    new_payload[b + 2] = payload.2;
}

/// Marks atom `i`'s remaining new slots empty.
#[cube]
pub fn slot_clear_from(new_partner: &mut [u32], i: u32, start: u32) {
    let base = i * comptime!(MAX_CONTACTS as u32);
    let mut k = start;
    while k < comptime!(MAX_CONTACTS as u32) {
        new_partner[(base + k) as usize] = comptime!(SENTINEL);
        k += 1u32;
    }
}

/// Double-buffered per-atom neighbor-state slots.
pub struct NeighborSlots {
    gpu: GpuDevice,
    n: usize,
    partner: [Handle; 2],
    payload: [Handle; 2],
    /// Which pair of buffers the next kernel reads as old.
    old: Cell<usize>,
}

impl NeighborSlots {
    /// Empty slots for `n` atoms.
    pub fn new(gpu: GpuDevice, n: usize) -> Self {
        let handle = |bytes: usize| gpu.client.empty(bytes.max(4));
        let partner_bytes = n * MAX_CONTACTS * 4;
        let payload_bytes = n * MAX_CONTACTS * PAYLOAD * std::mem::size_of::<GpuReal>();
        let slots = Self {
            partner: [handle(partner_bytes), handle(partner_bytes)],
            payload: [handle(payload_bytes), handle(payload_bytes)],
            old: Cell::new(0),
            n,
            gpu,
        };
        slots.clear();
        slots
    }

    /// How many atoms the slots hold.
    pub fn len(&self) -> usize {
        self.n
    }

    /// `true` when the slots hold no atoms.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Empties both buffers.
    pub fn clear(&self) {
        if self.n == 0 {
            return;
        }
        let partner = vec![SENTINEL; self.n * MAX_CONTACTS];
        let payload = vec![0.0 as GpuReal; self.n * MAX_CONTACTS * PAYLOAD];
        for k in 0..2 {
            self.gpu.client.write(&self.partner[k], Bytes::from_elems(partner.clone()));
            self.gpu.client.write(&self.payload[k], Bytes::from_elems(payload.clone()));
        }
    }

    /// `(old_partner, old_payload, new_partner, new_payload)` for this step's kernel.
    pub fn args(&self) -> (BufferArg, BufferArg, BufferArg, BufferArg) {
        let (old, new) = (self.old.get(), 1 - self.old.get());
        let (slots, floats) = (self.n.max(1) * MAX_CONTACTS, self.n.max(1) * MAX_CONTACTS * PAYLOAD);
        // SAFETY: the handles were allocated for this many elements.
        unsafe {
            (
                BufferArg::from_raw_parts(self.partner[old].clone(), slots),
                BufferArg::from_raw_parts(self.payload[old].clone(), floats),
                BufferArg::from_raw_parts(self.partner[new].clone(), slots),
                BufferArg::from_raw_parts(self.payload[new].clone(), floats),
            )
        }
    }

    /// Swap old and new for the next step. Call after each step's kernel.
    pub fn swap(&self) {
        self.old.set(1 - self.old.get());
    }

    /// Each slot's partner and payload as last written, for the first `rows` atoms, in the
    /// layout `partner[rows * MAX_CONTACTS]`, `payload[rows * MAX_CONTACTS * PAYLOAD]`.
    /// Blocks until kernels already launched have finished.
    pub fn read_latest(&self, rows: usize) -> (Vec<u32>, Vec<GpuReal>) {
        let old = self.old.get();
        let data = self
            .gpu
            .client
            .read(vec![self.partner[old].clone(), self.payload[old].clone()])
            .expect("soil_gpu: slot read failed");
        self.gpu.counts().record_download((data[0].len() + data[1].len()) as u64);
        (
            u32::from_bytes(&data[0])[..rows * MAX_CONTACTS].to_vec(),
            GpuReal::from_bytes(&data[1])[..rows * MAX_CONTACTS * PAYLOAD].to_vec(),
        )
    }

    /// Replaces what the next kernel reads, for the leading atoms these slices cover.
    pub fn write_for_next(&self, partner: &[u32], payload: &[GpuReal]) {
        let old = self.old.get();
        let client = &self.gpu.client;
        let mut partner = partner.to_vec();
        let mut payload = payload.to_vec();
        partner.resize(self.n * MAX_CONTACTS, SENTINEL);
        payload.resize(self.n * MAX_CONTACTS * PAYLOAD, 0.0);
        self.gpu.counts().record_upload(
            std::mem::size_of_val(partner.as_slice()) as u64 + std::mem::size_of_val(payload.as_slice()) as u64,
        );
        client.write(&self.partner[old], Bytes::from_elems(partner));
        client.write(&self.payload[old], Bytes::from_elems(payload));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Adds one to each atom's self-keyed payload, so history accumulates across swaps.
    #[cube(launch_unchecked)]
    fn accumulate<F: Float>(
        sizes: &[u32],
        old_partner: &[u32],
        old_payload: &[F],
        new_partner: &mut [u32],
        new_payload: &mut [F],
    ) {
        let i = ABSOLUTE_POS as u32;
        if i < sizes[0] {
            let old = slot_lookup::<F>(old_partner, old_payload, i, i);
            slot_write::<F>(new_partner, new_payload, i, 0u32, i, (old.0 + F::new(1.0f32), old.1, old.2));
            slot_clear_from(new_partner, i, 1u32);
        }
    }

    #[test]
    fn history_persists_across_swaps_and_round_trips() {
        let Ok(gpu) = GpuDevice::default_device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let n = 100;
        let slots = NeighborSlots::new(gpu.clone(), n);
        let sizes = gpu.client.create_from_slice(u32::as_bytes(&[n as u32]));
        let step = || {
            let (a, b, c, d) = slots.args();
            // SAFETY: bounded by `sizes`, which is the slots' atom count.
            unsafe {
                accumulate::launch_unchecked::<GpuReal>(
                    &gpu.client,
                    CubeCount::Static(2, 1, 1),
                    CubeDim::new_1d(64),
                    BufferArg::from_raw_parts(sizes.clone(), 1),
                    a,
                    b,
                    c,
                    d,
                );
            }
            slots.swap();
        };
        for _ in 0..7 {
            step();
        }
        let (partner, payload) = slots.read_latest(n);
        for i in 0..n {
            assert_eq!(partner[i * MAX_CONTACTS], i as u32);
            assert_eq!(payload[i * MAX_CONTACTS * PAYLOAD], 7.0);
            assert_eq!(partner[i * MAX_CONTACTS + 1], SENTINEL);
        }
        slots.write_for_next(&partner, &payload);
        step();
        let (_, payload) = slots.read_latest(n);
        assert_eq!(payload[0], 8.0);
    }
}
