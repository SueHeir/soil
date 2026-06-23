//! Host/device coherence primitive (substrate) — a `DualBuffer`.
//!
//! Tracks where the fresh copy of a field lives (host vs device) with two
//! "modified" flags, the Kokkos `DualView` model. Sync happens lazily and only
//! when needed:
//! - `ensure_device()` uploads iff the host was modified since the last sync;
//! - `ensure_host()` downloads iff the device was modified.
//!
//! A chain of device-only operations trips zero transfers (resident); a host
//! consumer that reads the field downloads only then, and only this field. This
//! is the building block the GPU step loop uses so "all-GPU = fast, mixed =
//! auto-synced, same API" holds. Invariant: at most one of the two modified
//! flags is set at a time (you sync before modifying the other side).

use bytemuck::{Pod, Zeroable};

use crate::GpuContext;

/// A field mirrored on host and device with lazy, minimal synchronization.
pub struct DualBuffer<T: Pod + Zeroable> {
    ctx: GpuContext,
    host: Vec<T>,
    device: wgpu::Buffer,
    staging: wgpu::Buffer,
    modified_host: bool,
    modified_device: bool,
}

impl<T: Pod + Zeroable> DualBuffer<T> {
    /// Create from initial host data. `extra_usage` is OR'd with COPY_SRC/COPY_DST
    /// (e.g. `STORAGE` for a compute buffer, `UNIFORM` for params). Starts in the
    /// "host fresh" state (the initial data must be uploaded before device use).
    pub fn new(ctx: GpuContext, host: Vec<T>, extra_usage: wgpu::BufferUsages) -> Self {
        let bytes = std::mem::size_of_val(host.as_slice()).max(std::mem::size_of::<T>()) as u64;
        let device = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dualbuffer"),
            size: bytes,
            usage: extra_usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dualbuffer staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { ctx, host, device, staging, modified_host: true, modified_device: false }
    }

    pub fn len(&self) -> usize { self.host.len() }
    pub fn is_empty(&self) -> bool { self.host.is_empty() }

    /// Read host data, syncing down from device first if the device is fresher.
    pub fn host(&mut self) -> &[T] {
        self.ensure_host();
        &self.host
    }

    /// Mutable host access; marks the device stale (a host write happened).
    pub fn host_mut(&mut self) -> &mut [T] {
        self.ensure_host();
        self.modified_host = true;
        &mut self.host
    }

    /// The device buffer, synced up from host first if the host is fresher.
    /// Call before binding it into a kernel that READS the field.
    pub fn device(&mut self) -> &wgpu::Buffer {
        self.ensure_device();
        &self.device
    }

    /// The device buffer without a sync — for binding into a kernel that fully
    /// OVERWRITES the field (no need to upload stale host data first). Pair with
    /// `mark_device_modified()` after the kernel runs.
    pub fn device_raw(&self) -> &wgpu::Buffer { &self.device }

    /// Record that a kernel wrote the device buffer (host now stale).
    pub fn mark_device_modified(&mut self) { self.modified_device = true; }

    /// Record that the host data changed out-of-band (device now stale).
    pub fn mark_host_modified(&mut self) { self.modified_host = true; }

    /// Upload iff the host was modified since the last sync. No-op otherwise.
    pub fn ensure_device(&mut self) {
        if self.modified_host {
            self.ctx.queue.write_buffer(&self.device, 0, bytemuck::cast_slice(&self.host));
            self.modified_host = false;
        }
    }

    /// Download iff the device was modified since the last sync. No-op otherwise.
    pub fn ensure_host(&mut self) {
        if !self.modified_device {
            return;
        }
        let bytes = std::mem::size_of_val(self.host.as_slice()) as u64;
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dualbuffer download"),
        });
        enc.copy_buffer_to_buffer(&self.device, 0, &self.staging, 0, bytes);
        self.ctx.queue.submit(Some(enc.finish()));
        let slice = self.staging.slice(0..bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range();
        self.host.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        self.staging.unmap();
        self.modified_device = false;
    }

    /// True if there are no pending syncs in either direction.
    pub fn is_coherent(&self) -> bool {
        !self.modified_host && !self.modified_device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dualbuffer_lazy_sync_roundtrip() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let mut db = DualBuffer::new(ctx, vec![1.0f32, 2.0, 3.0, 4.0], wgpu::BufferUsages::STORAGE);

        // Starts host-fresh: ensure_device uploads once, then is coherent.
        assert!(!db.is_coherent(), "fresh buffer has pending host upload");
        db.ensure_device();
        assert!(db.is_coherent());
        // ensure_device again is a no-op (nothing modified).
        db.ensure_device();
        assert!(db.is_coherent());

        // Simulate a device-side write: clobber the host mirror, mark device
        // modified, then ensure_host must restore the uploaded values from device.
        for x in db.host.iter_mut() { *x = -99.0; }
        db.mark_device_modified();
        assert!(!db.is_coherent());
        let restored = db.host().to_vec();
        assert_eq!(restored, vec![1.0, 2.0, 3.0, 4.0], "download restores device data");
        assert!(db.is_coherent());

        // A host write marks device stale; ensure_device re-uploads.
        db.host_mut()[0] = 42.0;
        assert!(!db.is_coherent());
        db.ensure_device();
        assert!(db.is_coherent());
        eprintln!("DualBuffer lazy sync round-trip OK");
    }
}
