//! Tensor shape definition.

use dashmap::DashMap;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferUsages, Device};

#[derive(Copy, Clone, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(16))]
/// The shape of a matrix view over a GPU tensor.
pub struct ViewShape {
    /// The view’s number of rows and columns.
    pub size: [u32; 2],
    /// The view’s column stride (number of elements between two columns).
    pub stride: u32,
    /// Index of the first element of the view on the underlying buffer.
    pub offset: u32,
}

/// A map between a `ViewShape` and an uniform storage `Buffer` containing its value on the gpu.
///
/// Ideally, we should use push-constants for view shapes. Unfortunately, push-constants is an
/// optional extension, so we have to emulate them with uniforms for maximum portability.
#[derive(Default)]
pub struct ViewShapeBuffers {
    buffers: DashMap<ViewShape, Arc<Buffer>>,
}

impl ViewShapeBuffers {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self {
            buffers: DashMap::new(),
        }
    }

    /// Gets of insert the gpu uniform storage `Buffer` containing the value of `shape`.
    pub fn get(&self, device: &Device, shape: ViewShape) -> Arc<Buffer> {
        self.buffers
            .entry(shape)
            .or_insert_with(|| {
                Arc::new(device.create_buffer_init(&BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&[shape]),
                    usage: BufferUsages::UNIFORM,
                }))
            })
            .clone()
    }
}
