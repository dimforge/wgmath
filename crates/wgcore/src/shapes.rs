//! Tensor shape definition.

use crate::tensor::MatrixOrdering;
use dashmap::DashMap;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferUsages, Device};

#[derive(Copy, Clone, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
/// The shape of a matrix view over a GPU tensor.
pub struct ViewShape {
    /// The tensor view’s number of rows, columns, and matrices.
    pub size: [u32; 3],
    /// The view’s column stride (number of elements between two columns).
    pub stride: u32,
    /// The view’s matrix stride (number of elements between two matrices in the tensor).
    pub stride_mat: u32,
    /// Index of the first element of the view on the underlying buffer.
    pub offset: u32,
}

impl ViewShape {
    /// Converts the shape `self` for a buffer `&[f32]` to a buffer `&[vec4f]`.
    pub fn f32_to_vec4<Ordering: MatrixOrdering>(self) -> Self {
        let size = if Ordering::is_column_major() {
            [self.size[0] / 4, self.size[1], self.size[2]]
        } else {
            [self.size[0], self.size[1] / 4, self.size[2]]
        };

        Self {
            size,
            stride: self.stride / 4,
            stride_mat: self.stride_mat / 4,
            offset: self.offset / 4,
        }
    }
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
