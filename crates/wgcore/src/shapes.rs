//! Tensor shape definition.

use crate::tensor::MatrixOrdering;
use dashmap::DashMap;
use std::sync::{Arc, Mutex};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferUsages, Device, Queue};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
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
    // TODO: once we switch to wgpu 14, we can store a `Buffer` directly instead of
    //       `Arc<Buffer>` (they will be clonable), and we can also store the `Device`
    //       here to simplify `self.get` and the kernel dispatch apis.
    buffers: DashMap<ViewShape, Arc<Buffer>>,
    tmp_buffers: DashMap<ViewShape, Arc<Buffer>>,
    recycled: Mutex<Vec<Arc<Buffer>>>,
}

impl ViewShapeBuffers {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self {
            buffers: DashMap::new(),
            tmp_buffers: DashMap::new(),
            recycled: Mutex::new(vec![]),
        }
    }

    pub fn clear_tmp(&self) {
        let mut recycled = self.recycled.lock().unwrap();
        self.tmp_buffers.retain(|_, buffer| {
            recycled.push(buffer.clone());
            false
        })
    }

    pub fn put_tmp(&self, device: &Device, queue: &Queue, shape: ViewShape) {
        if self.contains(shape) {
            return;
        }

        let mut recycled = self.recycled.lock().unwrap();
        let buffer = if let Some(buffer) = recycled.pop() {
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&[shape]));
            buffer
        } else {
            drop(recycled);
            Self::make_buffer(
                device,
                shape,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )
        };

        self.tmp_buffers.insert(shape, buffer);
    }

    fn make_buffer(device: &Device, shape: ViewShape, usage: BufferUsages) -> Arc<Buffer> {
        Arc::new(device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[shape]),
            usage,
        }))
    }

    pub fn contains(&self, shape: ViewShape) -> bool {
        self.buffers.contains_key(&shape) || self.tmp_buffers.contains_key(&shape)
    }

    /// Gets of insert the gpu uniform storage `Buffer` containing the value of `shape`.
    pub fn get(&self, device: &Device, shape: ViewShape) -> Arc<Buffer> {
        if let Some(buffer) = self.tmp_buffers.get(&shape) {
            return buffer.value().clone();
        }

        self.buffers
            .entry(shape)
            .or_insert_with(|| Self::make_buffer(device, shape, BufferUsages::UNIFORM))
            .clone()
    }
}
