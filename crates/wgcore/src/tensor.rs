//! Utilities for initializing and slicing tensors, matrices, vectors, and scalars gpu storage
//! buffers.

use crate::shapes::ViewShape;
use bytemuck::Pod;
use encase::internal::WriteInto;
use encase::{ShaderSize, ShaderType, StorageBuffer};
use nalgebra::{Dim, IsContiguous, Matrix, Storage};
use std::marker::PhantomData;
use std::mem::size_of;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoder, Device,
};

/// A storage buffer containing a single value.
pub type GpuScalar<T> = GpuTensor<T, 0>;
/// A storage buffer containing a vector.
pub type GpuVector<T> = GpuTensor<T, 1>;
/// A storage buffer containing a matrix.
pub type GpuMatrix<T> = GpuTensor<T, 2>;

/// A view, over a storage buffer, containing a single value.
pub type GpuScalarView<'a, T> = GpuTensorView<'a, T, 0>;
/// A view, over a storage buffer, containing a vector.
pub type GpuVectorView<'a, T> = GpuTensorView<'a, T, 1>;
/// A view, over a storage buffer, containing a matrix.
pub type GpuMatrixView<'a, T> = GpuTensorView<'a, T, 2>;

/// Helper struct for creating gpu storage buffers (scalars, vectors, matrices, tensors).
///
/// When building a scalar, vector, or matrix tensor, it might be more convenient to call
/// [`GpuScalar::init`], [`GpuVector::init`], [`GpuMatrix::init`] (or their `encase` variants:
/// [`GpuScalar::encase`], [`GpuVector::encase`], [`GpuMatrix::encase`]; or their uninitialized
/// variants [`GpuScalar::uninit`], [`GpuVector::uninit`], [`GpuMatrix::uninit`]).
pub struct TensorBuilder<const DIM: usize> {
    shape: [u32; DIM],
    usage: BufferUsages,
    label: Option<String>,
}

impl TensorBuilder<0> {
    /// Starts building a storage buffer containing a single scalar value.
    pub fn scalar(usage: BufferUsages) -> Self {
        Self::tensor([], usage)
    }
}

impl TensorBuilder<1> {
    /// Starts building a storage buffer containing a vector.
    pub fn vector(dim: u32, usage: BufferUsages) -> Self {
        Self::tensor([dim], usage)
    }
}

impl TensorBuilder<2> {
    /// Starts building a storage buffer containing a single matrix with `nrows` rows and
    /// `ncols` columns.
    pub fn matrix(nrows: u32, ncols: u32, usage: BufferUsages) -> Self {
        Self::tensor([nrows, ncols], usage)
    }
}

impl<const DIM: usize> TensorBuilder<DIM> {
    /// Starts building a storage buffer containing a tensor with the specified `shape`.
    pub fn tensor(shape: [u32; DIM], usage: BufferUsages) -> Self {
        Self {
            shape,
            usage,
            label: None,
        }
    }

    /// The number of elements in this tensor.
    fn len(&self) -> u64 {
        self.shape.into_iter().map(|s| s as u64).product()
    }

    /// Sets the debug label of this tensor.
    pub fn label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }

    /// Builds the gpu tensor.
    pub fn build<T: Pod>(self, device: &Device) -> GpuTensor<T, DIM> {
        let bytes_len = std::mem::size_of::<T>() as u64 * self.len();
        let buffer = device.create_buffer(&BufferDescriptor {
            label: self.label.as_deref(),
            size: bytes_len,
            usage: self.usage,
            mapped_at_creation: false,
        });

        GpuTensor {
            shape: self.shape,
            buffer,
            phantom: PhantomData,
        }
    }

    /// Builds this tensor with raw bytes given for its initial value.
    pub fn build_bytes<T>(self, device: &Device, data: &[u8]) -> GpuTensor<T, DIM> {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: self.label.as_deref(),
            contents: bytemuck::cast_slice(data),
            usage: self.usage,
        });

        GpuTensor {
            shape: self.shape,
            buffer,
            phantom: PhantomData,
        }
    }

    /// Builds this tensor with raw bytes given for its initial value.
    pub fn build_encase<T>(self, device: &Device, data: impl AsRef<[T]>) -> GpuTensor<T, DIM>
    where
        T: ShaderType + ShaderSize + WriteInto,
    {
        let vector = data.as_ref();
        let mut bytes = vec![]; // TODO: can we avoid the allocation?
        let mut buffer = StorageBuffer::new(&mut bytes);
        buffer.write(vector).unwrap();
        self.build_bytes(device, &bytes)
    }
    /// Builds this tensor with an array of values given for its initial value.
    pub fn build_init<T: Pod>(self, device: &Device, data: &[T]) -> GpuTensor<T, DIM> {
        assert!(
            data.len() as u64 >= self.len(),
            "Incorrect number of elements provided for initializing Tensor.\
            Expected at least {}, found {}",
            self.len(),
            data.len()
        );

        let len = self.len();
        self.build_bytes::<T>(device, bytemuck::cast_slice(&data[..len as usize]))
    }
}

/// A tensor stored in the GPU.
///
/// When the tensor is a matrix, they are generally seen as being column-major.
pub struct GpuTensor<T, const DIM: usize> {
    shape: [u32; DIM],
    buffer: Buffer,
    phantom: PhantomData<T>,
}

impl<T, const DIM: usize> GpuTensor<T, DIM> {
    /// Does this tensor contain zero elements?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The number of elements in this tensor.
    pub fn len(&self) -> u64 {
        self.shape.into_iter().map(|s| s as u64).product()
    }

    /// The size, in bytes, of this tensor’s content.
    pub fn bytes_len(&self) -> u64
    where
        T: Pod,
    {
        std::mem::size_of::<T>() as u64 * self.len()
    }

    /// Queues a buffer-to-buffer copy from `source` to `self`.
    ///
    /// Panics if the lengths do not match.
    pub fn copy_from(&self, encoder: &mut CommandEncoder, source: &GpuTensor<T, DIM>)
    where
        T: Pod,
    {
        assert_eq!(self.len(), source.len());
        encoder.copy_buffer_to_buffer(&source.buffer, 0, &self.buffer, 0, self.bytes_len())
    }

    /// Queues a buffer-to-buffer copy from `source` to `self`.
    pub fn copy_from_view<'a>(
        &self,
        encoder: &mut CommandEncoder,
        source: impl Into<GpuTensorView<'a, T, DIM>>,
    ) where
        T: Pod,
    {
        let source = source.into();
        assert_eq!(
            source.view_shape.size[0],
            if DIM == 0 { 1 } else { self.shape[0] }
        );

        encoder.copy_buffer_to_buffer(
            source.buffer,
            source.view_shape.offset as BufferAddress * size_of::<T>() as BufferAddress,
            &self.buffer,
            0,
            self.bytes_len(),
        )
    }

    /// The tensor’s shape (typically `[num_rows, num_cols, ...]`).
    pub fn shape(&self) -> [u32; DIM] {
        self.shape
    }

    /// The tensor’s underlying wgpu buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Extracts the underlying buffer.
    pub fn into_inner(self) -> Buffer {
        self.buffer
    }

    /// Builds a tensor view sharing the same shape, stride, and buffer, as `self`.
    pub fn as_view(&self) -> GpuTensorView<T, DIM> {
        self.into()
    }

    /// Reads the buffer’s content into a vector.
    pub async fn read(&self, device: &Device) -> anyhow::Result<Vec<T>>
    where
        T: Pod,
    {
        // TODO: could probably be optimized?
        let buffer_slice = self.buffer.slice(..);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let (sender, receiver) = async_channel::bounded(1);
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
                sender.send_blocking(v).unwrap()
            });
            device.poll(wgpu::Maintain::wait()).panic_on_timeout();
            receiver.recv().await?.unwrap();
        }
        #[cfg(target_arch = "wasm32")]
        {
            device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        }

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.buffer.unmap();
        Ok(result)
    }
}

impl<'a, T, const DIM: usize> From<&'a GpuTensor<T, DIM>> for GpuTensorView<'a, T, DIM> {
    fn from(val: &'a GpuTensor<T, DIM>) -> Self {
        let mut size = [1; 2];
        let mut stride = 0;

        if DIM >= 1 {
            size[0] = val.shape[0];
        }

        if DIM >= 2 {
            stride = val.shape[0];
            size[1] = val.shape[1];
        }

        GpuTensorView {
            view_shape: ViewShape {
                size,
                stride,
                offset: 0,
            },
            buffer: &val.buffer,
            phantom: PhantomData,
        }
    }
}

/// A view over a tensor.
///
/// This is typically useful to extract a single matrix or column from a tensor. Note that,
/// currently, two elements from the same rows are required to be consecutive (row stride = 1).
#[derive(Copy, Clone)]
pub struct GpuTensorView<'a, T, const DIM: usize> {
    view_shape: ViewShape,
    buffer: &'a Buffer,
    phantom: PhantomData<T>,
}

impl<'a, T, const DIM: usize> GpuTensorView<'a, T, DIM> {
    /// The view’s shape.
    pub fn shape(&self) -> ViewShape {
        self.view_shape
    }

    /// The view’s underlying buffer.
    pub fn buffer(&self) -> &'a Buffer {
        self.buffer
    }
}

impl<'a, T> GpuVectorView<'a, T> {
    /// Is this view empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of elements in this vector view.
    pub fn len(&self) -> u32 {
        self.view_shape.size[0]
    }
}

impl<T> GpuMatrix<T> {
    /// Allocates a new matrix on the gpu with uninitialized elements.
    pub fn uninit(device: &Device, nrows: u32, ncols: u32, usage: BufferUsages) -> Self
    where
        T: Pod,
    {
        TensorBuilder::matrix(nrows, ncols, usage).build(device)
    }

    /// Allocates a new matrix on the gpu initialized from `matrix`.
    pub fn init<R: Dim, C: Dim, S: Storage<T, R, C> + IsContiguous>(
        device: &Device,
        matrix: &Matrix<T, R, C, S>,
        usage: BufferUsages,
    ) -> Self
    where
        T: Pod + nalgebra::Scalar,
    {
        TensorBuilder::matrix(matrix.nrows() as u32, matrix.ncols() as u32, usage)
            .build_init(device, matrix.as_slice())
    }

    /// Takes a view over the `i`-th column of `self`.
    pub fn column(&self, i: u32) -> GpuVectorView<T> {
        GpuTensorView {
            view_shape: ViewShape {
                size: [self.shape[0], 1],
                stride: 1,
                offset: self.shape[0] * i,
            },
            buffer: &self.buffer,
            phantom: PhantomData,
        }
    }
}

impl<T> GpuVector<T> {
    /// Allocates a new vector on the gpu initialized from `vector`.
    ///
    /// If `T` implements `Pod`, use [`GpuMatrix::init`] instead.
    pub fn encase(device: &Device, vector: impl AsRef<[T]>, usage: BufferUsages) -> Self
    where
        T: ShaderType + ShaderSize + WriteInto,
    {
        let vector = vector.as_ref();
        TensorBuilder::vector(vector.len() as u32, usage).build_encase(device, vector)
    }

    /// Allocates a new uninitialized vector on the gpu for `len` elements of type `T`.
    pub fn uninit(device: &Device, len: u32, usage: BufferUsages) -> Self
    where
        T: Pod,
    {
        TensorBuilder::vector(len, usage).build(device)
    }

    /// Allocates a new vector on the gpu initialized from `vector`.
    ///
    /// If `T` does not implement `Pod`, use [`GpuMatrix::encase`] instead.
    pub fn init(device: &Device, vector: impl AsRef<[T]>, usage: BufferUsages) -> Self
    where
        T: Pod,
    {
        let v = vector.as_ref();
        TensorBuilder::vector(v.len() as u32, usage).build_init(device, v.as_ref())
    }

    /// Takes a view, over this vector, with `num_rows` rows starting at row `first_row`.
    pub fn rows(&self, first_row: u32, num_rows: u32) -> GpuVectorView<T> {
        GpuTensorView {
            view_shape: ViewShape {
                size: [num_rows, 1],
                stride: self.shape[0],
                offset: first_row,
            },
            buffer: &self.buffer,
            phantom: PhantomData,
        }
    }
}

impl<T> GpuScalar<T> {
    /// Allocates a new gpu storage buffer with a single uninitialized element.
    pub fn uninit(device: &Device, usage: BufferUsages) -> Self
    where
        T: Pod,
    {
        TensorBuilder::scalar(usage).build(device)
    }

    /// Allocates a new gpu storage buffer with a single element initialized to `value`.
    pub fn init(device: &Device, value: T, usage: BufferUsages) -> Self
    where
        T: Pod,
    {
        TensorBuilder::scalar(usage).build_init(device, &[value])
    }
}
