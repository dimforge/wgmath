use crate::linalg::shape::Shape;
use bytemuck::Pod;
use wgcore::kernel::KernelDispatch;
use wgcore::shapes::ViewShapeBuffers;
use wgcore::tensor::GpuCubeView;
use wgcore::Shader;
use wgpu::{ComputePass, ComputePipeline, Device};

#[derive(Shader)]
#[shader(derive(Shape), src = "gemm.wgsl", composable = false)]
/// Shader for computing the product of two matrices.
pub struct Gemm {
    /// The compute pipeline for `matrix1 * matrix2`.
    pub gemm: ComputePipeline,
    /// A compute pipeline for `matrix1 * matrix2` leveraging workgroup reduction.
    pub gemm_fast: ComputePipeline,
    /// The compute pipeline for `transpose(matrix1) * matrix2`.
    pub gemm_tr: ComputePipeline,
    /// A compute pipeline for `transpose(matrix1) * matrix2` leveraging workgroup reduction.
    pub gemm_tr_fast: ComputePipeline,
}

/// Variants used to select the specific kernel to dispatch from the [`Gemm`] shader.

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GemmVariant {
    /// The compute pipeline for `matrix1 * matrix2`.
    Gemm,
    /// A compute pipeline for `matrix1 * matrix2` leveraging workgroup reduction.
    GemmFast,
    /// The compute pipeline for `transpose(matrix1) * matrix2`.
    GemmTr,
    /// A compute pipeline for `transpose(matrix1) * matrix2` leveraging workgroup reduction.
    GemmTrFast,
}

impl Gemm {
    /// Dispatch this shader to compute `out = m1 * m2`.
    pub fn dispatch<'a, 'b, T: Pod>(
        &'a self,
        device: &Device,
        shapes: &ViewShapeBuffers,
        pass: &mut ComputePass,
        out: impl Into<GpuCubeView<'b, T>>,
        m1: impl Into<GpuCubeView<'b, T>>,
        m2: impl Into<GpuCubeView<'b, T>>,
    ) {
        self.dispatch_generic(device, shapes, pass, out, m1, m2, GemmVariant::Gemm)
    }

    /// Dispatch this shader to compute `out = tr(m1) * m2`.
    pub fn dispatch_tr<'a, 'b, T: Pod>(
        &'a self,
        device: &Device,
        shapes: &ViewShapeBuffers,
        pass: &mut ComputePass,
        out: impl Into<GpuCubeView<'b, T>>,
        m1: impl Into<GpuCubeView<'b, T>>,
        m2: impl Into<GpuCubeView<'b, T>>,
    ) {
        self.dispatch_generic(device, shapes, pass, out, m1, m2, GemmVariant::GemmTr)
    }

    /// Dispatches the matrix-vector multiplication variant indicated by the given [`GemmVariant`].
    pub fn dispatch_generic<'a, 'b, T: Pod>(
        &'a self,
        device: &Device,
        shapes: &ViewShapeBuffers,
        pass: &mut ComputePass,
        out: impl Into<GpuCubeView<'b, T>>,
        m1: impl Into<GpuCubeView<'b, T>>,
        m2: impl Into<GpuCubeView<'b, T>>,
        variant: GemmVariant,
    ) {
        let out = out.into();
        let m1 = m1.into();
        let m2 = m2.into();
        let [out_rows, out_cols, out_mats] = out.shape().size;

        // Check dimensions.
        {
            let (m_rows, m_cols) = match variant {
                GemmVariant::Gemm | GemmVariant::GemmFast => {
                    (m1.shape().size[0], m1.shape().size[1])
                }
                GemmVariant::GemmTr | GemmVariant::GemmTrFast => {
                    (m1.shape().size[1], m1.shape().size[0])
                }
            };

            assert_eq!(m_cols, m2.shape().size[0], "Gemm: dimension mismatch.");
            assert_eq!(m_rows, out_rows, "Gemm: dimension mismatch.");
            assert_eq!(out_cols, m2.shape().size[1], "Gemm: dimension mismatch.");
            assert_eq!(out_mats, m1.shape().size[2], "Gemm: dimension mismatch.");
            assert_eq!(out_mats, m2.shape().size[2], "Gemm: dimension mismatch.");
        }

        let out_shape_buf = shapes.get(device, out.shape());
        let m1_shape_buf = shapes.get(device, m1.shape());
        let m2_shape_buf = shapes.get(device, m2.shape());

        let pipeline = match variant {
            GemmVariant::Gemm => &self.gemm,
            GemmVariant::GemmFast => &self.gemm_fast,
            GemmVariant::GemmTr => &self.gemm_tr,
            GemmVariant::GemmTrFast => &self.gemm_tr_fast,
        };

        let dispatch = match variant {
            // Each thread handles 4 rows of the matrix, there is no special
            // consideration of workgroup threads.
            GemmVariant::Gemm | GemmVariant::GemmTr => out_rows.div_ceil(64),
            // Each workgroup handles 4 entire rows of the matrix.
            GemmVariant::GemmFast | GemmVariant::GemmTrFast => out_rows.div_ceil(4),
        };

        KernelDispatch::new(device, pass, pipeline)
            .bind0([
                &out_shape_buf,
                &m1_shape_buf,
                &m2_shape_buf,
                out.buffer(),
                m1.buffer(),
                m2.buffer(),
            ])
            .dispatch([dispatch, out_mats, 1]);
    }
}

#[cfg(test)]
mod test {
    use crate::GemmVariant;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::CommandEncoderExt;
    use wgcore::shapes::ViewShapeBuffers;
    use wgcore::tensor::TensorBuilder;
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_gemm() {
        let gpu = GpuInstance::new().await.unwrap();
        let gemm = super::Gemm::from_device(gpu.device()).unwrap();
        let shapes = ViewShapeBuffers::new();

        const NROWS: u32 = 256;
        const NCOLS: u32 = 256;

        let m1_cpu = DMatrix::<f32>::new_random(NROWS as usize, NCOLS as usize);
        let m2_cpu = DMatrix::<f32>::new_random(NCOLS as usize, NROWS as usize);
        let lhs_cpu = DMatrix::<f32>::zeros(NROWS as usize, NROWS as usize);

        let m1 = TensorBuilder::matrix(NROWS, NCOLS, BufferUsages::STORAGE)
            .build_init(gpu.device(), m1_cpu.as_slice());
        let m2 = TensorBuilder::matrix(NCOLS, NROWS, BufferUsages::STORAGE)
            .build_init(gpu.device(), m2_cpu.as_slice());
        let result =
            TensorBuilder::matrix(NROWS, NROWS, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
                .build_init(gpu.device(), lhs_cpu.as_slice());
        let staging = TensorBuilder::matrix(
            NROWS,
            NROWS,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        )
        .build(gpu.device());

        for variant in [
            GemmVariant::Gemm,
            GemmVariant::GemmTr,
            GemmVariant::GemmFast,
            GemmVariant::GemmTrFast,
        ] {
            println!("Checking variant: {:?}", variant);
            let mut encoder = gpu.device().create_command_encoder(&Default::default());
            let mut pass = encoder.compute_pass("test", None);
            gemm.dispatch_generic(
                gpu.device(),
                &shapes,
                &mut pass,
                result.as_embedded_view(),
                m1.as_embedded_view(),
                m2.as_embedded_view(),
                variant,
            );
            drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

            staging.copy_from(&mut encoder, &result);

            gpu.queue().submit(Some(encoder.finish()));
            let gpu_result = staging.read(gpu.device()).await.unwrap();
            let cpu_result = match variant {
                GemmVariant::Gemm | GemmVariant::GemmFast => &m1_cpu * &m2_cpu,
                GemmVariant::GemmTr | GemmVariant::GemmTrFast => m1_cpu.tr_mul(&m2_cpu),
            };

            let gpu_result = DMatrix::from_vec(NROWS as usize, NROWS as usize, gpu_result);
            assert_relative_eq!(gpu_result, cpu_result, epsilon = 1.0e-3);
        }
    }
}
