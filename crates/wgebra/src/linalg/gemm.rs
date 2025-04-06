use crate::linalg::shape::Shape;
use bytemuck::Pod;
use wgcore::kernel::{KernelDispatch, KernelInvocationQueue};
use wgcore::tensor::GpuCubeView;
use wgcore::Shader;
use wgpu::{ComputePass, ComputePipeline};

#[derive(Shader)]
#[shader(derive(Shape), src = "gemm.wgsl", composable = false)]
/// Shader for computing the product of a matrix and a vector.
pub struct Gemm {
    /// The compute pipeline for `matrix * vector`.
    pub gemm: ComputePipeline,
    pub gemm_fast: ComputePipeline,
    pub gemm_tr: ComputePipeline,
    pub gemm_tr_fast: ComputePipeline,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GemmVariant {
    Gemm,
    GemmFast,
    GemmTr,
    GemmTrFast,
}

impl Gemm {
    /// Dispatch this shader to compute `out = m * v`.
    pub fn dispatch<'a, 'b, T: Pod>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        pass: &mut ComputePass,
        out: impl Into<GpuCubeView<'b, T>>,
        m1: impl Into<GpuCubeView<'b, T>>,
        m2: impl Into<GpuCubeView<'b, T>>,
    ) {
        self.dispatch_generic(queue, pass, out, m1, m2, GemmVariant::Gemm)
    }

    /// Dispatch this shader to compute `out = tr(m) * v`.
    pub fn dispatch_tr<'a, 'b, T: Pod>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        pass: &mut ComputePass,
        out: impl Into<GpuCubeView<'b, T>>,
        m1: impl Into<GpuCubeView<'b, T>>,
        m2: impl Into<GpuCubeView<'b, T>>,
    ) {
        self.dispatch_generic(queue, pass, out, m1, m2, GemmVariant::GemmTr)
    }

    pub fn dispatch_generic<'a, 'b, T: Pod>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
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

        let out_shape_buf = queue.shape_buffer(out.shape());
        let m1_shape_buf = queue.shape_buffer(m1.shape());
        let m2_shape_buf = queue.shape_buffer(m2.shape());

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

        KernelDispatch::new(queue.device(), pass, pipeline)
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
    use wgcore::kernel::{CommandEncoderExt, KernelInvocationQueue};
    use wgcore::tensor::TensorBuilder;
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_gemm() {
        let gpu = GpuInstance::new().await.unwrap();
        let gemm = super::Gemm::from_device(gpu.device()).unwrap();
        let mut queue = KernelInvocationQueue::new(gpu.device());

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
                &mut queue,
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
