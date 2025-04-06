use crate::linalg::shape::Shape;
use bytemuck::Pod;
use wgcore::kernel::{KernelDispatch, KernelInvocationQueue};
use wgcore::tensor::GpuCubeView;
use wgcore::Shader;
use wgpu::{ComputePass, ComputePipeline};

#[derive(Shader)]
#[shader(derive(Shape), src = "gemv.wgsl", composable = false)]
/// Shader for computing the product of a matrix and a vector.
pub struct Gemv {
    /// The compute pipeline for `matrix * vector`.
    pub gemv: ComputePipeline,
    pub gemv_fast: ComputePipeline,
    pub gemv_tr: ComputePipeline,
    pub gemv_tr_fast: ComputePipeline,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GemvVariant {
    Gemv,
    GemvFast,
    GemvTr,
    GemvTrFast,
}

impl Gemv {
    /// Dispatches this shader to compute `out = m * v`.
    pub fn dispatch<'a, 'b, T: Pod>(
        &'a self,
        queue: &KernelInvocationQueue<'a>,
        pass: &mut ComputePass,
        out: impl Into<GpuCubeView<'b, T>>,
        m: impl Into<GpuCubeView<'b, T>>,
        v: impl Into<GpuCubeView<'b, T>>,
    ) {
        self.dispatch_generic(queue, pass, out, m, v, GemvVariant::Gemv)
    }

    /// Dispatches this shader to compute `out = tr(m) * v`.
    pub fn dispatch_tr<'a, 'b, T: Pod>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        pass: &mut ComputePass,
        out: impl Into<GpuCubeView<'b, T>>,
        m: impl Into<GpuCubeView<'b, T>>,
        v: impl Into<GpuCubeView<'b, T>>,
    ) {
        self.dispatch_generic(queue, pass, out, m, v, GemvVariant::GemvTr)
    }

    pub fn dispatch_generic<'a, 'b, T: Pod>(
        &'a self,
        queue: &KernelInvocationQueue<'a>,
        pass: &mut ComputePass,
        out: impl Into<GpuCubeView<'b, T>>,
        m: impl Into<GpuCubeView<'b, T>>,
        v: impl Into<GpuCubeView<'b, T>>,
        mut variant: GemvVariant,
    ) {
        let out = out.into();
        let m = m.into();
        let v = v.into();
        let [out_nrows, out_ncols, out_nmats] = out.shape().size;

        // Check dimensions.
        {
            let v_rows = v.shape().size[0];
            let (m_rows, m_cols) = match variant {
                GemvVariant::Gemv | GemvVariant::GemvFast => (m.shape().size[0], m.shape().size[1]),
                GemvVariant::GemvTr | GemvVariant::GemvTrFast => {
                    (m.shape().size[1], m.shape().size[0])
                }
            };

            assert_eq!(m_cols, v_rows, "Gemv: dimension mismatch.");
            assert_eq!(m_rows, out_nrows, "Gemv: dimension mismatch.");
        }

        let out_shape_buf = queue.shape_buffer(out.shape());
        let m_shape_buf = queue.shape_buffer(m.shape());
        let v_shape_buf = queue.shape_buffer(v.shape());

        // More compatibility check.
        // TODO: switch to a fallback version when any of these check don’t pass.
        if variant == GemvVariant::GemvTrFast {
            // Switch to the non-fast version if we dont have the right alignment.
            if m.shape().size[0] % (WORKGROUP_SIZE * 4) != 0 {
                variant = GemvVariant::GemvTr;
            }
        }

        let pipeline = match variant {
            GemvVariant::Gemv => &self.gemv,
            GemvVariant::GemvFast => &self.gemv_fast,
            GemvVariant::GemvTr => &self.gemv_tr,
            GemvVariant::GemvTrFast => &self.gemv_tr_fast,
        };

        const WORKGROUP_SIZE: u32 = 32;

        let dispatch = match variant {
            // Each thread handles 4 rows of the matrix, there is no special
            // consideration of workgroup threads.
            GemvVariant::Gemv | GemvVariant::GemvTr => out_nrows.div_ceil(WORKGROUP_SIZE),
            // Each workgroup handles 4 entire rows of the matrix.
            GemvVariant::GemvFast | GemvVariant::GemvTrFast => {
                // TODO: automatically fallback to the non-fast version if this condition isn’t met?
                assert_eq!(out_nrows % 4, 0);
                out_nrows.div_ceil(4)
            }
        };

        KernelDispatch::new(queue.device(), pass, pipeline)
            .bind0([
                &out_shape_buf,
                &m_shape_buf,
                &v_shape_buf,
                out.buffer(),
                m.buffer(),
                v.buffer(),
            ])
            .dispatch([dispatch, out_ncols, out_nmats]);
    }
}

#[cfg(test)]
mod test {
    use crate::GemvVariant;
    use nalgebra::{DMatrix, DVector};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::{CommandEncoderExt, KernelInvocationQueue};
    use wgcore::tensor::TensorBuilder;
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_gemv() {
        let gpu = GpuInstance::new().await.unwrap();
        let gemv = super::Gemv::from_device(gpu.device()).unwrap();
        let mut queue = KernelInvocationQueue::new(gpu.device());

        const NROWS: u32 = 1024;
        const NCOLS: u32 = 1024;

        let m_cpu = DMatrix::<f32>::new_random(NROWS as usize, NCOLS as usize);
        let v_cpu = DVector::<f32>::new_random(NCOLS as usize);
        let lhs_cpu = DVector::<f32>::new_random(NROWS as usize);

        let m = TensorBuilder::matrix(NROWS, NCOLS, BufferUsages::STORAGE)
            .build_init(gpu.device(), m_cpu.as_slice());
        let v = TensorBuilder::vector(v_cpu.nrows() as u32, BufferUsages::STORAGE)
            .build_init(gpu.device(), v_cpu.as_slice());
        let result = TensorBuilder::vector(NROWS, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
            .build_init(gpu.device(), lhs_cpu.as_slice());
        let staging = TensorBuilder::vector(NROWS, BufferUsages::MAP_READ | BufferUsages::COPY_DST)
            .build(gpu.device());

        for variant in [
            GemvVariant::Gemv,
            GemvVariant::GemvTr,
            GemvVariant::GemvFast,
            GemvVariant::GemvTrFast,
        ] {
            println!("Checking variant: {:?}", variant);
            let mut encoder = gpu.device().create_command_encoder(&Default::default());
            let mut pass = encoder.compute_pass("test", None);
            gemv.dispatch_generic(&mut queue, &mut pass, &result, &m, &v, variant);
            drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

            staging.copy_from(&mut encoder, &result);

            gpu.queue().submit(Some(encoder.finish()));
            let gpu_result = staging.read(gpu.device()).await.unwrap();
            let cpu_result = match variant {
                GemvVariant::Gemv | GemvVariant::GemvFast => &m_cpu * &v_cpu,
                GemvVariant::GemvTr | GemvVariant::GemvTrFast => m_cpu.tr_mul(&v_cpu),
            };

            approx::assert_relative_eq!(DVector::from(gpu_result), cpu_result, epsilon = 1.0e-3);
        }
    }
}
