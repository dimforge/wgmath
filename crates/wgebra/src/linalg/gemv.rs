use crate::linalg::shape::Shape;
use bytemuck::Pod;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::{GpuMatrixView, GpuVectorView};
use wgcore::Shader;
use wgpu::ComputePipeline;

#[derive(Shader)]
#[shader(derive(Shape), src = "gemv.wgsl", composable = false)]
/// Shader for computing the product of a matrix and a vector.
pub struct Gemv {
    /// The compute pipeline for `matrix * vector`.
    pub gemv0: ComputePipeline,
}

impl Gemv {
    /// Queues this shader to compute `out = m * v`.
    pub fn queue<'a, 'b, T: Pod>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        out: impl Into<GpuVectorView<'b, T>>,
        m: impl Into<GpuMatrixView<'b, T>>,
        v: impl Into<GpuVectorView<'b, T>>,
    ) {
        let out = out.into();
        let m = m.into();
        let v = v.into();

        assert_eq!(
            m.shape().size[1],
            v.shape().size[0],
            "Gemv: dimension mismatch."
        );
        assert_eq!(
            out.shape().size[0],
            m.shape().size[0],
            "Gemv: dimension mismatch."
        );
        let out_shape_buf = queue.shape_buffer(out.shape());
        let m_shape_buf = queue.shape_buffer(m.shape());
        let v_shape_buf = queue.shape_buffer(v.shape());

        KernelInvocationBuilder::new(queue, &self.gemv0)
            .bind0([
                &out_shape_buf,
                &m_shape_buf,
                &v_shape_buf,
                out.buffer(),
                m.buffer(),
                v.buffer(),
            ])
            .queue(m.shape().size[0].div_ceil(64));
    }
}

#[cfg(test)]
mod test {
    use nalgebra::{DMatrix, DVector};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::KernelInvocationQueue;
    use wgcore::tensor::TensorBuilder;
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_gemv() {
        let gpu = GpuInstance::new().await.unwrap();
        let gemv = super::Gemv::from_device(gpu.device());
        let mut queue = KernelInvocationQueue::new(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const NROWS: u32 = 1507;
        const NCOLS: u32 = 2333;

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

        gemv.queue(&mut queue, &result, &m, &v);

        queue.encode(&mut encoder, None);
        staging.copy_from(&mut encoder, &result);

        let t0 = std::time::Instant::now();
        gpu.queue().submit(Some(encoder.finish()));
        let gpu_result = staging.read(gpu.device()).await.unwrap();
        println!("Gpu time: {}", t0.elapsed().as_secs_f32());

        let t0 = std::time::Instant::now();
        let cpu_result = /* lhs_cpu + */ m_cpu * v_cpu;
        println!("Cpu time: {}", t0.elapsed().as_secs_f32());

        approx::assert_relative_eq!(DVector::from(gpu_result), cpu_result, epsilon = 1.0e-3);
    }
}
