use bytemuck::Pod;
use nalgebra::DVector;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::GpuVectorView;
use wgcore::Shader;
use wgebra::linalg::Shape;
use wgpu::ComputePipeline;

#[derive(Shader)]
#[shader(derive(Shape), src = "layernorm.wgsl", composable = false)]
/// Shader implementing the layer normalization kernel.
pub struct LayerNorm {
    pub main: ComputePipeline,
}

impl LayerNorm {
    pub fn queue<'a, 'b, T: Pod>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        out_vec: impl Into<GpuVectorView<'b, T>>,
        in_vec: impl Into<GpuVectorView<'b, T>>,
    ) {
        let in_vec = in_vec.into();
        let out_vec = out_vec.into();

        assert_eq!(
            in_vec.shape().size[0],
            out_vec.shape().size[0],
            "LayerNorm: dimension mismatch."
        );

        let in_shape = queue.shape_buffer(in_vec.shape());
        let out_shape = queue.shape_buffer(out_vec.shape());
        KernelInvocationBuilder::new(queue, &self.main)
            .bind0([&in_shape, &out_shape, in_vec.buffer(), out_vec.buffer()])
            .queue(1);
    }

    /// The layernorm function.
    ///
    /// See <https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html> for details on the
    /// math.
    pub fn run_cpu(res: &mut DVector<f32>, v: &DVector<f32>) {
        const NUDGE_FACTOR: f32 = 1.0e-5;
        let mean = v.mean();
        res.zip_apply(v, |y, v| *y = v - mean);
        let variance = res.norm_squared() / (res.len() as f32);
        let scale = 1.0 / (variance + NUDGE_FACTOR).sqrt();
        *res *= scale;
    }
}

#[cfg(test)]
mod test {
    use crate::ops::LayerNorm;
    use nalgebra::DVector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::KernelInvocationQueue;
    use wgcore::tensor::TensorBuilder;
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_layernorm() {
        let gpu = GpuInstance::new().await.unwrap();
        let layernorm = super::LayerNorm::from_device(gpu.device());
        let mut queue = KernelInvocationQueue::new(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const LEN: u32 = 1757;

        let v0 = DVector::new_random(LEN as usize);
        let out = DVector::new_random(LEN as usize);
        let gpu_v0 = TensorBuilder::vector(LEN, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
            .build_init(gpu.device(), v0.as_slice());
        let gpu_out = TensorBuilder::vector(LEN, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
            .build_init(gpu.device(), v0.as_slice());
        let staging = TensorBuilder::vector(LEN, BufferUsages::MAP_READ | BufferUsages::COPY_DST)
            .build(gpu.device());

        layernorm.queue(&mut queue, &gpu_out, &gpu_v0);

        queue.encode(&mut encoder, None);
        staging.copy_from(&mut encoder, &gpu_out);

        gpu.queue().submit(Some(encoder.finish()));

        let mut cpu_result = out;
        LayerNorm::run_cpu(&mut cpu_result, &v0);

        approx::assert_relative_eq!(
            DVector::from(staging.read(gpu.device()).await.unwrap()),
            cpu_result,
            epsilon = 1.0e-5
        );
    }
}
