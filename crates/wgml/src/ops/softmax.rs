use bytemuck::Pod;
use nalgebra::{Dyn, StorageMut, Vector};
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::GpuVectorView;
use wgcore::Shader;
use wgebra::linalg::Shape;
use wgpu::ComputePipeline;

#[derive(Shader)]
#[shader(derive(Shape), src = "softmax.wgsl", composable = false)]
/// Shader implementing the softmax kernel.
pub struct SoftMax {
    pub main: ComputePipeline,
}

impl SoftMax {
    pub fn queue<'a, 'b, T: Pod>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        in_out_vec: impl Into<GpuVectorView<'b, T>>,
    ) {
        let in_out_vec = in_out_vec.into();
        let shape_buf = queue.shape_buffer(in_out_vec.shape());
        KernelInvocationBuilder::new(queue, &self.main)
            .bind0([&shape_buf, in_out_vec.buffer()])
            .queue(1);
    }

    /// The softmax function.
    ///
    /// Converts a set of real number into a probability distribution.
    /// See <https://fr.wikipedia.org/wiki/Fonction_softmax>
    pub fn run_cpu<S: StorageMut<f32, Dyn>>(vals: &mut Vector<f32, Dyn, S>) {
        // Note that llama2.c also introduces a bias based on the max value
        // to improve numerical stability. So it is effectively computing:
        // softmax(z) = (e^z - max) / (e^z - max).sum()
        let max_val = vals.max();
        let mut sum = 0.0;

        vals.apply(|x| {
            *x = (*x - max_val).exp();
            sum += *x;
        });

        *vals /= sum;
    }
}

#[cfg(test)]
mod test {
    use crate::ops::SoftMax;
    use nalgebra::DVector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::KernelInvocationQueue;
    use wgcore::tensor::TensorBuilder;
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_softmax() {
        let gpu = GpuInstance::new().await.unwrap();
        let softmax = super::SoftMax::from_device(gpu.device());
        let mut queue = KernelInvocationQueue::new(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const LEN: u32 = 1757;

        let v0 = DVector::new_random(LEN as usize);
        let gpu_v0 = TensorBuilder::vector(LEN, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
            .build_init(gpu.device(), v0.as_slice());
        let staging = TensorBuilder::vector(LEN, BufferUsages::MAP_READ | BufferUsages::COPY_DST)
            .build(gpu.device());

        softmax.queue(&mut queue, &gpu_v0);

        queue.encode(&mut encoder, None);
        staging.copy_from(&mut encoder, &gpu_v0);

        gpu.queue().submit(Some(encoder.finish()));

        let mut cpu_result = v0;
        SoftMax::run_cpu(&mut cpu_result);

        approx::assert_relative_eq!(
            DVector::from(staging.read(gpu.device()).await.unwrap()),
            cpu_result,
            epsilon = 1.0e-7
        );
    }
}
