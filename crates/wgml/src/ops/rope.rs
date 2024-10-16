use crate::ops::BatchedMultiqueryAttention;
use bytemuck::Pod;
use nalgebra::{vector, DVector, DVectorViewMut, Rotation2};
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::{GpuScalar, GpuVectorView};
use wgcore::Shader;
use wgebra::linalg::Shape;
use wgpu::ComputePipeline;

#[derive(Shader)]
#[shader(derive(Shape), src = "rope.wgsl", composable = false)]
/// Shader implementing the Rotary Positional Encoding kernel.
pub struct RoPE {
    pub main: ComputePipeline,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
/// Parameters needed to run the [`RoPE`] kernel. Matches the layout of the
/// corresponding WGSL struct.
pub struct RoPEShape {
    pub head_size: u32,
    pub kv_dim: u32,
    pub pos: u32,
}

impl RoPE {
    pub fn queue<'a, 'b, T: Pod>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        shape: &GpuScalar<RoPEShape>,
        in_out_q: impl Into<GpuVectorView<'b, T>>,
        in_out_k: impl Into<GpuVectorView<'b, T>>,
    ) {
        let in_out_q = in_out_q.into();
        let in_out_k = in_out_k.into();

        assert_eq!(in_out_q.len() % 2, 0);
        assert_eq!(in_out_k.len() % 2, 0);
        assert!(
            in_out_q.len() >= in_out_k.len(),
            "The Query vector must be larger than, or as large as, the Key vector."
        );

        let shape_q = queue.shape_buffer(in_out_q.shape());
        let shape_k = queue.shape_buffer(in_out_k.shape());

        KernelInvocationBuilder::new(queue, &self.main)
            .bind0([
                &shape_q,
                &shape_k,
                shape.buffer(),
                in_out_q.buffer(),
                in_out_k.buffer(),
            ])
            // Use `q` as the reference for the workgroup count since it is a bigger vector.
            .queue((in_out_q.len() / 2).div_ceil(64));
    }

    // Rotary Positional Encoding (RoPE): complex-valued rotate q and k in each head.
    pub fn run_cpu(
        q: &mut DVector<f32>,
        k: &mut DVectorViewMut<f32>,
        head_size: usize,
        dim: usize,
        kv_dim: usize,
        pos: usize,
    ) {
        for i in (0..dim).step_by(2) {
            // For RoPE, we have one rotation matrix like https://youtu.be/Mn_9W1nCFLo?si=GLIXuFLGVG8q6v2u&t=1963
            // for each head. So we need to transform `i` into the corresponding index within
            // the head.
            let head_dim = (i % head_size) as f32;
            // Not that the formulae from the video linked above would be:
            //     10000.0.powf(-2.0 * ((i / 2) as f32 - 1.0) / dim as f32)
            // Although in the paper shown in the video, their index is 1-based which his why thy
            // have to subtract 1.0 whereas we don’t need to.The `i / 2` and multiplication by 2.0
            // are both accounted for by stepping only on even values for `i`.
            // Therefore, the formulae below is equivalent to the RoPE paper’s formulae.
            let theta = 10000.0_f32.powf(-head_dim / head_size as f32);
            let m_theta = pos as f32 * theta;
            let rot = Rotation2::new(m_theta);

            let qi = vector![q[i], q[i + 1]];
            let mut out_q = q.fixed_rows_mut::<2>(i);
            out_q.copy_from(&(rot * qi));

            // When i >= kv_dim, we are done rotating all the elements from the keys. That’s
            // because there are less key heads than query heads, but each key head sub-vector has
            // the same dimension as the query head (they loose dimension when multiplied with the
            // key weight matrices).
            if i < kv_dim {
                let ki = vector![k[i], k[i + 1]];
                let mut out_k = k.fixed_rows_mut::<2>(i);
                out_k.copy_from(&(rot * ki));
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::RoPEShape;
    use crate::ops::RoPE;
    use nalgebra::DVector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::KernelInvocationQueue;
    use wgcore::tensor::TensorBuilder;
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_rope() {
        let gpu = GpuInstance::new().await.unwrap();
        let rope = super::RoPE::from_device(gpu.device());
        let mut queue = KernelInvocationQueue::new(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const HEAD_SIZE: u32 = 128;
        const LEN_Q: u32 = 13 * HEAD_SIZE;
        const LEN_K: u32 = 9 * HEAD_SIZE;

        let rope_indices = RoPEShape {
            head_size: HEAD_SIZE,
            kv_dim: LEN_K,
            pos: 10,
        };

        let mut q = DVector::new_random(LEN_Q as usize);
        let mut k = DVector::new_random(LEN_K as usize);

        let gpu_indices =
            TensorBuilder::scalar(BufferUsages::UNIFORM).build_init(gpu.device(), &[rope_indices]);
        let gpu_q = TensorBuilder::vector(LEN_Q, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
            .build_init(gpu.device(), q.as_slice());
        let gpu_k = TensorBuilder::vector(LEN_K, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
            .build_init(gpu.device(), k.as_slice());
        let staging_q =
            TensorBuilder::vector(LEN_Q, BufferUsages::MAP_READ | BufferUsages::COPY_DST)
                .build(gpu.device());
        let staging_k =
            TensorBuilder::vector(LEN_K, BufferUsages::MAP_READ | BufferUsages::COPY_DST)
                .build(gpu.device());

        rope.queue(&mut queue, &gpu_indices, &gpu_q, &gpu_k);

        queue.encode(&mut encoder, None);
        staging_q.copy_from(&mut encoder, &gpu_q);
        staging_k.copy_from(&mut encoder, &gpu_k);

        gpu.queue().submit(Some(encoder.finish()));

        let result_q = DVector::from(staging_q.read(gpu.device()).await.unwrap());
        let result_k = DVector::from(staging_k.read(gpu.device()).await.unwrap());

        RoPE::run_cpu(
            &mut q,
            &mut k.rows_mut(0, LEN_K as usize),
            rope_indices.head_size as usize,
            LEN_Q as usize,
            rope_indices.kv_dim as usize,
            rope_indices.pos as usize,
        );

        // TODO: why is the epsilon so high? Is it a difference in sin/cos implementations?
        approx::assert_relative_eq!(result_q, q, epsilon = 1.0e-5);
        approx::assert_relative_eq!(result_k, k, epsilon = 1.0e-5);
    }
}
