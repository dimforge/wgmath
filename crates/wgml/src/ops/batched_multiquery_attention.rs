use crate::models::llama2::cpu::softmax;
use bytemuck::Pod;
use nalgebra::{DMatrix, DVector};
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::{GpuMatrix, GpuScalar, GpuVector};
use wgcore::Shader;
use wgebra::linalg::Shape;
use wgpu::ComputePipeline;

#[derive(Shader)]
#[shader(
    derive(Shape),
    src = "batched_multiquery_attention.wgsl",
    composable = false
)]
/// Shader implementing batched multi-query attention.
pub struct BatchedMultiqueryAttention {
    /// The compute pipeline representing the batched multi-query attention.
    pub main: ComputePipeline,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq, Eq, Debug)]
/// Parameters needed to run the [`BatchedMultiqueryAttention`] kernel. Matches the layout of the
/// corresponding WGSL struct.
pub struct BatchedMultiqueryAttentionParams {
    pub seq_len: u32,
    pub kv_dim: u32,
    pub kv_mul: u32,
    pub n_heads: u32,
    pub head_size: u32,
    pub pos: u32,
}

impl BatchedMultiqueryAttention {
    pub fn queue<'a, 'b, T: Pod>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        params: &GpuScalar<BatchedMultiqueryAttentionParams>,
        q: &GpuVector<T>,
        key_cache: &GpuMatrix<T>,
        value_cache: &GpuMatrix<T>,
        attn: &GpuMatrix<T>,
        xb: &GpuVector<T>,
    ) {
        KernelInvocationBuilder::new(queue, &self.main)
            .bind0([
                params.buffer(),
                q.buffer(),
                key_cache.buffer(),
                value_cache.buffer(),
                attn.buffer(),
                xb.buffer(),
            ])
            .queue(1);
    }

    pub fn run_cpu(
        params: &BatchedMultiqueryAttentionParams,
        q: &DVector<f32>,
        key_cache: &DMatrix<f32>,
        value_cache: &DMatrix<f32>,
        attn: &mut DMatrix<f32>,
        xb: &mut DVector<f32>,
    ) {
        // The number of embedding vector elements associated to each query head.
        let head_size = params.head_size as usize;
        // The number of query head associated to one key/value head.
        let kv_mul = params.kv_mul as usize;

        // Multihead attention. Iterate over all head.
        // TODO: in llama2.c, each head is iterated on in parallel.
        for h in 0..params.n_heads as usize {
            // Get the query vector for this head.
            let q = q.rows(h * head_size, head_size);
            // Attention scores for this head.
            let mut att = attn.column_mut(h);

            // Iterate over all timesteps (tokens in the sequence), including the current one, but
            // not past the current one due to causality.
            // See the KV cache explanation there: https://youtu.be/Mn_9W1nCFLo?si=3n4GH9f2OzMb5Np0&t=2940
            // -> This is iterating through all the green columns (from K^t) that are the rotated
            //    (by RoPE). The values set in this loop into the `att` variable here (attention
            //    scores) are the elements in the pink row (at the bottom of the QK^t matrix) divide
            //    by sqrt(params.head_size) (in other words, this is what’s given to softmax afterward.
            for t in 0..=params.pos as usize {
                // Get the key vector for this head and at this timestep.
                let k = key_cache.column(t); // TODO: does key_cache have the right dim?
                let k_head = k.rows((h / kv_mul) * head_size, head_size);

                // Calculate the attention score as the dot product of q and k.
                let mut score = q.dot(&k_head);
                score /= (head_size as f32).sqrt();
                // Save the score to the attention buffer.
                att[t] = score;
            }

            // Softmax the scores to get attention weights from 0..=pos inclusively.
            softmax(&mut att.rows_mut(0, params.pos as usize + 1));

            // Weighted sum of the values, store back into xb.
            // /!\ xb is now changing semantic, storing the weighted sums for all the heads.
            //       Now xb contains the "Attention 4" row from https://youtu.be/Mn_9W1nCFLo?si=550ar5aUg1I1k60l&t=2940.
            let mut xb = xb.rows_mut(h * head_size, head_size);
            xb.fill(0.0);
            for t in 0..=params.pos as usize {
                let v = value_cache.column(t);
                let v_head = v.rows((h / kv_mul) * head_size, head_size);
                xb.axpy(att[t], &v_head, 1.0);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::ops::BatchedMultiqueryAttentionParams;
    use nalgebra::{DMatrix, DVector};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::KernelInvocationQueue;
    use wgcore::tensor::{GpuMatrix, GpuScalar, GpuVector};
    use wgcore::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_attention() {
        let gpu = GpuInstance::new().await.unwrap();
        let batched_multihead_attention =
            super::BatchedMultiqueryAttention::from_device(gpu.device());
        let mut queue = KernelInvocationQueue::new(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        let params = BatchedMultiqueryAttentionParams {
            seq_len: 1024,
            kv_dim: 768,
            kv_mul: 1,
            n_heads: 12,
            head_size: 64,
            pos: 7,
        };

        let q = DVector::new_random(params.kv_dim as usize);
        let key_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let value_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let mut attn = DMatrix::new_random(params.seq_len as usize, params.n_heads as usize);
        let mut xb = DVector::new_random(params.kv_dim as usize);

        let gpu_params = GpuScalar::init(gpu.device(), params, BufferUsages::UNIFORM);
        let gpu_q = GpuVector::init(gpu.device(), q.as_slice(), BufferUsages::STORAGE);
        let gpu_key_cache = GpuMatrix::init(gpu.device(), &key_cache, BufferUsages::STORAGE);
        let gpu_value_cache = GpuMatrix::init(gpu.device(), &value_cache, BufferUsages::STORAGE);
        let gpu_attn = GpuMatrix::init(
            gpu.device(),
            &attn,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let gpu_xb = GpuVector::init(
            gpu.device(),
            xb.as_slice(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let gpu_staging_xb = GpuVector::uninit(
            gpu.device(),
            xb.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        let gpu_staging_attn = GpuMatrix::uninit(
            gpu.device(),
            attn.nrows() as u32,
            attn.ncols() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        batched_multihead_attention.queue(
            &mut queue,
            &gpu_params,
            &gpu_q,
            &gpu_key_cache,
            &gpu_value_cache,
            &gpu_attn,
            &gpu_xb,
        );

        queue.encode(&mut encoder, None);
        gpu_staging_xb.copy_from(&mut encoder, &gpu_xb);
        gpu_staging_attn.copy_from(&mut encoder, &gpu_attn);

        gpu.queue().submit(Some(encoder.finish()));

        super::BatchedMultiqueryAttention::run_cpu(
            &params,
            &q,
            &key_cache,
            &value_cache,
            &mut attn,
            &mut xb,
        );

        approx::assert_relative_eq!(
            DVector::from(gpu_staging_xb.read(gpu.device()).await.unwrap()),
            xb,
            epsilon = 1.0e-5
        );

        approx::assert_relative_eq!(
            DMatrix::from_vec(
                attn.nrows(),
                attn.ncols(),
                gpu_staging_attn.read(gpu.device()).await.unwrap()
            ),
            attn,
            epsilon = 1.0e-5
        );
    }
}
