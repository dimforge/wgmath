use crate::models::llama2::cpu::{Llama2Config, TransformerWeights};
use crate::ops::{
    BatchedMultiqueryAttention, BatchedMultiqueryAttentionParams, RmsNorm, RoPE, RoPEShape, Silu,
};
use naga_oil::compose::ComposerError;
use wgcore::kernel::KernelInvocationQueue;
use wgcore::tensor::{GpuMatrix, GpuScalar, GpuVector};
use wgcore::Shader;
use wgebra::linalg::{Gemv, OpAssign, OpAssignVariant};
use wgpu::{BufferUsages, Device};

pub struct Llama2State {
    /// Activation at current time stamp.
    pub x: GpuVector<f32>,
    /// Activation at current time stamp, inside a residual branch.
    xb: GpuVector<f32>,
    /// Additional buffer for convenience.
    xb2: GpuVector<f32>,
    /// Buffer for hidden dimension in the Feed-Forward net.
    hb: GpuVector<f32>,
    /// Another buffer for hidden dimension in the Feed-Forward net.
    hb2: GpuVector<f32>,
    /// Query.
    q: GpuVector<f32>,
    /// Scores/attention values.
    att: GpuMatrix<f32>,
    /// Output logits.
    logits: GpuVector<f32>,
    logits_readback: GpuVector<f32>,
    // KV cache. Each Vec contains `layer` elements.
    key_cache: Vec<GpuMatrix<f32>>,
    value_cache: Vec<GpuMatrix<f32>>,
    rope_shape: GpuScalar<RoPEShape>,
    attn_params: GpuScalar<BatchedMultiqueryAttentionParams>,
}

impl Llama2State {
    pub fn new(device: &Device, config: &Llama2Config) -> Self {
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_q_heads;
        const STORAGE: BufferUsages = BufferUsages::STORAGE;
        const UNIFORM: BufferUsages = BufferUsages::UNIFORM;

        Self {
            x: GpuVector::uninit(device, config.dim as u32, STORAGE | BufferUsages::COPY_DST),
            xb: GpuVector::uninit(device, config.dim as u32, STORAGE),
            xb2: GpuVector::uninit(device, config.dim as u32, STORAGE),
            hb: GpuVector::uninit(device, config.hidden_dim as u32, STORAGE),
            hb2: GpuVector::uninit(device, config.hidden_dim as u32, STORAGE),
            q: GpuVector::uninit(device, config.dim as u32, STORAGE),
            // TODO: for these two, the `kv_dim` doesn’t match the dimension in the field’s comment.
            key_cache: (0..config.n_layers)
                .map(|_| GpuMatrix::uninit(device, kv_dim as u32, config.seq_len as u32, STORAGE))
                .collect(),
            value_cache: (0..config.n_layers)
                .map(|_| GpuMatrix::uninit(device, kv_dim as u32, config.seq_len as u32, STORAGE))
                .collect(),
            att: GpuMatrix::uninit(
                device,
                config.seq_len as u32,
                config.n_q_heads as u32,
                STORAGE,
            ),
            logits: GpuVector::uninit(
                device,
                config.vocab_size as u32,
                STORAGE | BufferUsages::COPY_SRC,
            ),
            logits_readback: GpuVector::uninit(
                device,
                config.vocab_size as u32,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            ),
            rope_shape: GpuScalar::uninit(device, UNIFORM | BufferUsages::COPY_DST),
            attn_params: GpuScalar::uninit(device, UNIFORM | BufferUsages::COPY_DST),
        }
    }

    pub fn rope_shape(&self) -> &GpuScalar<RoPEShape> {
        &self.rope_shape
    }

    pub fn attn_params(&self) -> &GpuScalar<BatchedMultiqueryAttentionParams> {
        &self.attn_params
    }

    pub fn logits(&self) -> &GpuVector<f32> {
        &self.logits
    }

    pub fn logits_readback(&self) -> &GpuVector<f32> {
        &self.logits_readback
    }
}

pub struct Llama2LayerWeights {
    pub attn_k: GpuMatrix<f32>,
    pub attn_norm: GpuVector<f32>,
    pub attn_q: GpuMatrix<f32>,
    pub attn_v: GpuMatrix<f32>,
    pub ffn_down: GpuMatrix<f32>,
    pub ffn_gate: GpuMatrix<f32>,
    pub ffn_norm: GpuVector<f32>,
    pub ffn_up: GpuMatrix<f32>,
    pub attn_output: GpuMatrix<f32>,
}

pub struct Llama2Weights {
    pub layers: Vec<Llama2LayerWeights>,
    pub token_embd: GpuMatrix<f32>,
    pub output: GpuMatrix<f32>,
    pub output_norm: GpuVector<f32>,
}

impl Llama2Weights {
    pub fn from_ram(device: &Device, w: &TransformerWeights) -> Self {
        let usage = BufferUsages::STORAGE;

        let layers = w
            .layers
            .iter()
            .map(|l| Llama2LayerWeights {
                attn_k: GpuMatrix::init(device, &l.attn_k, usage),
                attn_norm: GpuVector::init(device, &l.attn_norm, usage),
                attn_q: GpuMatrix::init(device, &l.attn_q, usage),
                attn_v: GpuMatrix::init(device, &l.attn_v, usage),
                ffn_down: GpuMatrix::init(device, &l.ffn_down, usage),
                ffn_gate: GpuMatrix::init(device, &l.ffn_gate, usage),
                ffn_norm: GpuVector::init(device, &l.ffn_norm, usage),
                ffn_up: GpuMatrix::init(device, &l.ffn_up, usage),
                attn_output: GpuMatrix::init(device, &l.attn_output, usage),
            })
            .collect();

        let token_embd = GpuMatrix::init(device, &w.token_embd, usage | BufferUsages::COPY_SRC);
        let output = GpuMatrix::init(device, &w.output, usage);
        let output_norm = GpuVector::init(device, &w.output_norm, usage);

        Self {
            layers,
            token_embd,
            output,
            output_norm,
        }
    }
}

pub struct Llama2 {
    attn: BatchedMultiqueryAttention,
    rms_norm: RmsNorm,
    rope: RoPE,
    silu: Silu,
    matmul: Gemv,
    add_assign: OpAssign,
}

impl Llama2 {
    pub fn new(device: &Device) -> Result<Self, ComposerError> {
        Ok(Self {
            attn: BatchedMultiqueryAttention::from_device(device)?,
            rms_norm: RmsNorm::from_device(device)?,
            rope: RoPE::from_device(device)?,
            silu: Silu::from_device(device)?,
            matmul: Gemv::from_device(device)?,
            add_assign: OpAssign::new(device, OpAssignVariant::Add)?,
        })
    }

    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        state: &Llama2State,
        weights: &Llama2Weights,
        config: &Llama2Config,
        pos: u32,
    ) {
        for l in 0..config.n_layers {
            let wl = &weights.layers[l];
            self.rms_norm
                .queue(queue, &state.xb, &state.x, &wl.attn_norm);

            let k_cache = state.key_cache[l].column(pos);
            let v_cache = state.value_cache[l].column(pos);

            self.matmul.queue(queue, &state.q, &wl.attn_q, &state.xb);
            self.matmul.queue(queue, k_cache, &wl.attn_k, &state.xb);
            self.matmul.queue(queue, v_cache, &wl.attn_v, &state.xb);
            self.rope.queue(queue, &state.rope_shape, &state.q, k_cache);

            // Start attention.
            self.attn.queue(
                queue,
                &state.attn_params,
                &state.q,
                &state.key_cache[l],
                &state.value_cache[l],
                &state.att,
                &state.xb,
            );
            self.matmul
                .queue(queue, &state.xb2, &wl.attn_output, &state.xb);
            // End attention.

            self.add_assign.queue(queue, &state.x, &state.xb2);
            self.rms_norm
                .queue(queue, &state.xb, &state.x, &wl.ffn_norm);

            // Start ffn_silu
            self.matmul.queue(queue, &state.hb, &wl.ffn_gate, &state.xb);
            self.matmul.queue(queue, &state.hb2, &wl.ffn_up, &state.xb);
            self.silu.queue(queue, &state.hb, &state.hb2);
            self.matmul
                .queue(queue, &state.xb2, &wl.ffn_down, &state.hb);
            // End ffn_silu

            self.add_assign.queue(queue, &state.x, &state.xb2);
        }

        self.rms_norm
            .queue(queue, &state.xb, &state.x, &weights.output_norm);

        self.matmul
            .queue(queue, &state.logits, &weights.output, &state.xb);
    }
}
