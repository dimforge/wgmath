use crate::models::gpt2::cpu::{Gpt2Model, Gpt2Params};
use crate::ops::{
    BatchedMultiqueryAttention, BatchedMultiqueryAttentionParams, LayerNorm, Unary, UnaryOp,
};
use naga_oil::compose::ComposerError;
use wgcore::kernel::KernelInvocationQueue;
use wgcore::tensor::{GpuMatrix, GpuScalar, GpuVector};
use wgcore::Shader;
use wgebra::linalg::{Gemv, OpAssign, OpAssignVariant};
use wgpu::{BufferUsages, Device};

pub struct Gpt2State {
    memory_q: GpuVector<f32>,
    memory_att: GpuMatrix<f32>,
    layer_input: GpuVector<f32>,
    curr_768: GpuVector<f32>,
    curr_768_b: GpuVector<f32>,
    curr_2304: GpuVector<f32>,
    curr_3072: GpuVector<f32>,
    curr_vocab: GpuVector<f32>,
    logits_readback: GpuVector<f32>,
    attn_params: GpuScalar<BatchedMultiqueryAttentionParams>,
}

impl Gpt2State {
    pub fn new(device: &Device, config: &Gpt2Params) -> Self {
        const STORAGE: BufferUsages = BufferUsages::STORAGE;
        const UNIFORM: BufferUsages = BufferUsages::UNIFORM;

        Self {
            memory_q: GpuVector::uninit(device, config.n_embd as u32, STORAGE),
            memory_att: GpuMatrix::uninit(
                device,
                config.n_seq as u32,
                config.n_head as u32,
                STORAGE,
            ),
            layer_input: GpuVector::uninit(device, config.n_embd as u32, STORAGE),
            curr_768: GpuVector::uninit(device, config.n_embd as u32, STORAGE),
            curr_768_b: GpuVector::uninit(device, config.n_embd as u32, STORAGE),
            curr_2304: GpuVector::uninit(device, config.attn_b as u32, STORAGE),
            curr_3072: GpuVector::uninit(device, config.ff_len as u32, STORAGE),
            curr_vocab: GpuVector::uninit(
                device,
                config.n_vocab as u32,
                STORAGE | BufferUsages::COPY_SRC,
            ),
            attn_params: GpuScalar::uninit(device, UNIFORM | BufferUsages::COPY_DST),
            logits_readback: GpuVector::uninit(
                device,
                config.n_vocab as u32,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            ),
        }
    }

    pub fn logits_readback(&self) -> &GpuVector<f32> {
        &self.logits_readback
    }

    pub fn logits(&self) -> &GpuVector<f32> {
        &self.curr_vocab
    }

    pub fn attn_params(&self) -> &GpuScalar<BatchedMultiqueryAttentionParams> {
        &self.attn_params
    }
}

pub struct Gpt2LayerWeights {
    // Normalization.
    ln_1_g: GpuVector<f32>,
    ln_1_b: GpuVector<f32>,
    ln_2_g: GpuVector<f32>,
    ln_2_b: GpuVector<f32>,

    // attention
    c_attn_attn_w: GpuMatrix<f32>,
    c_attn_attn_b: GpuVector<f32>,
    c_attn_proj_w: GpuMatrix<f32>,
    c_attn_proj_b: GpuVector<f32>,

    // KV cache
    key_cache: GpuMatrix<f32>,
    value_cache: GpuMatrix<f32>,

    // mlp
    c_mlp_fc_w: GpuMatrix<f32>,
    c_mlp_fc_b: GpuVector<f32>,
    c_mlp_proj_w: GpuMatrix<f32>,
    c_mlp_proj_b: GpuVector<f32>,
}

pub struct Gpt2Weights {
    // Normalization
    ln_f_g: GpuVector<f32>,
    ln_f_b: GpuVector<f32>,

    wte: GpuMatrix<f32>,     // token embedding
    wpe: GpuMatrix<f32>,     // position embedding
    lm_head: GpuMatrix<f32>, // language model head

    layers: Vec<Gpt2LayerWeights>,
}

impl Gpt2Weights {
    pub fn from_ram(device: &Device, w: &Gpt2Model) -> Self {
        const STORAGE: BufferUsages = BufferUsages::STORAGE;

        Self {
            ln_f_g: GpuVector::init(device, &w.ln_f_g, STORAGE),
            ln_f_b: GpuVector::init(device, &w.ln_f_b, STORAGE),
            wte: GpuMatrix::init(device, &w.wte, STORAGE),
            wpe: GpuMatrix::init(device, &w.wpe, STORAGE),
            lm_head: GpuMatrix::init(device, &w.lm_head, STORAGE),
            layers: w
                .layers
                .iter()
                .map(|l| Gpt2LayerWeights {
                    ln_1_g: GpuVector::init(device, &l.ln_1_g, STORAGE),
                    ln_1_b: GpuVector::init(device, &l.ln_1_b, STORAGE),
                    ln_2_g: GpuVector::init(device, &l.ln_2_g, STORAGE),
                    ln_2_b: GpuVector::init(device, &l.ln_2_b, STORAGE),
                    c_attn_attn_w: GpuMatrix::init(device, &l.c_attn_attn_w, STORAGE),
                    c_attn_attn_b: GpuVector::init(device, &l.c_attn_attn_b, STORAGE),
                    c_attn_proj_w: GpuMatrix::init(device, &l.c_attn_proj_w, STORAGE),
                    c_attn_proj_b: GpuVector::init(device, &l.c_attn_proj_b, STORAGE),
                    key_cache: GpuMatrix::init(device, &l.key_cache, STORAGE),
                    value_cache: GpuMatrix::init(device, &l.value_cache, STORAGE),
                    c_mlp_fc_w: GpuMatrix::init(device, &l.c_mlp_fc_w, STORAGE),
                    c_mlp_fc_b: GpuVector::init(device, &l.c_mlp_fc_b, STORAGE),
                    c_mlp_proj_w: GpuMatrix::init(device, &l.c_mlp_proj_w, STORAGE),
                    c_mlp_proj_b: GpuVector::init(device, &l.c_mlp_proj_b, STORAGE),
                })
                .collect(),
        }
    }
}

pub struct Gpt2 {
    layernorm: LayerNorm,
    gelu: Unary,
    matmul: Gemv,
    attn: BatchedMultiqueryAttention,
    copy_from: OpAssign,

    // TODO: merge the add/mul kernels with the layernorm kernel?
    mul_assign: OpAssign,
    add_assign: OpAssign,
}

impl Gpt2 {
    pub fn new(device: &Device) -> Result<Self, ComposerError> {
        Ok(Self {
            layernorm: LayerNorm::from_device(device)?,
            gelu: Unary::new(device, UnaryOp::Gelu)?,
            matmul: Gemv::from_device(device)?,
            attn: BatchedMultiqueryAttention::from_device(device)?,
            copy_from: OpAssign::new(device, OpAssignVariant::Copy)?,
            mul_assign: OpAssign::new(device, OpAssignVariant::Mul)?,
            add_assign: OpAssign::new(device, OpAssignVariant::Add)?,
        })
    }

    pub fn queue<'a>(
        &'a self,
        queue: &mut KernelInvocationQueue<'a>,
        state: &Gpt2State,
        weights: &Gpt2Weights,
        config: &Gpt2Params,
        embd: u32,
        pos: u32,
    ) {
        // Positional encoding.
        self.copy_from
            .queue(queue, &state.layer_input, weights.wte.column(embd));
        self.add_assign
            .queue(queue, &state.layer_input, weights.wpe.column(pos));

        for layer in &weights.layers {
            // Layer norm.
            {
                self.layernorm
                    .queue(queue, &state.curr_768, &state.layer_input);

                // cur = ln_1_g*cur + ln_1_b
                self.mul_assign.queue(queue, &state.curr_768, &layer.ln_1_g);
                self.add_assign.queue(queue, &state.curr_768, &layer.ln_1_b);
            }

            // attn
            {
                self.matmul.queue(
                    queue,
                    &state.curr_2304,
                    &layer.c_attn_attn_w,
                    &state.curr_768,
                );
                self.add_assign
                    .queue(queue, &state.curr_2304, &layer.c_attn_attn_b);
            }

            // self-attention
            {
                let k_cache = layer.key_cache.column(pos);
                let v_cache = layer.value_cache.column(pos);

                self.copy_from.queue(
                    queue,
                    &state.memory_q,
                    state.curr_2304.rows(0, config.n_embd as u32),
                );
                self.copy_from.queue(
                    queue,
                    k_cache,
                    state
                        .curr_2304
                        .rows(config.n_embd as u32, config.n_embd as u32),
                );
                self.copy_from.queue(
                    queue,
                    v_cache,
                    state
                        .curr_2304
                        .rows(2 * config.n_embd as u32, config.n_embd as u32),
                );

                // attention.
                self.attn.queue(
                    queue,
                    &state.attn_params,
                    &state.memory_q,
                    &layer.key_cache,
                    &layer.value_cache,
                    &state.memory_att,
                    &state.curr_768,
                );
            }

            // projection
            // cur = proj_w*cur + proj_b
            {
                self.matmul.queue(
                    queue,
                    &state.curr_768_b,
                    &layer.c_attn_proj_w,
                    &state.curr_768,
                );
                self.add_assign
                    .queue(queue, &state.curr_768_b, &layer.c_attn_proj_b);
            }

            // add the input
            self.add_assign
                .queue(queue, &state.curr_768_b, &state.layer_input);

            // prep input for next layer
            self.copy_from
                .queue(queue, &state.layer_input, &state.curr_768_b);

            // feed-forward network
            {
                // norm
                {
                    self.layernorm
                        .queue(queue, &state.curr_768, &state.curr_768_b);

                    // cur = ln_2_g*cur + ln_2_b
                    self.mul_assign.queue(queue, &state.curr_768, &layer.ln_2_g);
                    self.add_assign.queue(queue, &state.curr_768, &layer.ln_2_b);
                }

                // fully connected
                self.matmul
                    .queue(queue, &state.curr_3072, &layer.c_mlp_fc_w, &state.curr_768);
                self.add_assign
                    .queue(queue, &state.curr_3072, &layer.c_mlp_fc_b);

                // GELU activation
                self.gelu
                    .queue(queue, &state.curr_3072, &state.curr_3072, None);

                // projection
                self.matmul.queue(
                    queue,
                    &state.curr_768,
                    &layer.c_mlp_proj_w,
                    &state.curr_3072,
                );
                self.add_assign
                    .queue(queue, &state.curr_768, &layer.c_mlp_proj_b);
            }

            // finalize input for next layer
            self.add_assign
                .queue(queue, &state.layer_input, &state.curr_768);
        }

        // norm
        {
            self.layernorm
                .queue(queue, &state.curr_768, &state.layer_input);

            // inpL = ln_f_g*inpL + ln_f_b
            self.mul_assign
                .queue(queue, &state.curr_768, &weights.ln_f_g);
            self.add_assign
                .queue(queue, &state.curr_768, &weights.ln_f_b);
        }

        // inpL = WTE * inpL
        self.matmul
            .queue(queue, &state.curr_vocab, &weights.lm_head, &state.curr_768);
    }
}
