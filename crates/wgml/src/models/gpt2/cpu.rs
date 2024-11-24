// Gpt-2 transformer, ported from ggml/examples/gpt-2/main-backend.cpp

use crate::gguf::Gguf;
use crate::ops::{
    BatchedMultiqueryAttention, BatchedMultiqueryAttentionParams, LayerNorm, UnaryOp,
};
use nalgebra::{DMatrix, DVector, Dyn, OMatrix, OVector, Vector4};

pub struct Transformer;

type VocabSize = Dyn;
type SeqLen = Dyn;
type NumHeads = Dyn;
type Dim = Dyn;
type Attn = Dyn; // 2304
type HiddenDim = Dyn; // 3072?

pub struct Gpt2Params {
    /// Size of the vocabulary.
    pub n_vocab: usize,
    /// Max sequence length.
    pub n_seq: usize,
    /// Token embedding length.
    pub n_embd: usize,
    /// Number of heads.
    pub n_head: usize,
    /// Number of layers.
    pub n_layer: usize,
    // Feed-forward length.
    pub ff_len: usize,
    pub attn_b: usize,
    pub ftype: usize,
}

impl Gpt2Params {
    pub fn from_gguf(gguf: &Gguf) -> Self {
        Self {
            n_vocab: gguf.metadata["tokenizer.ggml.tokens"].unwrap_array_len(),
            n_seq: gguf.metadata["gpt2.context_length"].unwrap_u32() as usize,
            n_embd: gguf.metadata["gpt2.embedding_length"].unwrap_u32() as usize,
            n_head: gguf.metadata["gpt2.attention.head_count"].unwrap_u32() as usize,
            n_layer: gguf.metadata["gpt2.block_count"].unwrap_u32() as usize,
            ftype: gguf.metadata["general.file_type"].unwrap_u32() as usize,
            ff_len: gguf.metadata["gpt2.feed_forward_length"].unwrap_u32() as usize,
            attn_b: gguf.tensors["blk.0.attn_qkv.bias"].dimensions()[0] as usize,
        }
    }
}

impl Default for Gpt2Params {
    fn default() -> Self {
        // Default params for GPT-2 117M
        Self {
            n_vocab: 50257,
            n_seq: 1024,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            attn_b: 2304,
            ftype: 1,
            ff_len: 3072,
        }
    }
}

pub struct Gpt2Layer {
    // Normalization.
    pub(crate) ln_1_g: OVector<f32, Dim>,
    pub(crate) ln_1_b: OVector<f32, Dim>,
    pub(crate) ln_2_g: OVector<f32, Dim>,
    pub(crate) ln_2_b: OVector<f32, Dim>,

    // attention
    pub(crate) c_attn_attn_w: OMatrix<f32, Attn, Dim>,
    pub(crate) c_attn_attn_b: OVector<f32, Attn>,
    pub(crate) c_attn_proj_w: OMatrix<f32, Dim, Attn>,
    pub(crate) c_attn_proj_b: OVector<f32, Dim>,

    // KV cache
    pub(crate) key_cache: OMatrix<f32, Dim, SeqLen>,
    pub(crate) value_cache: OMatrix<f32, Dim, SeqLen>,

    // mlp
    pub(crate) c_mlp_fc_w: OMatrix<f32, HiddenDim, Dim>,
    pub(crate) c_mlp_fc_b: OVector<f32, HiddenDim>,
    pub(crate) c_mlp_proj_w: OMatrix<f32, Dim, HiddenDim>,
    pub(crate) c_mlp_proj_b: OVector<f32, Dim>,
}

pub struct Gpt2Model {
    // Normalization
    pub(crate) ln_f_g: OVector<f32, Dim>,
    pub(crate) ln_f_b: OVector<f32, Dim>,

    pub(crate) wte: OMatrix<f32, Dim, VocabSize>, // token embedding
    pub(crate) wpe: OMatrix<f32, Dim, SeqLen>,    // position embedding
    pub(crate) lm_head: OMatrix<f32, VocabSize, Dim>, // language model head

    pub(crate) layers: Vec<Gpt2Layer>,

    // scratch memory
    memory_q: OVector<f32, Dim>,
    memory_att: OMatrix<f32, SeqLen, NumHeads>,
    layer_input: DVector<f32>,
    curr_768: DVector<f32>,
    curr_768_b: DVector<f32>,
    curr_2304: DVector<f32>,
    curr_3072: DVector<f32>,
    curr_vocab: DVector<f32>,
}

impl Gpt2Model {
    pub fn from_gguf(gguf: &Gguf) -> (Self, Gpt2Params) {
        let params = Gpt2Params::from_gguf(gguf);
        let mut layers = vec![];

        for i_layer in 0..params.n_layer {
            let ln_1_g = format!("blk.{}.attn_norm.weight", i_layer);
            let ln_1_b = format!("blk.{}.attn_norm.bias", i_layer);
            let ln_2_g = format!("blk.{}.ffn_norm.weight", i_layer);
            let ln_2_b = format!("blk.{}.ffn_norm.bias", i_layer);
            let c_attn_attn_w = format!("blk.{}.attn_qkv.weight", i_layer);
            let c_attn_attn_b = format!("blk.{}.attn_qkv.bias", i_layer);
            let c_attn_proj_w = format!("blk.{}.attn_output.weight", i_layer);
            let c_attn_proj_b = format!("blk.{}.attn_output.bias", i_layer);

            let c_mlp_fc_w = format!("blk.{}.ffn_up.weight", i_layer);
            let c_mlp_fc_b = format!("blk.{}.ffn_up.bias", i_layer);
            let c_mlp_proj_w = format!("blk.{}.ffn_down.weight", i_layer);
            let c_mlp_proj_b = format!("blk.{}.ffn_down.bias", i_layer);

            let ln_1_g = gguf.tensors[&ln_1_g].data().as_f32().unwrap();
            let ln_1_b = gguf.tensors[&ln_1_b].data().as_f32().unwrap();
            let ln_2_g = gguf.tensors[&ln_2_g].data().as_f32().unwrap();
            let ln_2_b = gguf.tensors[&ln_2_b].data().as_f32().unwrap();
            let c_attn_attn_w = &gguf.tensors[&c_attn_attn_w].data().dequantize().unwrap();
            let c_attn_attn_b = gguf.tensors[&c_attn_attn_b].data().as_f32().unwrap();
            let c_attn_proj_w = &gguf.tensors[&c_attn_proj_w].data().dequantize().unwrap();
            let c_attn_proj_b = gguf.tensors[&c_attn_proj_b].data().as_f32().unwrap();
            let c_mlp_fc_w = &gguf.tensors[&c_mlp_fc_w].data().dequantize().unwrap();
            let c_mlp_fc_b = gguf.tensors[&c_mlp_fc_b].data().as_f32().unwrap();
            let c_mlp_proj_w = &gguf.tensors[&c_mlp_proj_w].data().dequantize().unwrap();
            let c_mlp_proj_b = gguf.tensors[&c_mlp_proj_b].data().as_f32().unwrap();

            let ln_1_g = DVector::from_row_slice(ln_1_g);
            let ln_1_b = DVector::from_row_slice(ln_1_b);
            let ln_2_g = DVector::from_row_slice(ln_2_g);
            let ln_2_b = DVector::from_row_slice(ln_2_b);

            let c_attn_attn_w =
                DMatrix::from_row_slice(params.attn_b, params.n_embd, c_attn_attn_w);
            let c_attn_attn_b = DVector::from_row_slice(c_attn_attn_b);
            let c_attn_proj_w =
                DMatrix::from_row_slice(params.n_embd, params.n_embd, c_attn_proj_w);
            let c_attn_proj_b = DVector::from_row_slice(c_attn_proj_b);
            let c_mlp_fc_w = DMatrix::from_row_slice(params.ff_len, params.n_embd, c_mlp_fc_w);
            let c_mlp_fc_b = DVector::from_row_slice(c_mlp_fc_b);
            let c_mlp_proj_w = DMatrix::from_row_slice(params.n_embd, params.ff_len, c_mlp_proj_w);
            let c_mlp_proj_b = DVector::from_row_slice(c_mlp_proj_b);

            let layer = Gpt2Layer {
                ln_1_g,
                ln_1_b,
                ln_2_g,
                ln_2_b,
                c_attn_attn_w,
                c_attn_attn_b,
                c_attn_proj_w,
                c_attn_proj_b,
                c_mlp_fc_w,
                c_mlp_fc_b,
                c_mlp_proj_w,
                c_mlp_proj_b,
                key_cache: DMatrix::zeros(params.n_embd, params.n_seq),
                value_cache: DMatrix::zeros(params.n_embd, params.n_seq),
            };
            layers.push(layer);
        }

        let ln_f_g = gguf.tensors["output_norm.weight"].data().as_f32().unwrap();
        let ln_f_b = gguf.tensors["output_norm.bias"].data().as_f32().unwrap();
        let wte = gguf.tensors["token_embd.weight"]
            .data()
            .dequantize()
            .unwrap();
        let wpe = &gguf.tensors["position_embd.weight"]
            .data()
            .dequantize()
            .unwrap();

        let ln_f_g = DVector::from_row_slice(ln_f_g);
        let ln_f_b = DVector::from_row_slice(ln_f_b);
        let wte = DMatrix::from_column_slice(params.n_embd, params.n_vocab, &wte);
        let wpe = DMatrix::from_column_slice(params.n_embd, params.n_seq, wpe);
        // NOTE: GPT2 shares the lm_head tensor with wte.
        let lm_head = wte.transpose();

        let model = Self {
            ln_f_b,
            ln_f_g,
            wte,
            wpe,
            layers,
            lm_head,
            memory_q: DVector::zeros(params.n_embd),
            memory_att: DMatrix::zeros(params.n_seq, params.n_head),
            layer_input: DVector::zeros(params.n_embd),
            curr_768: DVector::zeros(params.n_embd),
            curr_768_b: DVector::zeros(params.n_embd),
            curr_2304: DVector::zeros(params.attn_b),
            curr_3072: DVector::zeros(params.ff_len),
            curr_vocab: DVector::zeros(params.n_vocab),
        };

        (model, params)
    }

    pub fn logits_mut(&mut self) -> &mut DVector<f32> {
        &mut self.curr_vocab
    }
}

impl Transformer {
    pub fn forward(params: &Gpt2Params, model: &mut Gpt2Model, embd: usize, pos: usize) {
        // Positional encoding.
        model.layer_input.copy_from(&model.wte.column(embd));
        model.layer_input += &model.wpe.column(pos);

        for layer in model.layers.iter_mut() {
            // Layer norm.
            {
                // NOTE: in this implementation, we always have N = 1
                // [ 768, N]
                LayerNorm::run_cpu(&mut model.curr_768, &model.layer_input);

                // cur = ln_1_g*cur + ln_1_b
                // [ 768, N]
                model.curr_768.component_mul_assign(&layer.ln_1_g);
                model.curr_768 += &layer.ln_1_b;
            }

            // attn
            // [2304, 768] - model.layers[il].c_attn_attn_w
            // [2304,   1] - model.layers[il].c_attn_attn_b
            // [ 768,   N] - cur (in)
            // [2304,   N] - cur (out)
            //
            // cur = attn_w*cur + attn_b
            // [2304, N]
            {
                model
                    .curr_2304
                    .gemv(1.0, &layer.c_attn_attn_w, &model.curr_768, 0.0);
                model.curr_2304 += &layer.c_attn_attn_b;
            }

            // self-attention
            // TODO: refactor this so that both llama2 and gpt2 share the attn code with KV cache.
            // TODO: implement flash attention.
            {
                // [2304, 1]
                let mut k_cache = layer.key_cache.column_mut(pos);
                let mut v_cache = layer.value_cache.column_mut(pos);

                model
                    .memory_q
                    .copy_from(&model.curr_2304.rows(0, params.n_embd));
                k_cache.copy_from(&model.curr_2304.rows(params.n_embd, params.n_embd));
                v_cache.copy_from(&model.curr_2304.rows(2 * params.n_embd, params.n_embd));

                let head_size = params.n_embd / params.n_head;
                let attn_params = BatchedMultiqueryAttentionParams {
                    seq_len: params.n_seq as u32,
                    kv_dim: params.n_embd as u32,
                    kv_mul: 1,
                    n_heads: params.n_head as u32,
                    head_size: head_size as u32,
                    pos: pos as u32,
                };

                BatchedMultiqueryAttention::run_cpu(
                    &attn_params,
                    &model.memory_q,
                    &layer.key_cache,
                    &layer.value_cache,
                    &mut model.memory_att,
                    &mut model.curr_768,
                );
            }

            // projection
            // [ 768, 768] - model.layers[il].c_attn_proj_w
            // [ 768,   1] - model.layers[il].c_attn_proj_b
            // [ 768,   N] - cur (in)
            // [ 768,   N] - cur (out)
            //
            // cur = proj_w*cur + proj_b
            // [768, N]
            {
                model
                    .curr_768_b
                    .gemv(1.0, &layer.c_attn_proj_w, &model.curr_768, 0.0);
                model.curr_768_b += &layer.c_attn_proj_b;
            }

            // add the input
            model.curr_768_b += &model.layer_input;

            // prep input for next layer
            model.layer_input.copy_from(&model.curr_768_b);

            // feed-forward network
            {
                // norm
                {
                    LayerNorm::run_cpu(&mut model.curr_768, &model.curr_768_b);

                    // cur = ln_2_g*cur + ln_2_b
                    // [ 768, N]
                    model.curr_768.component_mul_assign(&layer.ln_2_g);
                    model.curr_768 += &layer.ln_2_b;
                }

                // fully connected
                // [3072, 768] - model.layers[il].c_mlp_fc_w
                // [3072,   1] - model.layers[il].c_mlp_fc_b
                // [ 768,   N] - cur (in)
                // [3072,   N] - cur (out)
                //
                // cur = fc_w*cur + fc_b
                // [3072, N]
                model
                    .curr_3072
                    .gemv(1.0, &layer.c_mlp_fc_w, &model.curr_768, 0.0);
                model.curr_3072 += &layer.c_mlp_fc_b;

                // GELU activation
                // [3072, N]
                model
                    .curr_3072
                    .apply(|x| *x = UnaryOp::Gelu.eval(*x, Vector4::zeros()));

                // projection
                // [ 768, 3072] - model.layers[il].c_mlp_proj_w
                // [ 768,    1] - model.layers[il].c_mlp_proj_b
                // [3072,    N] - cur (in)
                // [ 768,    N] - cur (out)
                //
                // cur = proj_w*cur + proj_b
                // [768, N]
                model
                    .curr_768
                    .gemv(1.0, &layer.c_mlp_proj_w, &model.curr_3072, 0.0);
                model.curr_768 += &layer.c_mlp_proj_b;
            }

            // finalize input for next layer
            model.layer_input += &model.curr_768;
        }

        // norm
        {
            // [ 768, N]
            LayerNorm::run_cpu(&mut model.curr_768, &model.layer_input);

            // inpL = ln_f_g*inpL + ln_f_b
            // [ 768, N]
            model.curr_768.component_mul_assign(&model.ln_f_g);
            model.curr_768 += &model.ln_f_b;
        }

        // inpL = WTE * inpL
        // [ 768, 50257] - model.lm_head
        // [ 768, N]     - inpL
        model
            .curr_vocab
            .gemv(1.0, &model.lm_head, &model.curr_768, 0.0);

        // // logits -> probs
        // SoftMax::run_cpu(&mut curr2); // NOTE: done by the sampler
    }
}
