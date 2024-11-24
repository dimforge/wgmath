//! The CPU version of the llama2 transformer.

use crate::gguf::Gguf;
use nalgebra::{
    vector, DMatrix, DVector, DVectorViewMut, Dyn, OMatrix, OVector, Rotation2, Storage,
    StorageMut, Vector,
};
use std::ffi::c_int;

type Dim = Dyn;
type HiddenDim = Dyn;
type NumHeads = Dyn;
type SeqLen = Dyn;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RawConfig {
    /// The transformer dimension.
    /// In particular, this is the size of an embedding.
    dim: c_int,
    /// Number of dimension of the feed-forward neural net.
    hidden_dim: c_int,
    /// Number of layers.
    n_layers: c_int,
    /// Number of query heads.
    n_q_heads: c_int,
    /// Number of key/value heads (can be < than `n_q_heads` because of multiquery).
    /// See <https://youtu.be/Mn_9W1nCFLo?si=UnkLuzaHlX8JKyjl&t=3808> (Grouped-query diagram).
    n_kv_heads: c_int,
    /// Vocabulary size, usually 256 (byte -level).
    vocab_size: c_int,
    /// Max sequence length.
    seq_len: c_int,
}

/*
 * Important note: the original code (like most of the LLM literature) assumes row-major matrices
 * with left-multiplication (vector * Matrix).
 * nalgebra uses column-major with right-multiplication (Matrix * vector). So in the end the data layout still match,
 * we just have to swap al the matrix dimensions, access columns instead of rows (and vice versa),
 * and replace left-multiplication by right-multiplication.
 */
#[derive(Copy, Clone, Debug)]
pub struct Llama2Config {
    /// The transformer dimension.
    /// In particular, this is the size of an embedding.
    pub dim: usize,
    /// Number of dimension of the feed-forward neural net.
    pub hidden_dim: usize,
    /// Number of layers.
    pub n_layers: usize,
    /// Number of query heads.
    pub n_q_heads: usize,
    /// Number of key/value heads (can be < than `n_q_heads` because of multiquery).
    /// See <https://youtu.be/Mn_9W1nCFLo?si=UnkLuzaHlX8JKyjl&t=3808> (Grouped-query diagram).
    pub n_kv_heads: usize,
    /// Vocabulary size, usually 256 (byte -level).
    pub vocab_size: usize,
    /// Max sequence length.
    pub seq_len: usize,
    pub shared_weights: bool,
}

impl Llama2Config {
    pub fn read(bytes: &[u8]) -> Self {
        let elts: &[RawConfig] = bytemuck::cast_slice(&bytes[..std::mem::size_of::<RawConfig>()]);
        elts[0].into()
    }

    pub fn from_gguf(gguf: &Gguf) -> Self {
        Self {
            dim: gguf.metadata["llama.embedding_length"].unwrap_u32() as usize,
            hidden_dim: gguf.metadata["llama.feed_forward_length"].unwrap_u32() as usize,
            n_layers: gguf.metadata["llama.block_count"].unwrap_u32() as usize,
            n_q_heads: gguf.metadata["llama.attention.head_count"].unwrap_u32() as usize,
            n_kv_heads: gguf.metadata["llama.attention.head_count_kv"].unwrap_u32() as usize,
            vocab_size: gguf.metadata["tokenizer.ggml.tokens"].unwrap_array_len(),
            seq_len: gguf.metadata["llama.context_length"].unwrap_u32() as usize,
            shared_weights: true, // ???
        }
    }
}

impl From<RawConfig> for Llama2Config {
    fn from(c: RawConfig) -> Self {
        Self {
            dim: c.dim as usize,
            hidden_dim: c.hidden_dim as usize,
            n_layers: c.n_layers as usize,
            n_q_heads: c.n_q_heads as usize,
            n_kv_heads: c.n_kv_heads as usize,
            vocab_size: c.vocab_size.unsigned_abs() as usize,
            seq_len: c.seq_len as usize,
            shared_weights: c.vocab_size > 0,
        }
    }
}

pub struct TransformerLayerWeights {
    pub attn_k: DMatrix<f32>,
    pub attn_norm: DVector<f32>,
    pub attn_q: DMatrix<f32>,
    pub attn_v: DMatrix<f32>,
    pub ffn_down: DMatrix<f32>,
    pub ffn_gate: DMatrix<f32>,
    pub ffn_norm: DVector<f32>,
    pub ffn_up: DMatrix<f32>,
    pub attn_output: DMatrix<f32>,
}

pub struct TransformerWeights {
    pub layers: Vec<TransformerLayerWeights>,
    pub token_embd: DMatrix<f32>,
    pub output: DMatrix<f32>,
    pub output_norm: DVector<f32>,
}

impl TransformerWeights {
    pub fn from_gguf(config: &Llama2Config, gguf: &Gguf) -> Self {
        let head_size = config.dim / config.n_q_heads;
        let num_kv_heads_times_head_size = config.n_kv_heads * head_size;

        let mut layers = vec![];

        for i_layer in 0..config.n_layers {
            let attn_q = format!("blk.{}.attn_q.weight", i_layer);
            let attn_k = format!("blk.{}.attn_k.weight", i_layer);
            let attn_v = format!("blk.{}.attn_v.weight", i_layer);
            let attn_output = format!("blk.{}.attn_output.weight", i_layer);
            let ffn_down = format!("blk.{}.ffn_down.weight", i_layer);
            let ffn_gate = format!("blk.{}.ffn_gate.weight", i_layer);
            let ffn_up = format!("blk.{}.ffn_up.weight", i_layer);
            let ffn_norm = format!("blk.{}.ffn_norm.weight", i_layer);
            let attn_norm = format!("blk.{}.attn_norm.weight", i_layer);

            let attn_q = &gguf.tensors[&attn_q].data().dequantize().unwrap();
            let attn_k = &gguf.tensors[&attn_k].data().dequantize().unwrap();
            let attn_v = &gguf.tensors[&attn_v].data().dequantize().unwrap();
            let attn_output = &gguf.tensors[&attn_output].data().dequantize().unwrap();
            let ffn_down = &gguf.tensors[&ffn_down].data().dequantize().unwrap();
            let ffn_gate = &gguf.tensors[&ffn_gate].data().dequantize().unwrap();
            let ffn_up = &gguf.tensors[&ffn_up].data().dequantize().unwrap();
            let ffn_norm = gguf.tensors[&ffn_norm].data().as_f32().unwrap();
            let attn_norm = gguf.tensors[&attn_norm].data().as_f32().unwrap();

            let ffn_norm = DVector::from_row_slice(ffn_norm);
            let attn_norm = DVector::from_row_slice(attn_norm);

            let attn_q = DMatrix::from_row_slice(config.dim, config.dim, attn_q);
            let attn_k = DMatrix::from_row_slice(num_kv_heads_times_head_size, config.dim, attn_k);
            let attn_v = DMatrix::from_row_slice(num_kv_heads_times_head_size, config.dim, attn_v);
            let attn_output = DMatrix::from_row_slice(config.dim, config.dim, attn_output);
            let ffn_down = DMatrix::from_row_slice(config.dim, config.hidden_dim, ffn_down);
            let ffn_gate = DMatrix::from_row_slice(config.hidden_dim, config.dim, ffn_gate);
            let ffn_up = DMatrix::from_row_slice(config.hidden_dim, config.dim, ffn_up);

            layers.push(TransformerLayerWeights {
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                ffn_down,
                ffn_gate,
                ffn_up,
                ffn_norm,
                attn_norm,
            });
        }

        let token_embd = "token_embd.weight";
        let output = "output.weight";
        let output_norm = "output_norm.weight";

        let token_embd = &gguf.tensors[token_embd].data().dequantize().unwrap();
        let output = gguf
            .tensors
            .get(output)
            .map(|v| v.data().dequantize().unwrap());
        let output_norm = gguf.tensors[output_norm].data().as_f32().unwrap();

        let token_embd = DMatrix::from_column_slice(config.dim, config.vocab_size, token_embd);
        let output = output
            .map(|data| DMatrix::from_row_slice(config.vocab_size, config.dim, &data))
            .unwrap_or_else(|| token_embd.transpose());
        let output_norm = DVector::from_row_slice(output_norm);

        Self {
            layers,
            token_embd,
            output,
            output_norm,
        }
    }
}

struct RunState {
    // Current wave of activations.
    /// Activation at current time stamp.
    x: OVector<f32, Dim>,
    /// Activation at current time stamp, inside a residual branch.
    xb: OVector<f32, Dim>,
    /// Additional buffer for convenience.
    xb2: OVector<f32, Dim>,
    /// Buffer for hidden dimension in the Feed-Forward net.
    hb: OVector<f32, HiddenDim>,
    /// Another buffer for hidden dimension in the Feed-Forward net.
    hb2: OVector<f32, HiddenDim>,
    /// Query.
    q: OVector<f32, Dim>,
    /// Scores/attention values.
    att: OMatrix<f32, SeqLen, NumHeads>,
    /// Output logits.
    logits: OVector<f32, SeqLen>,
    // KV cache. Each Vec contains `layer` elements.
    key_cache: Vec<OMatrix<f32, Dim, SeqLen>>,
    value_cache: Vec<OMatrix<f32, Dim, SeqLen>>,
}

pub struct Transformer {
    /// The hyperparameters of the architecture (the blueprint).
    config: Llama2Config,
    /// The weights of the model.
    weights: TransformerWeights,
    /// Buffer of the "wave" of activations in the forward pass.
    state: RunState,
}

impl Transformer {
    pub fn new(config: Llama2Config, weights: TransformerWeights) -> Self {
        Self {
            state: RunState::new(&config),
            config,
            weights,
        }
    }

    pub fn logits_mut(&mut self) -> &mut OVector<f32, SeqLen> {
        &mut self.state.logits
    }
}

impl RunState {
    pub fn new(config: &Llama2Config) -> Self {
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_q_heads;
        Self {
            x: DVector::zeros(config.dim),
            xb: DVector::zeros(config.dim),
            xb2: DVector::zeros(config.dim),
            hb: DVector::zeros(config.hidden_dim),
            hb2: DVector::zeros(config.hidden_dim),
            q: DVector::zeros(config.dim),
            // TODO: for these two, the `kv_dim` doesn’t match the dimension in the field’s comment.
            key_cache: (0..config.n_layers)
                .map(|_| DMatrix::zeros(kv_dim, config.seq_len))
                .collect(),
            value_cache: (0..config.n_layers)
                .map(|_| DMatrix::zeros(kv_dim, config.seq_len))
                .collect(),
            att: DMatrix::zeros(config.seq_len, config.n_q_heads),
            logits: DVector::zeros(config.vocab_size),
        }
    }
}

/*
 *
 *
 * Neural net blocks. The dynamics of the Transformer.
 *
 *
 */
/// Implementation of the Root Mean Square Normalization.
///
/// This implementation of the RMS normalization from the "Root Mean Square
/// Normalization" paper by Zhang & Sennrich.
fn rms_norm<SW: Storage<f32, Dyn>>(
    out: &mut DVector<f32>,
    a: &DVector<f32>,
    w: &Vector<f32, Dyn, SW>,
) {
    const NUDGE_FACTOR: f32 = 1.0e-5;
    let rms = 1.0 / (a.norm_squared() / (a.nrows() as f32) + NUDGE_FACTOR).sqrt();
    out.zip_zip_apply(a, w, |o, a, w| *o = (a * rms) * w);
}

/// The softmax function.
///
/// Converts a set of real number into a probability distribution.
/// See <https://fr.wikipedia.org/wiki/Fonction_softmax>
pub fn softmax<S: StorageMut<f32, Dyn>>(vals: &mut Vector<f32, Dyn, S>) {
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

/// Most expensive part of the inference.
// TODO llama2.c also takes the dimensions n and do, but it’s unclear if this isn’t just
//      because the dimensions are not part of the float* input type.
fn matmul<SOut: StorageMut<f32, Dyn>>(
    out: &mut Vector<f32, Dyn, SOut>,
    x: &DVector<f32>,
    w: &DMatrix<f32>,
) {
    // TODO: parallelize per column? llama2.c paralelizes with openmp.
    // TODO: use blast/faer?
    out.gemv(1.0, w, x, 0.0);
}

impl Transformer {
    pub fn forward(&mut self, token: usize, pos: usize) {
        // A few convenience variables.
        let config = &self.config;
        let w = &self.weights;
        let s = &mut self.state;
        let dim = config.dim;
        // This is the number of key/value heads multiplied by the size of a query head: NumKvHeadsTimesHeadSize
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_q_heads;
        // The number of embedding vector elements associated to each query head.
        let head_size = dim / config.n_q_heads;

        // Copy the token embedding into x.
        // TODO: rename `x` to `token_embedding`?
        s.x.copy_from(&w.token_embd.column(token));

        // Forward all the layers.
        for l in 0..config.n_layers {
            let wl = &w.layers[l];

            // RMS norm before attention.
            // See https://youtu.be/Mn_9W1nCFLo?si=Ogz_O_6LUsumWovB&t=1367
            // TODO: rename `xb` to `normalized_token_embedding`?
            rms_norm(&mut s.xb, &s.x, &wl.attn_norm);

            // Key and value point to the KV cache.
            let mut k_cache = s.key_cache[l].column_mut(pos);
            let mut v_cache = s.value_cache[l].column_mut(pos);

            // qkv matmuls for this position.
            // This is self-attention, so `xb` is used for query, key, and value.
            // These are essentially one row of Q’, K’, V’ from https://youtu.be/Mn_9W1nCFLo?si=7B_g41B2iGZ5238a&t=2422
            // Note that despite keys/values having different number of heads as queries, the dimension of
            // each k/v head are the same as the query heads. The dimension change happens through the
            // multiplication by the weight matrices wk/wv.
            matmul(&mut s.q, &s.xb, &wl.attn_q);
            matmul(&mut k_cache, &s.xb, &wl.attn_k);
            matmul(&mut v_cache, &s.xb, &wl.attn_v);

            // Rotary Positional Encoding (RoPE).
            Self::rotary_positional_encoding(&mut s.q, &mut k_cache, head_size, dim, kv_dim, pos);

            // Batched multi-query attention.
            Self::attention(config, s, w, pos, l);

            // Residual connection back into x.
            // See the LLama graph on the right: https://youtu.be/Mn_9W1nCFLo?si=XMDdHlXxON2QhFCd&t=320
            // This step is the first big circled +
            s.x += &s.xb2;

            // RMSnorm before feed-forward.
            // /!\ xb changes semantic again. It now contains the normalized {attention output+input}.
            rms_norm(&mut s.xb, &s.x, &wl.ffn_norm);

            // Feed-forward.
            Self::ffn_silu(s, wl);

            // Residual connection.
            s.x += &s.xb2;
            // Loop on the next layer. This layer’s output is the next layer’s input.
        }

        // Final rmsnorm.
        // This is the top-most rmsnorm from https://youtu.be/Mn_9W1nCFLo?si=KO-aBXZo0DqCL4Qs&t=275
        // (diagram on the right).
        rms_norm(&mut s.xb, &s.x, &w.output_norm);

        // Classifier into logits.
        // This is the final "Linear" part from https://youtu.be/Mn_9W1nCFLo?si=-GT74rBY6j5TbbBO&t=275
        matmul(&mut s.logits, &s.xb, &w.output);
    }

    // Rotary Positional Encoding (RoPE): complex-valued rotate q and k in each head.
    pub fn rotary_positional_encoding(
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

    fn attention(
        config: &Llama2Config,
        s: &mut RunState,
        w: &TransformerWeights,
        pos: usize,
        l: usize,
    ) {
        // The number of embedding vector elements associated to each query head.
        let head_size = config.dim / config.n_q_heads;
        // The number of query head associated to one key/value head.
        let kv_mul = config.n_q_heads / config.n_kv_heads;

        // Multihead attention. Iterate over all head.
        // TODO: in llama2.c, each head is iterated on in parallel.
        for h in 0..config.n_q_heads {
            // Get the query vector for this head.
            let q = s.q.rows(h * head_size, head_size);
            // Attention scores for this head.
            let mut att = s.att.column_mut(h);

            // Iterate over all timesteps (tokens in the sequence), including the current one, but
            // not past the current one due to causality.
            // See the KV cache explanation there: https://youtu.be/Mn_9W1nCFLo?si=3n4GH9f2OzMb5Np0&t=2940
            // -> This is iterating through all the green columns (from K^t) that are the rotated
            //    (by RoPE). The values set in this loop into the `att` variable here (attention
            //    scores) are the elements in the pink row (at the bottom of the QK^t matrix) divide
            //    by sqrt(head_size) (in other words, this is what’s given to softmax afterward.
            for t in 0..=pos {
                // Get the key vector for this head and at this timestep.
                let k = s.key_cache[l].column(t);
                let k_head = k.rows((h / kv_mul) * head_size, head_size);

                // Calculate the attention score as the dot product of q and k.
                let mut score = q.dot(&k_head);
                score /= (head_size as f32).sqrt();
                // Save the score to the attention buffer.
                att[t] = score;
            }

            // Softmax the scores to get attention weights from 0..=pos inclusively.
            softmax(&mut att.rows_mut(0, pos + 1));

            // Weighted sum of the values, store back into xb.
            // /!\ xb is now changing semantic, storing the weighted sums for all the heads.
            //       Now xb contains the "Attention 4" row from https://youtu.be/Mn_9W1nCFLo?si=550ar5aUg1I1k60l&t=2940.
            let mut xb = s.xb.rows_mut(h * head_size, head_size);
            xb.fill(0.0);
            for t in 0..=pos {
                let v = s.value_cache[l].column(t);
                let v_head = v.rows((h / kv_mul) * head_size, head_size);
                xb.axpy(att[t], &v_head, 1.0);
            }
        }

        // Final matmul to get the output of the attention.
        // TODO: rename xb2 to `attention_output`?
        matmul(&mut s.xb2, &s.xb, &w.layers[l].attn_output);
    }

    fn ffn_silu(s: &mut RunState, wl: &TransformerLayerWeights) {
        // We have: self.w2(F.silu(self.w1(x)) * self.w3(x)) first calculate self.w1(x) and
        // self.w3(x)
        //
        // For this part, see https://youtu.be/Mn_9W1nCFLo?si=Ub9m1NeAzkmn-G8G&t=3973
        // We have: w1 := W, w3 := V, w2 := W2
        s.hb.gemv(1.0, &wl.ffn_gate, &s.xb, 0.0);
        s.hb2.gemv(1.0, &wl.ffn_up, &s.xb, 0.0);

        // SwiGLU non-linearity.
        fn swish(x: f32, beta: f32) -> f32 {
            // This is the swish function from https://youtu.be/Mn_9W1nCFLo?si=LT6puSAfzgpP6ydz&t=3973
            x / (1.0 + (-beta * x).exp())
        }

        s.hb.zip_apply(&s.hb2, |h, h2| *h = h2 * swish(*h, 1.0));

        // Final matmul to get the output of the feed-forward net.
        matmul(&mut s.xb2, &s.hb, &wl.ffn_down);
    }
}
