// NOTE: inspired from https://raw.githubusercontent.com/rahoua/pecca-rs/main/src/main.rs

use std::fs::File;
use std::io::{self, Write};
use std::time::Instant;

use clap::Parser;
use nalgebra::DVector;
use wgcore::gpu::GpuInstance;
use wgcore::kernel::KernelInvocationQueue;
use wgml::gguf::Gguf;
use wgml::models::llama2::cpu::{Llama2Config, Transformer, TransformerWeights};
use wgml::models::llama2::{Llama2, Llama2State, Llama2Weights, Tokenizer};
use wgml::models::sampler::Sampler;
use wgml::ops::{BatchedMultiqueryAttentionParams, RoPEShape};

extern crate num_derive;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    model_file: String,

    // 0.0 = greedy deterministic. 1.0 = original.
    #[arg(short, long, default_value = "0.9")]
    temperature: f32,
    // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower.
    #[arg(short = 'p', long, default_value = "1.0")]
    topp: f32,

    #[arg(short, long, default_value = "256")]
    steps: usize,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    system_prompt: Option<String>,

    /// Overrides default tokenizer
    #[arg(short = 'k', long)]
    tokenizer: Option<String>,
}

#[async_std::main]
async fn main() {
    let gpu = GpuInstance::new().await.unwrap();
    let args = Args::parse();

    let file = File::open(args.model_file).expect("Unable to open the checkpoint file");

    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let gguf = Gguf::from_bytes(&mmap[..]).unwrap();
    gguf.print_metadata();
    gguf.print_tensors();

    let config = Llama2Config::from_gguf(&gguf);

    println!("Read config: {:#?}", config);

    // Finish reading the checkpoint file by loading weights
    let start = Instant::now();
    println!("Reading model weights, takes a little while...");
    let weights = TransformerWeights::from_gguf(&config, &gguf);
    println!("Done reading from disk.");

    let gpu_weights = Llama2Weights::from_ram(gpu.device(), &weights);
    println!("Done creating gpu buffers.");
    let gpu_transformer = Llama2::new(gpu.device()).unwrap();
    let gpu_state = Llama2State::new(gpu.device(), &config);

    println!(
        "Read model weights in {:.2}s.",
        start.elapsed().as_secs_f64()
    );

    let steps = args.steps.max(1).min(config.seq_len);
    let tok = Tokenizer::from_gguf(&gguf);
    let mut sampler = Sampler::new(config.vocab_size, args.temperature, args.topp);
    let mut trans = Transformer::new(config, weights);

    generate_gpu(
        &gpu,
        &gpu_transformer,
        &gpu_state,
        &gpu_weights,
        &config,
        &tok,
        &mut sampler,
        steps,
        args.prompt.clone(),
    )
    .await;
    generate(&mut trans, &tok, &mut sampler, steps, args.prompt.clone());
}

pub async fn generate_gpu<'a>(
    gpu: &GpuInstance,
    transformer: &Llama2,
    state: &Llama2State,
    weights: &Llama2Weights,
    config: &Llama2Config,
    tok: &Tokenizer,
    sampler: &mut Sampler,
    steps: usize,
    prompt: Option<String>,
) {
    let prompt = prompt.unwrap_or_else(|| "".to_string());
    let prompt_toks = tok.encode(&prompt, true, false);

    let start = Instant::now();
    let mut token = prompt_toks[0];
    println!("<s>");

    let mut transformer_time = 0.0;
    let mut queue_time = 0.0;

    for pos in 0..steps {
        let dim = config.dim;
        let kv_dim = ((config.dim * config.n_kv_heads) / config.n_q_heads) as u32;
        let head_size = (dim / config.n_q_heads) as u32;
        let kv_mul = (config.n_q_heads / config.n_kv_heads) as u32;

        let rope_shape = RoPEShape {
            head_size,
            kv_dim,
            pos: pos as u32,
        };

        let attn_params = BatchedMultiqueryAttentionParams {
            seq_len: config.seq_len as u32,
            kv_dim,
            kv_mul,
            n_heads: config.n_q_heads as u32,
            head_size,
            pos: pos as u32,
        };

        let t0 = Instant::now();
        let mut queue = KernelInvocationQueue::new(gpu.device());
        transformer.queue(&mut queue, state, weights, config, pos as u32);
        queue_time += t0.elapsed().as_secs_f64();

        // Run the transformer.
        let t0 = Instant::now();
        let mut logits = {
            let mut encoder = gpu.device().create_command_encoder(&Default::default());
            gpu.queue().write_buffer(
                state.rope_shape().buffer(),
                0,
                bytemuck::cast_slice(&[rope_shape]),
            );
            gpu.queue().write_buffer(
                state.attn_params().buffer(),
                0,
                bytemuck::cast_slice(&[attn_params]),
            );

            state
                .x
                .copy_from_view(&mut encoder, weights.token_embd.column(token as u32));
            queue.encode(&mut encoder, None);
            state
                .logits_readback()
                .copy_from(&mut encoder, state.logits());
            gpu.queue().submit(Some(encoder.finish()));

            // TODO: don’t allocate for the readback.
            let logits = DVector::from(state.logits_readback().read(gpu.device()).await.unwrap());
            transformer_time += t0.elapsed().as_secs_f64();
            logits
        };

        // Find the token and loop.
        let next = if pos < prompt_toks.len() - 1 {
            prompt_toks[pos + 1]
        } else {
            sampler.sample(&mut logits)
        };

        print!("{}", tok.decode(token, next));
        io::stdout().flush().unwrap();
        token = next;
    }
    println!(
        "\n[GPU] achieved tok/s: {}, transformer time: {}, queue_time: {}",
        steps as f64 / start.elapsed().as_secs_f64(),
        transformer_time / steps as f64,
        queue_time / steps as f64,
    );
}

pub fn generate(
    transformer: &mut Transformer,
    tok: &Tokenizer,
    sampler: &mut Sampler,
    steps: usize,
    prompt: Option<String>,
) {
    let prompt = prompt.unwrap_or_else(|| "".to_string());
    let prompt_toks = tok.encode(&prompt, true, false);

    let start = Instant::now();
    let mut token = prompt_toks[0];
    println!("<s>");

    let mut transformer_time = 0.0;

    for pos in 0..steps {
        let t0 = Instant::now();
        transformer.forward(token, pos);
        transformer_time += t0.elapsed().as_secs_f64();

        let next = if pos < prompt_toks.len() - 1 {
            prompt_toks[pos + 1]
        } else {
            sampler.sample(transformer.logits_mut())
        };

        print!("{}", tok.decode(token, next));
        io::stdout().flush().unwrap();
        token = next;
    }
    println!(
        "\n[CPU] achieved tok/s: {}, transformer time: {}",
        steps as f64 / start.elapsed().as_secs_f64(),
        transformer_time / steps as f64
    );
}
