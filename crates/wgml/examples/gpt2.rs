// NOTE: inspired from https://raw.githubusercontent.com/rahoua/pecca-rs/main/src/main.rs

use std::fs::File;
use std::io::{self, Write};
use std::time::Instant;

use clap::Parser;
use nalgebra::DVector;
use wgcore::gpu::GpuInstance;
use wgcore::kernel::KernelInvocationQueue;
use wgcore::timestamps::GpuTimestamps;
use wgml::gguf::{Gguf, GgufMetadataValue};
use wgml::models::gpt2::cpu::{Gpt2Model, Gpt2Params, Transformer};
use wgml::models::gpt2::Tokenizer;
use wgml::models::sampler::Sampler;
use wgml::ops::BatchedMultiqueryAttentionParams;

use wgml::models::gpt2::transformer::{Gpt2, Gpt2State, Gpt2Weights};

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

    // Initialize config from file
    let start = Instant::now();

    let file = File::open(args.model_file).expect("Unable to open the checkpoint file");
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let gguf = Gguf::from_bytes(&mmap[..]).unwrap();
    gguf.print_metadata();
    gguf.print_tensors();

    let (mut gpt_data, config) = Gpt2Model::from_gguf(&gguf);
    println!(
        "Read model weights in {:.2}s.",
        start.elapsed().as_secs_f64()
    );

    let steps = args.steps.max(1).min(config.n_seq);
    let tok = Tokenizer::from_gguf(&gguf);
    let mut sampler = Sampler::new(config.n_vocab as usize, args.temperature, args.topp);

    println!("Creating gpu buffers.");
    let t_gpu_buf = Instant::now();
    let gpu_weights = Gpt2Weights::from_ram(gpu.device(), &gpt_data);
    let gpu_transformer = Gpt2::new(gpu.device()).unwrap();
    let gpu_state = Gpt2State::new(gpu.device(), &config);
    println!(
        "Done creating gpu buffers: {}",
        t_gpu_buf.elapsed().as_secs_f64()
    );

    forward_gpu(
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
    forward_cpu(
        &config,
        &mut gpt_data,
        &tok,
        &mut sampler,
        steps,
        args.prompt.clone(),
    );
}

pub fn forward_cpu(
    config: &Gpt2Params,
    model: &mut Gpt2Model,
    tok: &Tokenizer,
    sampler: &mut Sampler,
    steps: usize,
    prompt: Option<String>,
) {
    let prompt = prompt.unwrap_or_else(|| "".to_string());
    let prompt_toks = tok.encode(&prompt);

    println!("Prompt tokens: {:?}", prompt_toks);

    let start = Instant::now();
    let mut token = prompt_toks[0];
    println!("<s>");

    print!("{}", tok.decode(&[prompt_toks[0]]));

    let mut transformer_time = 0.0;

    for pos in 0..steps {
        let t0 = Instant::now();
        Transformer::forward(&config, model, token as usize, pos);
        transformer_time += t0.elapsed().as_secs_f64();

        let next = if pos < prompt_toks.len() - 1 {
            prompt_toks[pos + 1]
        } else {
            sampler.sample(model.logits_mut()) as u32
        };

        print!("{}", tok.decode(&[next]));
        io::stdout().flush().unwrap();
        token = next;
    }
    println!(
        "\n[CPU] achieved tok/s: {}, transformer time: {}",
        steps as f64 / start.elapsed().as_secs_f64(),
        transformer_time / steps as f64
    );
}

pub async fn forward_gpu<'a>(
    gpu: &GpuInstance,
    transformer: &Gpt2,
    state: &Gpt2State,
    weights: &Gpt2Weights,
    config: &Gpt2Params,
    tok: &Tokenizer,
    sampler: &mut Sampler,
    steps: usize,
    prompt: Option<String>,
) {
    let prompt = prompt.unwrap_or_else(|| "".to_string());
    let prompt_toks = tok.encode(&prompt);

    let start = Instant::now();
    let mut token = prompt_toks[0];
    println!("<s>");

    let mut transformer_time = 0.0;
    let mut kernels_time = 0.0;
    let mut queue_time = 0.0;

    let mut timestamps = GpuTimestamps::new(gpu.device(), 2);

    for pos in 0..steps {
        let head_size = config.n_embd / config.n_head;
        let attn_params = BatchedMultiqueryAttentionParams {
            seq_len: config.n_seq as u32,
            kv_dim: config.n_embd as u32,
            kv_mul: 1,
            n_heads: config.n_head as u32,
            head_size: head_size as u32,
            pos: pos as u32,
        };

        timestamps.clear();
        let t0 = Instant::now();
        let mut queue = KernelInvocationQueue::new(gpu.device());
        queue.compute_pass("main_pass", true);
        transformer.queue(&mut queue, state, weights, config, token, pos as u32);
        queue_time += t0.elapsed().as_secs_f64();

        // Run the transformer.
        let t0 = Instant::now();
        let mut logits = {
            let mut encoder = gpu.device().create_command_encoder(&Default::default());
            gpu.queue().write_buffer(
                state.attn_params().buffer(),
                0,
                bytemuck::cast_slice(&[attn_params]),
            );

            queue.encode(&mut encoder, Some(&mut timestamps));
            timestamps.resolve(&mut encoder);
            state
                .logits_readback()
                .copy_from(&mut encoder, state.logits());
            gpu.queue().submit(Some(encoder.finish()));

            // TODO: don’t allocate for the readback.
            let logits = DVector::from(state.logits_readback().read(gpu.device()).await.unwrap());
            transformer_time += t0.elapsed().as_secs_f64();
            logits
        };

        let ts = timestamps.wait_for_results_ms(gpu.device(), gpu.queue());
        kernels_time += ts[1] - ts[0];

        // Find the token and loop.
        let next = if pos < prompt_toks.len() - 1 {
            prompt_toks[pos + 1]
        } else {
            sampler.sample(&mut logits) as u32
        };

        print!("{}", tok.decode(&[next]));
        io::stdout().flush().unwrap();
        token = next;
    }
    println!(
        "\n[GPU] achieved tok/s: {}, transformer time: {}, kernels time: {}, queue_time: {}",
        steps as f64 / start.elapsed().as_secs_f64(),
        transformer_time / steps as f64,
        kernels_time / steps as f64,
        queue_time / steps as f64,
    );
}
