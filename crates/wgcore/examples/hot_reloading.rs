use nalgebra::{DVector, Vector4};
use wgcore::composer::ComposerExt;
use wgcore::gpu::GpuInstance;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::{GpuScalar, GpuVector};
use wgcore::Shader;
use wgpu::{BufferUsages, ComputePipeline};
use wgcore::hot_reloading::HotReloadState;

#[derive(Shader)]
#[shader(
    src = "hot_reloading.wgsl",
    composable = false
)]
struct ShaderHotReloading {
    main: ComputePipeline
}

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    // Initialize the gpu device and its queue.
    //
    // Note that `GpuInstance` is just a simple helper struct for initializing the gpu resources.
    // You are free to initialize them independently if more control is needed, or reuse the ones
    // that were already created/owned by e.g., a game engine.
    let gpu = GpuInstance::new().await?;

    // Load and compile our kernel. The `from_device` function was generated by the `Shader` derive.
    // Note that its dependency to `Composable` is automatically resolved by the `Shader` derive
    // too.
    let mut kernel = ShaderHotReloading::from_device(gpu.device())?;

    // Create the buffers.
    let buffer = GpuScalar::init(gpu.device(), 0u32, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
    let staging = GpuScalar::init(gpu.device(), 0u32, BufferUsages::COPY_DST | BufferUsages::MAP_READ);

    // Init hot-reloading.
    let mut hot_reload = HotReloadState::new()?;
    ShaderHotReloading::watch_sources(&mut hot_reload)?;

    // Queue the operation.
    println!("#############################");
    println!("Edit the file `hot_reloading.wgsl`.\nThe updated result will be printed below whenever a change is detected.");
    println!("#############################");

    for loop_id in 0.. {
        // Detect & apply changes.
        hot_reload.update_changes();
        match kernel.reload_if_changed(gpu.device(), &hot_reload) {
            Ok(changed) => {
                if changed || loop_id == 0 {
                    // We detected a change (or this is the first loop).
                    // Read the result.
                    let mut queue = KernelInvocationQueue::new(gpu.device());
                    KernelInvocationBuilder::new(&mut queue, &kernel.main)
                        .bind0([buffer.buffer()])
                        .queue(1);

                    // Encode & submit the operation to the gpu.
                    let mut encoder = gpu.device().create_command_encoder(&Default::default());
                    // Run our kernel.
                    queue.encode(&mut encoder, None);
                    // Copy the result to the staging buffer.
                    staging.copy_from(&mut encoder, &buffer);
                    gpu.queue().submit(Some(encoder.finish()));

                    let result_read = staging.read(gpu.device()).await.unwrap();
                    println!("Current result value: {}", result_read[0]);
                }
            }
            Err(e) => {
                // Hot-reloading failed, likely due to a syntax error in the shader.
                println!("Hot reloading error: {:?}", e);
            }
        }
    }

    Ok(())
}