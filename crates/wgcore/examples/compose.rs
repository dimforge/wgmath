use nalgebra::DVector;
use wgcore::composer::ComposerExt;
use wgcore::gpu::GpuInstance;
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::tensor::GpuVector;
use wgcore::Shader;
use wgpu::{BufferUsages, ComputePipeline};

// Declare our shader module that contains our composable functions.
// Note that we don’t build any compute pipeline from this wgsl file.
#[derive(Shader)]
#[shader(
    src = "composable.wgsl" // Shader source code, will be embedded in the exe with `include_str!`
)]
struct Composable;

#[derive(Shader)]
#[shader(
    derive(Composable), // This shader depends on the `Composable` shader.
    src = "kernel.wgsl",  // Shader source code, will be embedded in the exe with `include_str!`.
    composable = false    // This shader doesn’t export any symbols reusable from other wgsl shaders.
)]
struct WgKernel {
    main: ComputePipeline,
}

#[derive(Copy, Clone, PartialEq, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MyStruct {
    value: f32,
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
    let kernel = WgKernel::from_device(gpu.device());

    // Create the buffers.
    const LEN: u32 = 1000;
    let a_data = DVector::from_fn(LEN as usize, |i, _| MyStruct { value: i as f32 });
    let b_data = DVector::from_fn(LEN as usize, |i, _| MyStruct {
        value: i as f32 * 10.0,
    });
    let a_buf = GpuVector::init(gpu.device(), &a_data, BufferUsages::STORAGE);
    let b_buf = GpuVector::init(gpu.device(), &b_data, BufferUsages::STORAGE);

    // Queue the operation.
    let mut queue = KernelInvocationQueue::new(gpu.device());
    KernelInvocationBuilder::new(&mut queue, &kernel.main)
        .bind0([a_buf.buffer(), b_buf.buffer()])
        .queue(LEN.div_ceil(64));

    // Encode & submit the operation to the gpu.
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    queue.encode(&mut encoder, None);
    gpu.queue().submit(Some(encoder.finish()));

    Ok(())
}
