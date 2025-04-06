use nalgebra::DVector;
use wgcore::gpu::GpuInstance;
use wgcore::tensor::GpuVector;
use wgpu::BufferUsages;

#[async_std::main]
async fn main() -> anyhow::Result<()> {
    // Initialize the gpu device and its queue.
    //
    // Note that `GpuInstance` is just a simple helper struct for initializing the gpu resources.
    // You are free to initialize them independently if more control is needed, or reuse the ones
    // that were already created/owned by e.g., a game engine.
    let gpu = GpuInstance::new().await?;

    // Create the buffers.
    const LEN: u32 = 10;
    let buffer_data = DVector::from_fn(LEN as usize, |i, _| i as u32);
    let buffer = GpuVector::init(
        gpu.device(),
        &buffer_data,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let staging = GpuVector::uninit(
        gpu.device(),
        LEN,
        BufferUsages::COPY_DST | BufferUsages::MAP_READ,
    );

    // Queue the operation.
    // Encode & submit the operation to the gpu.
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    // Copy the result to the staging buffer.
    staging.copy_from(&mut encoder, &buffer);
    gpu.queue().submit(Some(encoder.finish()));

    let read = DVector::from(staging.read(gpu.device()).await?);
    assert_eq!(buffer_data, read);
    println!("Buffer copy & read succeeded!");
    println!("Original: {:?}", buffer_data);
    println!("Readback: {:?}", read);

    Ok(())
}
