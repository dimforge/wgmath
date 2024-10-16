use nalgebra::{Matrix2, Vector2};
use wgcore::Shader;
#[cfg(test)]
use {
    naga_oil::compose::NagaModuleDescriptor,
    wgpu::{ComputePipeline, Device},
};

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
/// GPU representation of a 2x2 matrix SVD.
pub struct GpuSvd2 {
    /// First orthogonal matrix of the SVD.
    pub u: Matrix2<f32>,
    /// Singular values.
    pub s: Vector2<f32>,
    /// Second orthogonal matrix of the SVD.
    pub vt: Matrix2<f32>,
}

#[derive(Shader)]
#[shader(src = "svd2.wgsl")]
/// Shader for computing the Singular Value Decomposition of 2x2 matrices.
pub struct WgSvd2;

impl WgSvd2 {
    #[cfg(test)]
    pub fn tests(device: &Device) -> ComputePipeline {
        let test_kernel = r#"
 @group(0) @binding(0)
 var<storage, read_write> in: array<mat2x2<f32>>;
 @group(0) @binding(1)
 var<storage, read_write> out: array<Svd>;

 @compute @workgroup_size(1, 1, 1)
 fn test(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
     let i = invocation_id.x;
     out[i] = svd(in[i]);
 }
        "#;

        let src = format!("{}\n{}", Self::src(), test_kernel);
        let module = naga_oil::compose::Composer::default()
            .make_naga_module(NagaModuleDescriptor {
                source: &src,
                file_path: Self::FILE_PATH,
                ..Default::default()
            })
            .unwrap();
        wgcore::utils::load_module(device, "test", module)
    }
}

#[cfg(test)]
mod test {
    use super::GpuSvd2;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, Matrix2};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
    use wgcore::tensor::GpuVector;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_svd2() {
        let gpu = GpuInstance::new().await.unwrap();
        let svd = super::WgSvd2::tests(gpu.device());
        let mut queue = KernelInvocationQueue::new(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const LEN: usize = 345;
        let mut matrices: DVector<Matrix2<f32>> = DVector::new_random(LEN);
        matrices[0] = Matrix2::zeros(); // The zero matrix can cause issues on some platforms (like macos) with unspecified atan2 on (0, 0).
        matrices[1] = Matrix2::identity(); // The identity matrix can cause issues on some platforms.
        let inputs = GpuVector::init(gpu.device(), &matrices, BufferUsages::STORAGE);
        let result: GpuVector<GpuSvd2> = GpuVector::uninit(
            gpu.device(),
            matrices.len() as u32,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let staging: GpuVector<GpuSvd2> = GpuVector::uninit(
            gpu.device(),
            matrices.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        // Queue the test.
        KernelInvocationBuilder::new(&mut queue, &svd)
            .bind0([inputs.buffer(), result.buffer()])
            .queue(matrices.len() as u32);

        // Run.
        queue.encode(&mut encoder, None);
        staging.copy_from(&mut encoder, &result);
        gpu.queue().submit(Some(encoder.finish()));

        // Check the result is correct.
        let gpu_result = staging.read(gpu.device()).await.unwrap();

        for (m, svd) in matrices.iter().zip(gpu_result.iter()) {
            let reconstructed = svd.u * Matrix2::from_diagonal(&svd.s) * svd.vt;
            assert_relative_eq!(*m, reconstructed, epsilon = 1.0e-4);
        }
    }
}
