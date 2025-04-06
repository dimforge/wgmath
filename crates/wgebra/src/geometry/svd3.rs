use crate::WgQuat;
use nalgebra::{Matrix4x3, Vector4};
use wgcore::Shader;
#[cfg(test)]
use {
    naga_oil::compose::NagaModuleDescriptor,
    wgpu::{ComputePipeline, Device},
};

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
/// A 3D SVD as represented on the gpu, with padding (every fourth rows
/// can be ignored).
// TODO: switch to encase?
pub struct GpuSvd3 {
    /// First orthogonal matrix of the SVD.
    u: Matrix4x3<f32>,
    /// Singular values.
    s: Vector4<f32>,
    /// Second orthogonal matrix of the SVD.
    vt: Matrix4x3<f32>,
}

#[derive(Shader)]
#[shader(derive(WgQuat), src = "svd3.wgsl")]
/// Shader for computing the Singular Value Decomposition of 3x3 matrices.
pub struct WgSvd3;

impl WgSvd3 {
    #[cfg(test)]
    pub fn tests(device: &Device) -> ComputePipeline {
        let test_kernel = r#"
@group(0) @binding(0)
var<storage, read_write> in: array<mat3x3<f32>>;
@group(0) @binding(1)
var<storage, read_write> out: array<Svd>;

@compute @workgroup_size(1, 1, 1)
fn test(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;
    out[i] = svd(in[i]);
}
        "#;

        let src = format!("{}\n{}", Self::src(), test_kernel);
        let module = WgQuat::composer()
            .unwrap()
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
    use super::GpuSvd3;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, Matrix3, Matrix4x3};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
    use wgcore::tensor::GpuVector;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_svd3() {
        let gpu = GpuInstance::new().await.unwrap();
        let svd = super::WgSvd3::tests(gpu.device());
        let mut queue = KernelInvocationQueue::new(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const LEN: usize = 345;
        let matrices: DVector<Matrix4x3<f32>> = DVector::new_random(LEN);
        let inputs = GpuVector::init(gpu.device(), &matrices, BufferUsages::STORAGE);
        let result: GpuVector<GpuSvd3> = GpuVector::uninit(
            gpu.device(),
            matrices.len() as u32,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let staging: GpuVector<GpuSvd3> = GpuVector::uninit(
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
            let m = m.fixed_rows::<3>(0).into_owned();
            let reconstructed = svd.u.fixed_rows::<3>(0).into_owned()
                * Matrix3::from_diagonal(&svd.s.fixed_rows::<3>(0))
                * svd.vt.fixed_rows::<3>(0).into_owned();
            assert_relative_eq!(m, reconstructed, epsilon = 1.0e-4);
        }
    }
}
