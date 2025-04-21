use nalgebra::{Matrix2, Vector2};
use wgcore::Shader;
#[cfg(test)]
use {
    crate::utils::WgTrig,
    naga_oil::compose::NagaModuleDescriptor,
    wgpu::{ComputePipeline, Device},
};

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
/// GPU representation of a symmetric 2x2 matrix eigendecomposition.
///
/// See the [nalgebra](https://nalgebra.rs/docs/user_guide/decompositions_and_lapack/#eigendecomposition-of-a-hermitian-matrix)
/// documentation for details on the eigendecomposition
pub struct GpuSymmetricEigen2 {
    /// Eigenvectors of the matrix.
    pub eigenvectors: Matrix2<f32>,
    /// Eigenvalues of the matrix.
    pub eigenvalues: Vector2<f32>,
}

#[derive(Shader)]
#[shader(src = "eig2.wgsl")]
/// Shader for computing the eigendecomposition of symmetric 2x2 matrices.
pub struct WgSymmetricEigen2;

impl WgSymmetricEigen2 {
    #[doc(hidden)]
    #[cfg(test)]
    pub fn tests(device: &Device) -> ComputePipeline {
        let test_kernel = r#"
 @group(0) @binding(0)
 var<storage, read_write> in: array<mat2x2<f32>>;
 @group(0) @binding(1)
 var<storage, read_write> out: array<SymmetricEigen>;

 @compute @workgroup_size(1, 1, 1)
 fn test(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
     let i = invocation_id.x;
     out[i] = symmetric_eigen(in[i]);
 }
        "#;

        let src = format!("{}\n{}", Self::src(), test_kernel);
        let module = WgTrig::composer()
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
    use super::GpuSymmetricEigen2;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, Matrix2};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::{CommandEncoderExt, KernelDispatch};
    use wgcore::tensor::GpuVector;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_eig2() {
        let gpu = GpuInstance::new().await.unwrap();
        let svd = super::WgSymmetricEigen2::tests(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const LEN: usize = 345;
        let mut matrices: DVector<Matrix2<f32>> = DVector::new_random(LEN);
        // matrices[0] = Matrix2::zeros(); // The zero matrix can cause issues on some platforms (like macos) with unspecified atan2 on (0, 0).
        // matrices[1] = Matrix2::identity(); // The identity matrix can cause issues on some platforms.
        for mat in matrices.iter_mut() {
            *mat = mat.transpose() * *mat; // Make it symmetric.
        }

        let inputs = GpuVector::init(gpu.device(), &matrices, BufferUsages::STORAGE);
        let result: GpuVector<GpuSymmetricEigen2> = GpuVector::uninit(
            gpu.device(),
            matrices.len() as u32,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let staging: GpuVector<GpuSymmetricEigen2> = GpuVector::uninit(
            gpu.device(),
            matrices.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        // Dispatch the test.
        let mut pass = encoder.compute_pass("test", None);
        KernelDispatch::new(gpu.device(), &mut pass, &svd)
            .bind0([inputs.buffer(), result.buffer()])
            .dispatch(matrices.len() as u32);
        drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

        staging.copy_from(&mut encoder, &result);
        gpu.queue().submit(Some(encoder.finish()));

        // Check the result is correct.
        let gpu_result = staging.read(gpu.device()).await.unwrap();

        for (m, eigen) in matrices.iter().zip(gpu_result.iter()) {
            let reconstructed = eigen.eigenvectors
                * Matrix2::from_diagonal(&eigen.eigenvalues)
                * eigen.eigenvectors.transpose();
            assert_relative_eq!(*m, reconstructed, epsilon = 1.0e-4);
        }
    }
}
