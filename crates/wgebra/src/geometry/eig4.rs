use crate::utils::WgMinMax;
use crate::{WgRot2, WgSymmetricEigen2};
use nalgebra::{Matrix4, Vector4};
use wgcore::{test_shader_compilation, Shader};
#[cfg(test)]
use {
    naga_oil::compose::NagaModuleDescriptor,
    wgpu::{ComputePipeline, Device},
};

#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
/// GPU representation of a symmetric 4x4 matrix eigendecomposition.
///
/// See the [nalgebra](https://nalgebra.rs/docs/user_guide/decompositions_and_lapack/#eigendecomposition-of-a-hermitian-matrix)
/// documentation for details on the eigendecomposition
pub struct GpuSymmetricEigen4 {
    /// Eigenvectors of the matrix.
    pub eigenvectors: Matrix4<f32>,
    /// Eigenvalues of the matrix.
    pub eigenvalues: Vector4<f32>,
}

#[derive(Shader)]
#[shader(derive(WgMinMax, WgSymmetricEigen2, WgRot2), src = "eig4.wgsl")]
/// Shader for computing the eigendecomposition of symmetric 4x4 matrices.
pub struct WgSymmetricEigen4;

test_shader_compilation!(WgSymmetricEigen4);

impl WgSymmetricEigen4 {
    #[cfg(test)]
    pub fn tests(device: &Device) -> ComputePipeline {
        let test_kernel = r#"
 @group(0) @binding(0)
 var<storage, read_write> in: array<mat4x4<f32>>;
 @group(0) @binding(1)
 var<storage, read_write> out: array<SymmetricEigen>;

 @compute @workgroup_size(1, 1, 1)
 fn test(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
     let i = invocation_id.x;
     out[i] = symmetric_eigen(in[i]);
 }
        "#;

        let src = format!("{}\n{}", Self::src(), test_kernel);
        let module = WgSymmetricEigen4::composer()
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
    use super::GpuSymmetricEigen4;
    use approx::{assert_relative_eq, relative_eq};
    use nalgebra::{DVector, Matrix4};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::{CommandEncoderExt, KernelDispatch};
    use wgcore::tensor::GpuVector;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_eig4() {
        let gpu = GpuInstance::new().await.unwrap();
        let svd = super::WgSymmetricEigen4::tests(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const LEN: usize = 345;
        let mut matrices: DVector<Matrix4<f32>> = DVector::new_random(LEN);
        for mat in matrices.iter_mut() {
            *mat = mat.transpose() * *mat; // Make it symmetric.
        }

        let inputs = GpuVector::init(gpu.device(), &matrices, BufferUsages::STORAGE);
        let result: GpuVector<GpuSymmetricEigen4> = GpuVector::uninit(
            gpu.device(),
            matrices.len() as u32,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let staging: GpuVector<GpuSymmetricEigen4> = GpuVector::uninit(
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
        let mut allowed_fails = 0;

        for (m, eigen) in matrices.iter().zip(gpu_result.iter()) {
            println!("eig: (gpu) {:?}", eigen);
            println!("eig (na):      {:?}", m.symmetric_eigen());

            let reconstructed = eigen.eigenvectors
                * Matrix4::from_diagonal(&eigen.eigenvalues)
                * eigen.eigenvectors.transpose();
            println!("reconstructed: {:?}", m.symmetric_eigen().recompose());

            // NOTE: we allow about 2% of the decompositions to fail, to account for occasionally
            //       bad random matrices that will fail the test due to an unsuitable epsilon.
            //       Ideally this percentage should be kept as low as possible, but likely not
            //       removable entirely.
            if allowed_fails == matrices.len() * 2 / 100 {
                assert_relative_eq!(*m, reconstructed, epsilon = 1.0e-4);
            } else if !relative_eq!(*m, reconstructed, epsilon = 1.0e-4) {
                allowed_fails += 1;
            }
        }

        println!("Num fails: {}/{}", allowed_fails, matrices.len());
    }
}
