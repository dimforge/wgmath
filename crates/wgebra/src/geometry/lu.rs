use nalgebra::{SMatrix, SVector};
use wgcore::re_exports::encase::ShaderType;
use wgcore::{test_shader_compilation, Shader};

fn substitute2(src: &str) -> String {
    src.replace("NROWS", "2u")
        .replace("NCOLS", "2u")
        .replace("PERM", "vec2<u32>")
        .replace("MAT", "mat2x2<f32>")
        .replace("IMPORT_PATH", "wgebra::lu2")
}

fn substitute3(src: &str) -> String {
    src.replace("NROWS", "3u")
        .replace("NCOLS", "3u")
        .replace("PERM", "vec3<u32>")
        .replace("MAT", "mat3x3<f32>")
        .replace("IMPORT_PATH", "wgebra::lu3")
}

fn substitute4(src: &str) -> String {
    src.replace("NROWS", "4u")
        .replace("NCOLS", "4u")
        .replace("PERM", "vec4<u32>")
        .replace("MAT", "mat4x4<f32>")
        .replace("IMPORT_PATH", "wgebra::lu4")
}

macro_rules! gpu_output_types(
    ($GpuPermutation: ident, $GpuLU: ident, $R: literal, $C: literal, $Perm: literal) => {
        #[derive(ShaderType, Copy, Clone, PartialEq)]
        #[repr(C)]
        pub struct $GpuPermutation {
            pub ia: SVector<u32, $Perm>,
            pub ib: SVector<u32, $Perm>,
            pub len: u32,
        }

        #[derive(ShaderType, Copy, Clone, PartialEq)]
        #[repr(C)]
        pub struct $GpuLU {
            pub lu: SMatrix<f32, $R, $C>,
            pub p: $GpuPermutation,
        }
    }
);

gpu_output_types!(GpuPermutations2, GpuLU2, 2, 2, 2);
gpu_output_types!(GpuPermutations3, GpuLU3, 3, 3, 3);
gpu_output_types!(GpuPermutations4, GpuLU4, 4, 4, 4);

// TODO: rectangular matrices
#[derive(Shader)]
#[shader(src = "lu.wgsl", src_fn = "substitute2")]
/// Shader for computing the Cholesky decomposition of a symmetric-definite-positive 2x2 matrix.
pub struct WgLU2;

#[derive(Shader)]
#[shader(src = "lu.wgsl", src_fn = "substitute3")]
/// Shader for computing the Cholesky decomposition of a symmetric-definite-positive 2x2 matrix.
pub struct WgLU3;

#[derive(Shader)]
#[shader(src = "lu.wgsl", src_fn = "substitute4")]
/// Shader for computing the Cholesky decomposition of a symmetric-definite-positive 2x2 matrix.
pub struct WgLU4;

test_shader_compilation!(WgLU2);
test_shader_compilation!(WgLU3);
test_shader_compilation!(WgLU4);

#[cfg(test)]
mod test {
    use super::{GpuLU2, GpuLU3, GpuLU4};
    use approx::assert_relative_eq;
    use naga_oil::compose::Composer;
    use nalgebra::{DVector, Matrix2, Matrix4, Matrix4x3};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::{CommandEncoderExt, KernelDispatch};
    use wgcore::tensor::GpuVector;
    use wgcore::Shader;
    use wgpu::BufferUsages;
    use {
        naga_oil::compose::NagaModuleDescriptor,
        wgpu::{ComputePipeline, Device},
    };

    pub fn test_pipeline<S: Shader>(
        device: &Device,
        substitute: fn(&str) -> String,
    ) -> ComputePipeline {
        let test_kernel = r#"
    @group(0) @binding(0)
    var<storage, read_write> in: array<MAT>;
    @group(0) @binding(1)
    var<storage, read_write> out: array<LU>;

    @compute @workgroup_size(1, 1, 1)
    fn test(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
        let i = invocation_id.x;
        out[i] = lu(in[i]);
    }
            "#;

        let src = substitute(&format!("{}\n{}", S::src(), test_kernel));
        let module = Composer::default()
            .make_naga_module(NagaModuleDescriptor {
                source: &src,
                file_path: "",
                ..Default::default()
            })
            .unwrap();
        wgcore::utils::load_module(device, "test", module)
    }

    macro_rules! gen_test {
        ($name: ident, $kernel: ident, $mat: ident, $out: ident, $substitute: ident, $dim: expr) => {
            #[futures_test::test]
            #[serial_test::serial]
            async fn $name() {
                let gpu = GpuInstance::new().await.unwrap();
                let lu = test_pipeline::<super::$kernel>(gpu.device(), super::$substitute);
                let mut encoder = gpu.device().create_command_encoder(&Default::default());

                type Mat = $mat<f32>;
                type GpuOut = $out;

                const LEN: usize = 345;
                let mut matrices: DVector<Mat> = DVector::new_random(LEN);
                for i in 0..matrices.len() {
                    let sdp = matrices[i].fixed_rows::<$dim>(0).transpose()
                        * matrices[i].fixed_rows::<$dim>(0);
                    matrices[i].fixed_rows_mut::<$dim>(0).copy_from(&sdp);
                }

                let inputs = GpuVector::init(gpu.device(), &matrices, BufferUsages::STORAGE);
                let result: GpuVector<GpuOut> = GpuVector::uninit_encased(
                    gpu.device(),
                    matrices.len() as u32,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                );
                let staging: GpuVector<GpuOut> = GpuVector::uninit_encased(
                    gpu.device(),
                    matrices.len() as u32,
                    BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                );

                // Dispatch the test.
                let mut pass = encoder.compute_pass("test", None);
                KernelDispatch::new(gpu.device(), &mut pass, &lu)
                    .bind0([inputs.buffer(), result.buffer()])
                    .dispatch(matrices.len() as u32);
                drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

                // Submit.
                staging.copy_from_encased(&mut encoder, &result);
                gpu.queue().submit(Some(encoder.finish()));

                // Check the result is correct.
                let gpu_result = staging.read_encased(gpu.device()).await.unwrap();

                for (m, lu) in matrices.iter().zip(gpu_result.iter()) {
                    let lu_cpu = m.fixed_rows::<$dim>(0).lu();
                    assert_relative_eq!(lu_cpu.lu_internal(), &lu.lu, epsilon = 1.0e-3);
                    // TODO: check the permutation vectors
                }
            }
        };
    }

    gen_test!(gpu_lu2, WgLU2, Matrix2, GpuLU2, substitute2, 2);
    // NOTE: for the 3x3 test we need Matrix4x3 to account for the WGSL mat4x3 padding/alignment.
    gen_test!(gpu_lu3, WgLU3, Matrix4x3, GpuLU3, substitute3, 3);
    gen_test!(gpu_lu4, WgLU4, Matrix4, GpuLU4, substitute4, 4);
}
