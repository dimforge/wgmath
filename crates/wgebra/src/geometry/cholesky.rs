use wgcore::{test_shader_compilation, Shader};

fn substitute2(src: &str) -> String {
    src.replace("DIM", "2")
        .replace("MAT", "mat2x2<f32>")
        .replace("IMPORT_PATH", "wgebra::cholesky2")
}

fn substitute3(src: &str) -> String {
    src.replace("DIM", "3")
        .replace("MAT", "mat3x3<f32>")
        .replace("IMPORT_PATH", "wgebra::cholesky3")
}

fn substitute4(src: &str) -> String {
    src.replace("DIM", "4")
        .replace("MAT", "mat4x4<f32>")
        .replace("IMPORT_PATH", "wgebra::cholesky4")
}

#[derive(Shader)]
#[shader(src = "cholesky.wgsl", src_fn = "substitute2")]
/// Shader for computing the Cholesky decomposition of a symmetric-definite-positive 2x2 matrix.
pub struct WgCholesky2;

#[derive(Shader)]
#[shader(src = "cholesky.wgsl", src_fn = "substitute3")]
/// Shader for computing the Cholesky decomposition of a symmetric-definite-positive 2x2 matrix.
pub struct WgCholesky3;

#[derive(Shader)]
#[shader(src = "cholesky.wgsl", src_fn = "substitute4")]
/// Shader for computing the Cholesky decomposition of a symmetric-definite-positive 2x2 matrix.
pub struct WgCholesky4;

test_shader_compilation!(WgCholesky2);
test_shader_compilation!(WgCholesky3);
test_shader_compilation!(WgCholesky4);

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use naga_oil::compose::Composer;
    use nalgebra::{DVector, Matrix2, Matrix3, Matrix4, Matrix4x3};
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
    var<storage, read_write> out: array<MAT>;

    @compute @workgroup_size(1, 1, 1)
    fn test(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
        let i = invocation_id.x;
        out[i] = cholesky(in[i]);
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
        ($name: ident, $kernel: ident, $mat: ident, $substitute: ident, $dim: expr) => {
            #[futures_test::test]
            #[serial_test::serial]
            async fn $name() {
                let gpu = GpuInstance::new().await.unwrap();
                let chol = test_pipeline::<super::$kernel>(gpu.device(), super::$substitute);
                let mut encoder = gpu.device().create_command_encoder(&Default::default());

                type Mat = $mat<f32>;

                const LEN: usize = 345;
                let mut matrices: DVector<Mat> = DVector::new_random(LEN);
                for i in 0..matrices.len() {
                    let sdp = matrices[i].fixed_rows::<$dim>(0).transpose()
                        * matrices[i].fixed_rows::<$dim>(0);
                    matrices[i].fixed_rows_mut::<$dim>(0).copy_from(&sdp);
                }

                let inputs = GpuVector::init(gpu.device(), &matrices, BufferUsages::STORAGE);
                let result: GpuVector<Mat> = GpuVector::uninit(
                    gpu.device(),
                    matrices.len() as u32,
                    BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                );
                let staging: GpuVector<Mat> = GpuVector::uninit(
                    gpu.device(),
                    matrices.len() as u32,
                    BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                );

                // Dispatch the test.
                let mut pass = encoder.compute_pass("test", None);
                KernelDispatch::new(gpu.device(), &mut pass, &chol)
                    .bind0([inputs.buffer(), result.buffer()])
                    .dispatch(matrices.len() as u32);
                drop(pass); // Ensure the pass is ended before the encoder is borrowed again.

                // Submit.
                staging.copy_from(&mut encoder, &result);
                gpu.queue().submit(Some(encoder.finish()));

                // Check the result is correct.
                let gpu_result = staging.read(gpu.device()).await.unwrap();

                let mut allowed_fails = 0;

                for (m, chol) in matrices.iter().zip(gpu_result.iter()) {
                    if let Some(chol_cpu) = m.fixed_rows::<$dim>(0).cholesky() {
                        let chol = chol.fixed_rows::<$dim>(0).into_owned();

                        if allowed_fails == matrices.len() / 100 {
                            assert_relative_eq!(chol_cpu.unpack_dirty(), chol, epsilon = 1.0e-3);
                        } else if !approx::relative_eq!(
                            chol_cpu.unpack_dirty(),
                            chol,
                            epsilon = 1.0e-3
                        ) {
                            allowed_fails += 1;
                        }
                    }
                }

                println!("Num fails: {}/{}", allowed_fails, matrices.len());
            }
        };
    }

    gen_test!(gpu_cholesky2, WgCholesky2, Matrix2, substitute2, 2);
    // NOTE: for the 3x3 test we need Matrix4x3 to account for the WGSL mat4x3 padding/alignment.
    gen_test!(gpu_cholesky3, WgCholesky3, Matrix4x3, substitute3, 3);
    gen_test!(gpu_cholesky4, WgCholesky4, Matrix4, substitute4, 4);
}
