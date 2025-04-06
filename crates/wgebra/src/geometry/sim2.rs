use crate::WgRot2;
use nalgebra::{Isometry2, Similarity2};
use wgcore::Shader;

#[derive(Copy, Clone, PartialEq, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
/// A GPU-compatible 2D similarity (uniform scale + rotation + translation).
pub struct GpuSim2 {
    /// The similarity value.
    pub similarity: Similarity2<f32>,
    // TODO: use encase instead of an explicit padding.
    /// Extra padding to fit the layout on the gpu. Its value is irrelevant.
    pub padding: f32,
}

impl GpuSim2 {
    /// The identity similarity (scale = 1, rotation = identity, translation = 0).
    pub fn identity() -> Self {
        Self {
            similarity: Similarity2::identity(),
            padding: 0.0,
        }
    }
}

impl From<Similarity2<f32>> for GpuSim2 {
    fn from(value: Similarity2<f32>) -> Self {
        Self {
            similarity: value,
            padding: 0.0,
        }
    }
}

impl From<Isometry2<f32>> for GpuSim2 {
    fn from(value: Isometry2<f32>) -> Self {
        Self {
            similarity: Similarity2::from_isometry(value, 1.0),
            padding: 0.0,
        }
    }
}

#[derive(Shader)]
#[shader(derive(WgRot2), src = "sim2.wgsl")]
/// Shader exposing a 2D similarity (uniform scale + rotation + translation) type and operations.
pub struct WgSim2;

impl WgSim2 {
    #[cfg(test)]
    pub fn tests(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let test_kernel = r#"
@group(0) @binding(0)
var<storage, read_write> test_s1: array<Sim2>;
@group(0) @binding(1)
var<storage, read_write> test_s2: array<Sim2>;
@group(0) @binding(2)
var<storage, read_write> test_p1: array<vec2<f32>>;
@group(0) @binding(3)
var<storage, read_write> test_p2: array<vec2<f32>>;
@group(0) @binding(4)
var<storage, read_write> test_v1: array<vec2<f32>>;
@group(0) @binding(5)
var<storage, read_write> test_v2: array<vec2<f32>>;
@group(0) @binding(6)
var<storage, read_write> test_id: Sim2;

@compute @workgroup_size(1, 1, 1)
fn test(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;
    test_p1[i] = mulPt(test_s1[i], test_p1[i]);
    test_p2[i] = invMulPt(test_s2[i],test_p2[i]);
    test_v1[i] = mulVec(test_s1[i], test_v1[i]);
    test_v2[i] = invMulVec(test_s2[i],test_v2[i]);
    test_s1[i] = mul(test_s1[i], test_s2[i]);
    test_s2[i] = inv(test_s2[i]);

    if i == 0 {
        test_id = identity();
    }
}
        "#;

        let src = format!("{}\n{}", Self::src(), test_kernel);
        let module = Self::composer()
            .unwrap()
            .make_naga_module(naga_oil::compose::NagaModuleDescriptor {
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
    use crate::GpuSim2;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, Point2, Similarity2, Vector2};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
    use wgcore::tensor::{GpuScalar, GpuVector};
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_sim2() {
        let gpu = GpuInstance::new().await.unwrap();
        let sim2 = super::WgSim2::tests(gpu.device());
        let mut queue = KernelInvocationQueue::new(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const LEN: u32 = 345;

        let unaligned_s1: DVector<Similarity2<f32>> = DVector::new_random(LEN as usize);
        let unaligned_s2: DVector<Similarity2<f32>> = DVector::new_random(LEN as usize);
        let test_s1: DVector<GpuSim2> = unaligned_s1.map(GpuSim2::from);
        let test_s2: DVector<GpuSim2> = unaligned_s2.map(GpuSim2::from);
        let test_p1: DVector<Point2<f32>> = DVector::new_random(LEN as usize);
        let test_p2: DVector<Point2<f32>> = DVector::new_random(LEN as usize);
        let test_v1: DVector<Vector2<f32>> = DVector::new_random(LEN as usize);
        let test_v2: DVector<Vector2<f32>> = DVector::new_random(LEN as usize);
        let test_id = GpuSim2::identity();

        let usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let gpu_test_s1 = GpuVector::init(gpu.device(), &test_s1, usages);
        let gpu_test_s2 = GpuVector::init(gpu.device(), &test_s2, usages);
        let gpu_test_p1 = GpuVector::init(gpu.device(), &test_p1, usages);
        let gpu_test_p2 = GpuVector::init(gpu.device(), &test_p2, usages);
        let gpu_test_v1 = GpuVector::init(gpu.device(), &test_v1, usages);
        let gpu_test_v2 = GpuVector::init(gpu.device(), &test_v2, usages);
        let gpu_test_id = GpuScalar::init(gpu.device(), test_id, usages);

        let usages = BufferUsages::MAP_READ | BufferUsages::COPY_DST;
        let staging_test_s1 = GpuVector::uninit(gpu.device(), LEN, usages);
        let staging_test_s2 = GpuVector::uninit(gpu.device(), LEN, usages);
        let staging_test_p1 = GpuVector::uninit(gpu.device(), LEN, usages);
        let staging_test_p2 = GpuVector::uninit(gpu.device(), LEN, usages);
        let staging_test_v1 = GpuVector::uninit(gpu.device(), LEN, usages);
        let staging_test_v2 = GpuVector::uninit(gpu.device(), LEN, usages);
        let staging_test_id = GpuScalar::uninit(gpu.device(), usages);

        KernelInvocationBuilder::new(&mut queue, &sim2)
            .bind0([
                gpu_test_s1.buffer(),
                gpu_test_s2.buffer(),
                gpu_test_p1.buffer(),
                gpu_test_p2.buffer(),
                gpu_test_v1.buffer(),
                gpu_test_v2.buffer(),
                gpu_test_id.buffer(),
            ])
            .queue(LEN);

        queue.encode(&mut encoder, None);
        staging_test_s1.copy_from(&mut encoder, &gpu_test_s1);
        staging_test_s2.copy_from(&mut encoder, &gpu_test_s2);
        staging_test_p1.copy_from(&mut encoder, &gpu_test_p1);
        staging_test_p2.copy_from(&mut encoder, &gpu_test_p2);
        staging_test_v1.copy_from(&mut encoder, &gpu_test_v1);
        staging_test_v2.copy_from(&mut encoder, &gpu_test_v2);
        staging_test_id.copy_from(&mut encoder, &gpu_test_id);
        gpu.queue().submit(Some(encoder.finish()));

        let result_s1 = staging_test_s1.read(gpu.device()).await.unwrap();
        let result_s2 = staging_test_s2.read(gpu.device()).await.unwrap();
        let result_p1 = staging_test_p1.read(gpu.device()).await.unwrap();
        let result_p2 = staging_test_p2.read(gpu.device()).await.unwrap();
        let result_v1 = staging_test_v1.read(gpu.device()).await.unwrap();
        let result_v2 = staging_test_v2.read(gpu.device()).await.unwrap();
        let result_id = staging_test_id.read(gpu.device()).await.unwrap();

        for i in 0..LEN as usize {
            assert_relative_eq!(
                result_s1[i].similarity,
                unaligned_s1[i] * unaligned_s2[i],
                epsilon = 1.0e-5
            );
            assert_relative_eq!(
                result_s2[i].similarity,
                unaligned_s2[i].inverse(),
                epsilon = 1.0e-4
            );
            assert_relative_eq!(result_p1[i], unaligned_s1[i] * test_p1[i], epsilon = 1.0e-4);
            assert_relative_eq!(
                result_p2[i],
                unaligned_s2[i].inverse_transform_point(&test_p2[i]),
                epsilon = 1.0e-4
            );
            assert_relative_eq!(result_v1[i], unaligned_s1[i] * test_v1[i], epsilon = 1.0e-4);
            assert_relative_eq!(
                result_v2[i],
                unaligned_s2[i].inverse_transform_vector(&test_v2[i]),
                epsilon = 1.0e-4
            );
        }

        assert_eq!(result_id[0].similarity, Similarity2::identity());
    }
}
