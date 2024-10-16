//! The cuboid shape.

use crate::math::{Point, Vector};
use crate::ray::WgRay;
use crate::{dim_shader_defs, substitute_aliases};
use encase::ShaderType;
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Copy, Clone, ShaderType)]
#[cfg_attr(feature = "dim2", derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
/// A cuboid shape with a layout compatible with the corresponding wgsl struct.
pub struct GpuCuboid {
    /// The cuboid’s half-width along each cordinate axis.
    pub half_extents: Vector<f32>,
}

#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
/// The result of point-projection with a layout compatible with the corresponding wgsl struct.
pub struct GpuProjectionResult {
    /// The projected point.
    pub point: Point<f32>,
    /// Was the point to project inside the shape? (0 = false, other values = true)
    pub is_inside: u32,
    #[cfg(feature = "dim2")]
    /// Extra padding. Its value is irrelevant.
    pub padding: u32,
}

impl GpuCuboid {
    /// Creates a `GpuCuboid` from its half-extents (half width along each axis).
    pub fn new(half_extents: Vector<f32>) -> Self {
        Self { half_extents }
    }
}

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgRay),
    src = "cuboid.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Shader defining the cuboid shape as well as its ray-casting and point-projection functions.
pub struct WgCuboid;

impl WgCuboid {
    #[cfg(test)]
    pub fn tests(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let test_kernel = substitute_aliases(
            r#"
struct ProjectionResultHostShareable {
    point: Vector,
    is_inside: u32,
}

@group(0) @binding(0)
var<storage, read> test_cuboids: array<Cuboid>;
@group(0) @binding(1)
var<storage, read> test_points: array<Vector>;
@group(0) @binding(2)
var<storage, read_write> projs: array<Vector>;
@group(0) @binding(3)
var<storage, read_write> projs_on_boundary: array<ProjectionResultHostShareable>;

@compute @workgroup_size(1, 1, 1)
fn test(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let i = invocation_id.x;
    let point = test_points[i];
    projs[i] = projectLocalPoint(test_cuboids[i], point);

    let proj = projectLocalPointOnBoundary(test_cuboids[i], point);
    projs_on_boundary[i] = ProjectionResultHostShareable(proj.point, u32(proj.is_inside));
}
        "#,
        );

        let src = format!("{}\n{}", Self::src(), test_kernel);
        let module = Self::composer()
            .make_naga_module(naga_oil::compose::NagaModuleDescriptor {
                source: &src,
                file_path: Self::FILE_PATH,
                shader_defs: dim_shader_defs(),
                ..Default::default()
            })
            .unwrap();
        wgcore::utils::load_module(device, "test", module)
    }
}

#[cfg(test)]
mod test {
    use super::{GpuCuboid, GpuProjectionResult};
    use nalgebra::vector;
    #[cfg(feature = "dim2")]
    use parry2d::{query::PointQuery, shape::Cuboid};
    #[cfg(feature = "dim3")]
    use parry3d::{query::PointQuery, shape::Cuboid};
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
    use wgcore::tensor::GpuVector;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_cuboid() {
        let gpu = GpuInstance::new().await.unwrap();
        let wg_cuboid = super::WgCuboid::tests(gpu.device());
        let mut queue = KernelInvocationQueue::new(gpu.device());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());

        const LEN: u32 = 30;

        #[cfg(feature = "dim2")]
        let cuboid = GpuCuboid::new(vector![1.0, 2.0]);
        #[cfg(feature = "dim3")]
        let cuboid = GpuCuboid::new(vector![1.0, 2.0, 3.0]);

        let mut points = vec![];
        let step = cuboid.half_extents * 4.0 / (LEN as f32);
        let nk = if cfg!(feature = "dim2") { 1 } else { LEN };

        for i in 0..LEN {
            for j in 0..LEN {
                for _k in 0..nk {
                    let origin = -cuboid.half_extents * 2.0;
                    #[cfg(feature = "dim2")]
                    let pt = origin + vector![i as f32, j as f32].component_mul(&step);
                    #[cfg(feature = "dim3")]
                    let pt = (origin + vector![i as f32, j as f32, _k as f32].component_mul(&step))
                        .push(0.0);
                    points.push(pt);
                }
            }
        }

        #[cfg(feature = "dim2")]
        type GpuPoint = na::Point2<f32>;
        #[cfg(feature = "dim3")]
        type GpuPoint = na::Point4<f32>;

        let usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let cuboids = vec![cuboid; (LEN * LEN * LEN) as usize];
        let gpu_cuboid = GpuVector::encase(gpu.device(), &cuboids, usages);
        let gpu_points = GpuVector::init(gpu.device(), &points, usages);
        let gpu_projs: GpuVector<GpuPoint> =
            GpuVector::uninit(gpu.device(), points.len() as u32, usages);
        let gpu_projs_on_boundary: GpuVector<GpuProjectionResult> =
            GpuVector::uninit(gpu.device(), points.len() as u32, usages);

        let usages = BufferUsages::MAP_READ | BufferUsages::COPY_DST;
        let staging_projs: GpuVector<GpuPoint> =
            GpuVector::uninit(gpu.device(), points.len() as u32, usages);
        let staging_projs_on_boundary: GpuVector<GpuProjectionResult> =
            GpuVector::uninit(gpu.device(), points.len() as u32, usages);

        KernelInvocationBuilder::new(&mut queue, &wg_cuboid)
            .bind0([
                gpu_cuboid.buffer(),
                gpu_points.buffer(),
                gpu_projs.buffer(),
                gpu_projs_on_boundary.buffer(),
            ])
            .queue(points.len() as u32);

        queue.encode(&mut encoder, None);
        staging_projs.copy_from(&mut encoder, &gpu_projs);
        staging_projs_on_boundary.copy_from(&mut encoder, &gpu_projs_on_boundary);
        gpu.queue().submit(Some(encoder.finish()));

        let result_projs = staging_projs.read(gpu.device()).await.unwrap();
        let result_projs_on_boundary = staging_projs_on_boundary.read(gpu.device()).await.unwrap();

        #[cfg(feature = "dim2")]
        for (i, pt) in points.iter().enumerate() {
            let cuboid = Cuboid::new(cuboid.half_extents);
            let proj = cuboid.project_local_point(&(*pt).into(), true);
            assert_eq!(proj.point, result_projs[i]);

            let proj = cuboid.project_local_point(&(*pt).into(), false);
            assert_eq!(proj.is_inside, result_projs_on_boundary[i].is_inside != 0);
            assert_eq!(proj.point, result_projs_on_boundary[i].point);
        }

        #[cfg(feature = "dim3")]
        for (i, pt) in points.iter().enumerate() {
            let cuboid = Cuboid::new(cuboid.half_extents);
            let proj = cuboid.project_local_point(&pt.xyz().into(), true);
            assert_eq!(proj.point, result_projs[i].xyz());

            let proj = cuboid.project_local_point(&pt.xyz().into(), false);
            assert_eq!(proj.is_inside, result_projs_on_boundary[i].is_inside != 0);
            assert_eq!(proj.point, result_projs_on_boundary[i].point);
        }
    }
}