//! The segment shape.

use crate::projection::WgProjection;
use crate::ray::WgRay;
use crate::{dim_shader_defs, substitute_aliases};
use wgcore::Shader;
use wgebra::{WgSim2, WgSim3};

#[derive(Shader)]
#[shader(
    derive(WgSim3, WgSim2, WgRay, WgProjection),
    src = "segment.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Shader defining the segment shape as well as its ray-casting and point-projection functions.
pub struct WgSegment;

// TODO:
// #[cfg(test)]
// mod test {
//     use super::WgSegment;
//     use parry::shape::Segment;
//     use wgcore::tensor::GpuVector;
//
//     #[futures_test::test]
//     #[serial_test::serial]
//     async fn gpu_segment() {
//         crate::projection::test_utils::test_point_projection::<WgSegment, _>(
//             "Segment",
//             Segment::new(1.0, 0.5),
//             |device, shapes, usages| GpuVector::encase(device, shapes, usages).into_inner(),
//         )
//         .await;
//     }
// }
