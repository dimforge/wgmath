use wgcore::Shader;

// NOTE: interesting perf. comparison between quaternions and matrices:
//       https://tech.metail.com/performance-quaternions-gpu/

#[derive(Shader)]
#[shader(src = "quat.wgsl")]
/// Shader exposing a quaternion type and operations for representing 3D rotations.
pub struct WgQuat;

wgcore::test_shader_compilation!(WgQuat);
