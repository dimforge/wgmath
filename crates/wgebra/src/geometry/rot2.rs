use wgcore::Shader;

#[derive(Shader)]
#[shader(src = "rot2.wgsl")]
/// Shader exposing a 2D rotation type and operations.
pub struct WgRot2;

wgcore::test_shader_compilation!(WgRot2);
