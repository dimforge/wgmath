use wgcore::Shader;

#[derive(Shader)]
#[shader(src = "inv.wgsl")]
/// Shader exposing small matrix inverses.
pub struct WgInv;

wgcore::test_shader_compilation!(WgInv);
