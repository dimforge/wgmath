use naga_oil::compose::ShaderDefValue;
use std::collections::HashMap;
use wgcore::Shader;

#[derive(Shader)]
#[shader(src = "shape.wgsl")]
/// A shader for handling matrix/vector indexing based on their shape of type
/// [`wgcore::shapes::ViewShape`].
pub struct Shape;

pub fn row_major_shader_defs() -> HashMap<String, ShaderDefValue> {
    [("ROW_MAJOR".to_string(), ShaderDefValue::Bool(true))].into()
}
