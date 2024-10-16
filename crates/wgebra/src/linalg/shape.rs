use wgcore::Shader;

#[derive(Shader)]
#[shader(src = "shape.wgsl")]
/// A shader for handling matrix/vector indexing based on their shape of type
/// [`wgcore::shapes::ViewShape`].
pub struct Shape;
