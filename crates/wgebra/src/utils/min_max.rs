use wgcore::Shader;

/// Helper shader functions for calculating the min/max elements of a vector or matrix.
#[derive(Shader)]
#[shader(src = "min_max.wgsl")]
pub struct WgMinMax;
