use wgcore::Shader;

/// Alternative implementations of some geometric functions on the gpu.
///
/// Some platforms (Metal in particular) has implementations of some trigonometric functions
/// that are not numerically stable. This is the case for example for `atan2` and `atanh` that
/// may occasionally lead to NaNs. This shader exposes alternative implementations for numerically
/// stable versions of these functions to ensure good behavior across all platforms.
#[derive(Shader)]
#[shader(src = "trig.wgsl")]
pub struct WgTrig;
