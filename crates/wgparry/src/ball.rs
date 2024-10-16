//! The ball shape.

use crate::ray::WgRay;
use wgcore::Shader;
use wgebra::WgSim3;

#[derive(Shader)]
#[shader(derive(WgSim3, WgRay), src = "ball.wgsl")]
/// Shader defining the ball shape as well as its ray-casting and point-projection functions.
pub struct WgBall;

wgcore::test_shader_compilation!(WgBall);
