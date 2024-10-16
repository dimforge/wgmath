//! Force and velocity integration.

use crate::dynamics::body::{GpuBodySet, WgBody};
use wgcore::kernel::{KernelInvocationBuilder, KernelInvocationQueue};
use wgcore::Shader;
use wgparry::{dim_shader_defs, substitute_aliases};
use wgpu::ComputePipeline;

#[derive(Shader)]
#[shader(
    derive(WgBody),
    src = "integrate.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Shaders exposing composable functions for force and velocity integration.
pub struct WgIntegrate {
    /// Compute shader for integrating forces and velocities of every rigid-body.
    pub integrate: ComputePipeline,
}

impl WgIntegrate {
    const WORKGROUP_SIZE: u32 = 64;

    /// Queues an invocation of [`WgIntegrate::integrate`] for integrating forces and velocities
    /// of every rigid-body in the given [`GpuBodySet`]:
    pub fn queue<'a>(&'a self, queue: &mut KernelInvocationQueue<'a>, bodies: &GpuBodySet) {
        KernelInvocationBuilder::new(queue, &self.integrate)
            .bind0([
                bodies.mprops.buffer(),
                bodies.local_mprops.buffer(),
                bodies.poses.buffer(),
                bodies.vels.buffer(),
            ])
            .queue(bodies.len().div_ceil(Self::WORKGROUP_SIZE));
    }
}

wgcore::test_shader_compilation!(WgIntegrate);
