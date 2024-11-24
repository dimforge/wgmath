//! Rigid-body definition and set.

use encase::ShaderType;
use wgcore::tensor::GpuVector;
use wgcore::Shader;
use wgebra::{WgQuat, WgSim2, WgSim3};
use wgparry::cuboid::GpuCuboid;
use wgparry::math::{AngVector, AngularInertia, GpuSim, Vector};
use wgparry::{dim_shader_defs, substitute_aliases};
use wgpu::{BufferUsages, Device};

#[derive(ShaderType, Copy, Clone, PartialEq)]
#[repr(C)]
/// Linear and angular forces with a layout compatible with the corresponding WGSL struct.
pub struct GpuForce {
    /// The linear part of the force.
    pub linear: Vector<f32>,
    /// The angular part of the force (aka. the torque).
    pub angular: AngVector<f32>,
}

#[derive(ShaderType, Copy, Clone, PartialEq, Default)]
#[repr(C)]
/// Linear and angular velocities with a layout compatible with the corresponding WGSL struct.
pub struct GpuVelocity {
    /// The linear (translational) velocity.
    pub linear: Vector<f32>,
    /// The angular (rotational) velocity.
    pub angular: AngVector<f32>,
}

#[derive(ShaderType, Copy, Clone, PartialEq)]
#[repr(C)]
/// Rigid-body mass-properties, with a layout compatible with the corresponding WGSL struct.
pub struct GpuMassProperties {
    /// The inverse angular inertia tensor.
    pub inv_inertia: AngularInertia<f32>,
    /// The inverse mass.
    pub inv_mass: Vector<f32>,
    /// The center-of-mass.
    pub com: Vector<f32>, // ShaderType isn’t implemented for Point
}

impl Default for GpuMassProperties {
    fn default() -> Self {
        GpuMassProperties {
            #[rustfmt::skip]
            #[cfg(feature = "dim2")]
            inv_inertia: 1.0,
            #[cfg(feature = "dim3")]
            inv_inertia: AngularInertia::identity(),
            inv_mass: Vector::repeat(1.0),
            com: Vector::zeros(),
        }
    }
}

/// A set of rigid-bodies stored on the gpu.
pub struct GpuBodySet {
    len: u32,
    pub(crate) mprops: GpuVector<GpuMassProperties>,
    pub(crate) local_mprops: GpuVector<GpuMassProperties>,
    pub(crate) vels: GpuVector<GpuVelocity>,
    pub(crate) poses: GpuVector<GpuSim>,
    // TODO: support other shape types.
    // TODO: support a shape with a shift relative to the body.
    pub(crate) shapes: GpuVector<GpuCuboid>,
}

#[derive(Copy, Clone)]
/// Helper struct for defining a rigid-body to be added to a [`GpuBodySet`].
pub struct BodyDesc {
    /// The rigid-body’s mass-properties.
    pub mprops: GpuMassProperties,
    /// The rigid-body’s linear and angular velocities.
    pub vel: GpuVelocity,
    /// The rigid-body’s world-space pose.
    pub pose: GpuSim,
    /// The rigid-body’s shape.
    pub shape: GpuCuboid,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            mprops: Default::default(),
            vel: Default::default(),
            pose: Default::default(),
            shape: GpuCuboid::new(Vector::repeat(0.5)),
        }
    }
}

impl GpuBodySet {
    /// Is this set empty?
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of rigid-bodies in this set.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Create a set of `bodies` on the gpu.
    pub fn new(device: &Device, bodies: &[BodyDesc]) -> Self {
        let (mprops, (vels, (poses, shapes))): (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))) = bodies
            .iter()
            .copied()
            // NOTE: Looks silly, but we can’t just collect into (Vec, Vec, Vec).
            .map(|b| (b.mprops, (b.vel, (b.pose, b.shape))))
            .collect();
        Self {
            len: bodies.len() as u32,
            mprops: GpuVector::encase(device, &mprops, BufferUsages::STORAGE),
            local_mprops: GpuVector::encase(device, &mprops, BufferUsages::STORAGE),
            vels: GpuVector::encase(device, &vels, BufferUsages::STORAGE),
            poses: GpuVector::init(device, &poses, BufferUsages::STORAGE),
            shapes: GpuVector::encase(device, &shapes, BufferUsages::STORAGE),
        }
    }

    /// GPU storage buffer containing the poses of every rigid-body.
    pub fn poses(&self) -> &GpuVector<GpuSim> {
        &self.poses
    }

    /// GPU storage buffer containing the velocities of every rigid-body.
    pub fn vels(&self) -> &GpuVector<GpuVelocity> {
        &self.vels
    }

    /// GPU storage buffer containing the world-space mass-properties of every rigid-body.
    pub fn mprops(&self) -> &GpuVector<GpuMassProperties> {
        &self.mprops
    }

    /// GPU storage buffer containing the local-space mass-properties of every rigid-body.
    pub fn local_mprops(&self) -> &GpuVector<GpuMassProperties> {
        &self.local_mprops
    }

    /// GPU storage buffer containing the shape of every rigid-body.
    pub fn shapes(&self) -> &GpuVector<GpuCuboid> {
        &self.shapes
    }
}

#[derive(Shader)]
#[shader(
    derive(WgQuat, WgSim3, WgSim2),
    src = "body.wgsl",
    src_fn = "substitute_aliases",
    shader_defs = "dim_shader_defs"
)]
/// Shader defining structs related to rigid-bodies, as well as functions to compute point velocities
/// and update world-space mass-properties.
pub struct WgBody;

// TODO: this test won’t pass due to the lack of `substitute_aliases`
//       and `dim_shader_defs` in the macro. Figure out a way to make this work.
// wgcore::test_shader_compilation!(WgBody);
