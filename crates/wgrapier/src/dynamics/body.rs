//! Rigid-body definition and set.

use encase::ShaderType;
use num_traits::Zero;
use rapier::geometry::{ColliderHandle, TypedShape};
use rapier::prelude::MassProperties;
use rapier::{
    dynamics::{RigidBodyHandle, RigidBodySet},
    geometry::ColliderSet,
};
use wgcore::tensor::GpuVector;
use wgcore::Shader;
use wgebra::{GpuSim3, WgQuat, WgSim2, WgSim3};
use wgparry::math::{AngVector, AngularInertia, GpuSim, Point, Vector};
use wgparry::parry::shape::Cuboid;
use wgparry::shape::{GpuShape, ShapeBuffers};
use wgparry::{dim_shader_defs, substitute_aliases};
use wgpu::BindingType::Buffer;
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

impl From<MassProperties> for GpuMassProperties {
    fn from(props: MassProperties) -> Self {
        GpuMassProperties {
            #[cfg(feature = "dim2")]
            inv_inertia: props.inv_principal_inertia_sqrt * props.inv_principal_inertia_sqrt,
            #[cfg(feature = "dim3")]
            inv_inertia: props.reconstruct_inverse_inertia_matrix(),
            inv_mass: Vector::repeat(props.inv_mass),
            com: props.local_com.coords,
        }
    }
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
    shapes_data: Vec<GpuShape>, // TODO: exists only for convenience in the MPM simulation.
    pub(crate) mprops: GpuVector<GpuMassProperties>,
    pub(crate) local_mprops: GpuVector<GpuMassProperties>,
    pub(crate) vels: GpuVector<GpuVelocity>,
    pub(crate) poses: GpuVector<GpuSim>,
    // TODO: support other shape types.
    // TODO: support a shape with a shift relative to the body.
    pub(crate) shapes: GpuVector<GpuShape>,
    // TODO: it’s a bit weird that we store the vertex buffer but not the
    //       index buffer. This is because our only use-case currently
    //       is from wgsparkl which has its own way of storing indices.
    pub(crate) shapes_local_vertex_buffers: GpuVector<Point<f32>>,
    pub(crate) shapes_vertex_buffers: GpuVector<Point<f32>>,
    pub(crate) shapes_vertex_collider_id: GpuVector<u32>, // NOTE: this is a bit of a hack for wgsparkl
}

#[derive(Copy, Clone)]
/// Helper struct for defining a rigid-body to be added to a [`GpuBodySet`].
pub struct BodyDesc {
    /// The rigid-body’s mass-properties in local-space.
    pub local_mprops: GpuMassProperties,
    /// The rigid-body’s mass-properties in world-space.
    pub mprops: GpuMassProperties,
    /// The rigid-body’s linear and angular velocities.
    pub vel: GpuVelocity,
    /// The rigid-body’s world-space pose.
    pub pose: GpuSim,
    /// The rigid-body’s shape.
    pub shape: GpuShape,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            local_mprops: Default::default(),
            mprops: Default::default(),
            vel: Default::default(),
            pose: Default::default(),
            shape: GpuShape::cuboid(Vector::repeat(0.5)),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum BodyCoupling {
    OneWay,
    #[default]
    TwoWays,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BodyCouplingEntry {
    pub body: RigidBodyHandle,
    pub collider: ColliderHandle,
    pub mode: BodyCoupling,
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

    pub fn from_rapier(
        device: &Device,
        bodies: &RigidBodySet,
        colliders: &ColliderSet,
        coupling: &[BodyCouplingEntry],
    ) -> Self {
        let mut shape_buffers = ShapeBuffers::default();
        let mut gpu_bodies = vec![];
        let mut pt_collider_ids = vec![];

        for (co_id, coupling) in coupling.iter().enumerate() {
            let co = &colliders[coupling.collider];
            let rb = &bodies[coupling.body];

            let prev_len = shape_buffers.vertices.len();
            let shape = GpuShape::from_parry(co.shape(), &mut shape_buffers)
                .expect("Unsupported shape type");
            for _ in prev_len..shape_buffers.vertices.len() {
                pt_collider_ids.push(co_id as u32);
            }

            let zero_mprops = MassProperties::zero();
            let two_ways_coupling = rb.is_dynamic() && coupling.mode == BodyCoupling::TwoWays;
            let desc = BodyDesc {
                vel: GpuVelocity {
                    linear: *rb.linvel(),
                    angular: rb.angvel().clone(),
                },
                #[cfg(feature = "dim2")]
                pose: (*rb.position()).into(),
                #[cfg(feature = "dim3")]
                pose: GpuSim3::from_isometry(*rb.position(), 1.0),
                shape,
                local_mprops: if two_ways_coupling {
                    rb.mass_properties().local_mprops.into()
                } else {
                    zero_mprops.into()
                },
                mprops: if two_ways_coupling {
                    rb.mass_properties()
                        .local_mprops
                        .transform_by(rb.position())
                        .into()
                } else {
                    zero_mprops.into()
                },
            };
            gpu_bodies.push(desc);
        }

        Self::new(device, &gpu_bodies, &pt_collider_ids, &shape_buffers)
    }

    /// Create a set of `bodies` on the gpu.
    pub fn new(
        device: &Device,
        bodies: &[BodyDesc],
        pt_collider_ids: &[u32],
        shape_buffers: &ShapeBuffers,
    ) -> Self {
        let (local_mprops, (mprops, (vels, (poses, shapes_data)))): (
            Vec<_>,
            (Vec<_>, (Vec<_>, (Vec<_>, Vec<_>))),
        ) = bodies
            .iter()
            .copied()
            // NOTE: Looks silly, but we can’t just collect into (Vec, Vec, Vec).
            .map(|b| (b.local_mprops, (b.mprops, (b.vel, (b.pose, b.shape)))))
            .collect();
        // TODO: (api design) how can we let the user pick the buffer usages?
        Self {
            len: bodies.len() as u32,
            mprops: GpuVector::encase(device, &mprops, BufferUsages::STORAGE),
            local_mprops: GpuVector::encase(device, &local_mprops, BufferUsages::STORAGE),
            vels: GpuVector::encase(
                device,
                &vels,
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            ),
            poses: GpuVector::init(
                device,
                &poses,
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            ),
            shapes: GpuVector::init(device, &shapes_data, BufferUsages::STORAGE),
            shapes_local_vertex_buffers: GpuVector::encase(
                device,
                &shape_buffers.vertices,
                BufferUsages::STORAGE,
            ),
            shapes_vertex_buffers: GpuVector::encase(
                device,
                // TODO: init in world-space directly?
                &shape_buffers.vertices,
                BufferUsages::STORAGE,
            ),
            shapes_vertex_collider_id: GpuVector::init(
                device,
                &pt_collider_ids,
                BufferUsages::STORAGE,
            ),
            shapes_data,
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
    pub fn shapes(&self) -> &GpuVector<GpuShape> {
        &self.shapes
    }

    pub fn shapes_vertex_buffers(&self) -> &GpuVector<Point<f32>> {
        &self.shapes_vertex_buffers
    }

    pub fn shapes_local_vertex_buffers(&self) -> &GpuVector<Point<f32>> {
        &self.shapes_local_vertex_buffers
    }

    pub fn shapes_vertex_collider_id(&self) -> &GpuVector<u32> {
        &self.shapes_vertex_collider_id
    }

    pub fn shapes_data(&self) -> &[GpuShape] {
        &self.shapes_data
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
