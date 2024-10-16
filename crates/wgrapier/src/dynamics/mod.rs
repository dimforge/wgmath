//! Rigid-body dynamics (forces, velocities, etc.)

pub use body::{BodyDesc, GpuBodySet, GpuForce, GpuMassProperties, GpuVelocity, WgBody};
pub use integrate::WgIntegrate;

pub mod body;
pub mod integrate;
