//! Geometric transformations.

pub use cholesky::*;
pub use eig2::*;
pub use eig3::*;
pub use eig4::*;
pub use inv::*;
pub use lu::*;
pub use qr2::*;
pub use qr3::*;
pub use qr4::*;
pub use quat::*;
pub use rot2::*;
pub use sim2::*;
pub use sim3::*;
pub use svd2::*;
pub use svd3::*;

mod cholesky;
mod eig2;
mod eig3;
mod eig4;
mod inv;
mod lu;
mod qr2;
mod qr3;
mod qr4;
mod quat;
mod rot2;
mod sim2;
mod sim3;
mod svd2;
mod svd3;
