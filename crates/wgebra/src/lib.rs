#![doc = include_str!("../README.md")]
#![allow(clippy::too_many_arguments)]
// #![warn(missing_docs)]

pub use geometry::*;
pub use linalg::*;

pub mod geometry;
pub mod linalg;
pub mod utils;
