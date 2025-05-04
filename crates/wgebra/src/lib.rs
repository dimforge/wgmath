#![doc = include_str!("../README.md")]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::result_large_err)]
#![warn(missing_docs)]

pub use geometry::*;
pub use linalg::*;

pub mod geometry;
pub mod linalg;
pub mod utils;
