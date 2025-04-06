//! Fundamental linear-algebra matrix/vector operations.

mod gemm;
mod gemv;
mod op_assign;
mod reduce;
mod shape;

pub use gemm::{Gemm, GemmVariant};
pub use gemv::{Gemv, GemvVariant};
pub use op_assign::{OpAssign, OpAssignVariant};
pub use reduce::{Reduce, ReduceOp};
pub use shape::{row_major_shader_defs, Shape};
