//! Fundamental linear-algebra matrix/vector operations.

mod gemv;
mod op_assign;
mod reduce;
mod shape;

pub use gemv::Gemv;
pub use op_assign::{OpAssign, OpAssignVariant};
pub use reduce::{Reduce, ReduceOp};
pub use shape::Shape;
