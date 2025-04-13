//! Utilities to address some platform-dependent differences
//! (e.g. for some trigonometric functions).

pub use self::min_max::WgMinMax;
pub use self::trig::WgTrig;

mod min_max;
mod trig;
