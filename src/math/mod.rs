//! The math module is a collection of machine learning mathematical functionality

pub use matrix::Matrix;
pub use matrix::MatrixMapFunc;
pub use matrix::MatrixMapFunc1Arg;
pub use unsafe_sync::UnsafeSync;

mod matrix;
mod unsafe_sync;