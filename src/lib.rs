//! # Grape Machine learning
//! Grape Machine learning is a library written in Rust.
//!
//! This library provides many neural networks implementations, that can be
//! used depending the problem to solve.
//!
//! These implementations shares the same interfaces and can be wired each
//! other in order to create a more complex brain that is able to solve more
//! complicated tasks.

#![warn(
    missing_debug_implementations,
    missing_docs,
    rust_2018_idioms,
    rust_2018_compatibility,
    clippy::all
)]

pub mod math;

#[cfg(test)]
mod matrix_tests {

    use super::math;

    #[test]
    fn check_map() {
        const MUL_2: math::MatrixMapFunc = |v: f32| -> f32 { v * 2.0 };
        const _POW: math::MatrixMapFunc1Arg = |v: f32, e: f32| -> f32 { v.powf(e) };

        let mut m1 = math::Matrix::new_with(1, 2, vec![1.0, 1.0]);

        m1.map(MUL_2);
        m1.map_arg(_POW, 3.0);

        let m2 = math::Matrix::new_with(1, 2, vec![8.0, 8.0]);
        assert_eq!(m1 == m2, true);
    }

    #[test]
    fn check_equal() {
        let m1 = math::Matrix::new_with(1, 2, vec![1.0, 2.0]);
        let m2 = math::Matrix::new_with(1, 2, vec![1.0, 2.0]);
        assert_eq!(m1 == m2, true);
    }

    #[test]
    fn check_copy_clone() {
        let m1 = math::Matrix::new_with(1, 2, vec![1.0, 2.0, 3.0]);
        let m2 = m1.clone();
        let mut m3 = math::Matrix::new(0, 0);
        m3.clone_from(&m2);
        let m3 = m3;

        assert_eq!(m1 == m2, true);
        assert_eq!(m1 == m3, true);
    }

    #[test]
    fn check_size() {
        let m = math::Matrix::new(1, 2);
        assert_eq!(m.len(), 2);
    }
}
