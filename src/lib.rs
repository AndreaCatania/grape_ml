mod math;

#[cfg(test)]
mod matrix_tests {

    use super::math;

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
