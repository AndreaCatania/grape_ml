use grape::math::Matrix;

fn main() {
    let matrix_size = 2048;

    let mut m1 = Matrix::new(matrix_size, matrix_size);
    m1.fill_rand(0.0, 2.0);

    let mut m2 = Matrix::new(matrix_size, matrix_size);
    m2.fill_rand(0.0, 2.0);

    let _r = m1.element_wise_mul(m2);

    //assert!(r.len() > 0);
}
