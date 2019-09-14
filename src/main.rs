use grape::math::Matrix;

fn main() {
    let m1 = Matrix::new_with(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let m2 = Matrix::new_with(2, 3, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);

    let res = m1 * m2;
    dbg!(res);
}
