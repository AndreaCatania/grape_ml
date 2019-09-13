use std::fmt;
use std::ops::Mul;

use rand::Rng;

macro_rules! cell_id {
    ($row:expr, $col:expr, $_self:expr) => {
        $col + $row * $_self.columns
    };
}

/// This type function is used to map a matrix.
/// It receive the current I value and return it mapped.
pub type MatrixMapFunc = fn(f32) -> f32;

/// This type function is used to map a matrix.
/// It receives the current I value, and the user argument value and returns
/// the mapped value.
pub type MatrixMapFunc1Arg = fn(f32, f32) -> f32;

/// The `Matrix` is a dynamic resizable matrix, which allow to perform all
/// matrix operations.
///
/// _In future it will be capable of run in the GPU_
///
/// # Example
///
/// Create an empty matrix with 1 Row and 2 columns.
/// ```
/// use grape::math::Matrix;
/// let m1 = Matrix::new(1, 2);
/// ```
///
/// Create a 1x2 matrix and filling with data
/// ```
/// // [1.0, 2.0]
/// // [3.0, 4.0]
/// use grape::math::Matrix;
/// let m = Matrix::new_with(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
/// ```
pub struct Matrix {
    rows: usize,
    columns: usize,
    data: Vec<f32>,
}

impl Matrix {
    /// Creates new matrix, with uninitialized data.
    ///
    /// ```
    /// use grape::math::Matrix;
    ///
    /// let m = Matrix::new(2, 3);
    /// assert_eq!(m.len(), 6);
    /// ```
    pub fn new(rows: usize, columns: usize) -> Matrix {
        let mut v = Vec::with_capacity(rows * columns);
        unsafe {
            // Enlarge its size right away instead to wait other operations.
            // This operation is safe because we just created the array with this capacity.
            v.set_len(rows * columns);
        }
        Matrix::new_with(rows, columns, v)
    }

    /// Creates new matrix by cloning passed data
    ///
    /// ## Note:
    /// If the data cloned size is not compatible with rows and columns the matrix
    /// is created anyway, but its internal data will be truncated or expanded
    /// depending the case, and an error is print.
    ///
    /// ```
    /// use grape::math::Matrix;
    ///
    /// let m = Matrix::new_from(2, 3, &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    /// assert_eq!(m.len(), 6);
    /// assert_eq!(m.get(0, 0), 6.0);
    /// assert_eq!(m.get(0, 1), 5.0);
    /// assert_eq!(m.get(0, 2), 4.0);
    /// assert_eq!(m.get(1, 0), 3.0);
    /// assert_eq!(m.get(1, 1), 2.0);
    /// assert_eq!(m.get(1, 2), 1.0);
    /// ```
    pub fn new_from(rows: usize, columns: usize, data: &[f32]) -> Matrix {
        Matrix::new_with(rows, columns, data.to_vec())
    }

    /// Creates new matrix with the passed data
    ///
    /// ## Note:
    /// If the passed data size is not compatible with rows and columns the matrix
    /// is created anyway, but the data will be truncated or expanded
    /// depending the case, and an error is print.
    ///
    /// ```
    /// use grape::math::Matrix;
    ///
    /// let m = Matrix::new_with(2, 3, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    /// assert_eq!(m.len(), 6);
    /// assert_eq!(m.get(0, 0), 6.0);
    /// assert_eq!(m.get(0, 1), 5.0);
    /// assert_eq!(m.get(0, 2), 4.0);
    /// assert_eq!(m.get(1, 0), 3.0);
    /// assert_eq!(m.get(1, 1), 2.0);
    /// assert_eq!(m.get(1, 2), 1.0);
    /// ```
    pub fn new_with(rows: usize, columns: usize, data: Vec<f32>) -> Matrix {
        let mut data = data;
        if data.len() != rows * columns {
            data.resize(rows * columns, 0.0);
            println!("Warning the passed data size is different from expected");
        }

        Matrix {
            rows,
            columns,
            data,
        }
    }
}

impl Matrix {
    /// Returns true when the matrix is empty
    ///
    /// ```
    /// use grape::math::Matrix;
    ///
    /// let m1 = Matrix::new(0, 0);
    /// let m2 = Matrix::new(1, 0);
    /// let m3 = Matrix::new(0, 1);
    /// let m4 = Matrix::new(1, 1);
    ///
    /// assert_eq!(m1.is_empty(), true);
    /// assert_eq!(m2.is_empty(), true);
    /// assert_eq!(m3.is_empty(), true);
    /// assert_eq!(m4.is_empty(), false);
    /// ```
    pub fn is_empty(&self) -> bool {
        self.rows * self.columns == 0
    }

    /// Returns the size of the matrix
    ///
    /// ```
    /// use grape::math::Matrix;
    /// let m1 = Matrix::new(2, 3);
    /// assert_eq!(m1.len(), 6);
    /// ```
    pub fn len(&self) -> usize {
        (self.rows * self.columns) as usize
    }

    /// Returns the rows of the matrix
    ///
    /// ```
    /// use grape::math::Matrix;
    /// let m1 = Matrix::new(2, 3);
    /// assert_eq!(m1.rows(), 2);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the columns of the matrix
    ///
    /// ```
    /// use grape::math::Matrix;
    /// let m1 = Matrix::new(2, 3);
    /// assert_eq!(m1.columns(), 3);
    /// ```
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Fill the entire matrix with the passed value
    ///
    /// ```
    /// use grape::math::Matrix;
    ///
    /// let m1 = Matrix::new_with(1, 2, vec![2.0, 2.0]);
    ///
    /// let mut m2 = Matrix::new(1, 2);
    /// m2.fill_with(2.0);
    ///
    /// assert_eq!(m1, m2);
    /// ```
    pub fn fill_with(&mut self, val: f32) {
        for v in self.data.iter_mut() {
            *v = val;
        }
    }

    /// Fills the entire matrix with random values.
    /// * `range_min` inclusive.
    /// * `range_max` exclusive.
    ///
    /// ```
    /// use grape::math::Matrix;
    ///
    /// let mut m = Matrix::new(2, 2);
    /// m.fill_rand(0.0, 1.0);
    ///
    /// assert!(m.get(0, 0) >= 0.0 && m.get(0, 0) < 1.0);
    /// assert!(m.get(0, 1) >= 0.0 && m.get(0, 1) < 1.0);
    /// assert!(m.get(1, 0) >= 0.0 && m.get(1, 0) < 1.0);
    /// assert!(m.get(1, 1) >= 0.0 && m.get(1, 1) < 1.0);
    /// ```
    pub fn fill_rand(&mut self, range_min: f32, range_max: f32) {
        let mut rng = rand::thread_rng();
        for v in self.data.iter_mut() {
            *v = rng.gen_range(range_min, range_max);
        }
    }

    /// Map all values with the passed function.
    ///
    /// You can pass a function pointer of this type: MatrixMapFunc.
    ///
    /// # Example
    ///
    /// ```
    /// use grape::math::{Matrix, MatrixMapFunc};
    ///
    /// let mut m1 = Matrix::new_with(1, 2, vec![1.0, 1.0]);
    /// m1.map(|v: f32| -> f32 {v*2.0});
    ///
    /// let m2 = Matrix::new_with(1, 2, vec![2.0, 2.0]);
    /// assert_eq!(m1, m2);
    /// ```
    pub fn map(&mut self, func: MatrixMapFunc) {
        for i in self.data.iter_mut() {
            *i = func(*i);
        }
    }

    /// Map all values with the passed function. This version accept a custom argument
    /// that can be used during the mapping.
    ///
    /// Useful when you have an helper function which accept a custom argument.
    ///
    /// # Example
    ///
    /// ```
    /// let func : MatrixMapFunc1Arg = |v: f32, e:f32| -> f32 {v.powf(e)};
    ///
    /// use grape::math::{Matrix, MatrixMapFunc1Arg};
    ///
    /// let mut m1 = Matrix::new_with(1, 2, vec![2.0, 2.0]);
    /// m1.map_arg(func, 3.0);
    ///
    /// let m2 = Matrix::new_with(1, 2, vec![8.0, 8.0]);
    /// assert_eq!(m1 == m2, true);
    /// ```
    pub fn map_arg(&mut self, func: MatrixMapFunc1Arg, arg: f32) {
        for i in self.data.iter_mut() {
            *i = func(*i, arg);
        }
    }

    /// Set the value to a specific cell
    ///
    /// ```
    /// use grape::math::{Matrix, MatrixMapFunc};
    ///
    /// let mut m = Matrix::new_with(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// m.set(1, 0, 8.0);
    /// assert_eq!(m.get(1, 0), 8.0);
    /// ```
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[cell_id!(row, col, self)] = val;
    }

    /// Get the value contained in the cell.
    ///
    /// ```
    /// use grape::math::{Matrix, MatrixMapFunc};
    ///
    /// let m = Matrix::new_with(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(m.get(1, 0), 3.0);
    /// ```
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[cell_id!(row, col, self)]
    }

    /// Returns a string with the matrix values in an human readable format.
    pub fn to_str(&self) -> String {
        let mut s = String::new();

        s += "┏━━━━┉\n┃\n";
        for x in 0..self.rows {
            s += "┃  [";
            for y in 0..self.columns {
                s += format!("{:>+5.2}, ", self.get(x, y)).as_str();
            }
            s += "]\n";
        }
        s += "┃\n┗━━━━┉\n";

        s
    }

    /// Returns a string with the matrix values in an human readable format.
    ///
    /// This version add a name, and is useful to defferentiate the matrix.
    pub fn to_str_with_name(&self, name: &str) -> String {
        let mut s = String::new();

        s += format!("┏━━━━┉  {}  ┉━\n┃\n", name).as_str();
        for x in 0..self.rows {
            s += "┃  [";
            for y in 0..self.columns {
                s += format!("{:>+5.2}, ", self.get(x, y)).as_str();
            }
            s += "]\n";
        }
        s += "┃\n┗━━━━┉\n";

        s
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Matrix {
        Matrix::new_from(self.rows, self.columns, &self.data)
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
        if self.rows != other.rows {
            return false;
        }

        if self.columns != other.columns {
            return false;
        }

        self.data == other.data
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\n{}", self.to_str())
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\n{}", self.to_str())
    }
}

// TODO is it correct compare the result matric in this way?
//
/// ```
/// use grape::math::{Matrix, MatrixMapFunc};
///
/// let res = Matrix::new_with(3, 3, vec![1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0]) * Matrix::new_with(3, 2, vec![2.0, 5.0, 6.0, 7.0, 1.0, 8.0]);
/// assert_eq!(res, Matrix::new_with(3, 2, vec![15.0, 27.0, 6.0, 7.0, 26.0, 63.0]));
///
/// let res = Matrix::new_with(2, 4, vec![1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 5.0]) * Matrix::new_with(4, 2, vec![2.0, 5.0, 6.0, 7.0, 1.0, 8.0, 3.0, 1.0]);
/// assert_eq!(res, Matrix::new_with(2, 2, vec![18.0, 28.0, 21.0, 12.0]));
///
/// let res = Matrix::new_with(2, 4, vec![1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 0.0, 5.0]) * Matrix::new_with(4, 5, vec![2.0, 5.0, 6.0, 1.0, 4.0, 6.0, 7.0, 9.0, 7.0, 3.0, 1.0, 8.0, 2.0, 5.0, 4.0, 3.0, 1.0, 6.0, 8.0, 2.0]);
/// assert_eq!(res, Matrix::new_with(2, 5, vec![18.0, 28.0, 32.0, 28.0, 16.0, 21.0, 12.0, 39.0, 47.0, 13.0]));
/// ```
impl Mul for Matrix {
    type Output = Matrix;

    fn mul(self, other: Matrix) -> Matrix {
        internal_mut_runtime_select(self, other)
    }
}

use simdeez::avx2::*;
use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;
use simdeez::*;

simd_runtime_generate!(
    fn internal_mut(left_matrix: Matrix, right_matrix: Matrix) -> Matrix {
        if left_matrix.columns != right_matrix.rows {
            println!(
                "This matrix multiplication can't be performed: {} x {}",
                left_matrix, right_matrix
            );
            return Matrix::new(0, 0);
        }

        let mut res = Matrix::new_with(
            left_matrix.rows,
            right_matrix.columns,
            vec![0.0; left_matrix.rows * right_matrix.columns],
        );

        for r in 0..left_matrix.rows {
            for c in 0..left_matrix.columns {
                let left_matrix_val = left_matrix.data[cell_id!(r, c, left_matrix)];
                let simd_left_matrix = S::set1_ps(left_matrix_val);

                for other_c in (0..right_matrix.columns).step_by(S::VF32_WIDTH) {

                    let simd_current_value = S::loadu_ps(&res.data[cell_id!(r, other_c, res)]);
                    let simd_right_matrix = S::loadu_ps(&right_matrix.data[cell_id!(c, other_c, right_matrix)]);

                    //let simd_mul_res = S::mul_ps(simd_left_matrix, simd_right_matrix);
                    //let simd_res = S::add_ps(simd_current_value, simd_mul_res);

                    //S::storeu_ps(&mut res.data[cell_id!(r, other_c, res)], simd_res);
                    //S::storeu_ps(&mut res.data[cell_id!(r, other_c, res)], simd_current_value);

                    //let simd_res = simd_current_value + (simd_left_matrix * simd_right_matrix);
                    let simd_res = simd_current_value + simd_left_matrix;
                    S::storeu_ps(&mut res.data[cell_id!(r, other_c, res)], simd_res);

                    // TODO remove this please
                    //res.data[cell_id!(r, other_c, res)] = res.data
                    //    [cell_id!(r, other_c, res)]
                    //    + left_matrix.data[cell_id!(r, c, left_matrix)]
                    //        * right_matrix.data[cell_id!(c, other_c, right_matrix)];
                }
            }
        }

        res
    }
);
