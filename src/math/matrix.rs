use std::fmt;
use std::ops::Mul;
use std::sync::atomic::{AtomicPtr, Ordering};

use rand::Rng;
use rayon;
use simdeez::avx2::*;
use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;
use simdeez::*;

// TODO how to make this more dynamic or customizable?
const ELEMENT_WISE_MUL_MT_THRESHOLD: usize = 10_000;
const ELEMENT_WISE_MUL_MT_THREADS: usize = 4;

macro_rules! cell_id {
    ($row:expr, $col:expr, $_self:expr) => {
        $col + $row * $_self.columns
    };
}

macro_rules! transposed_cell_id {
    ($row:expr, $col:expr, $_self:expr) => {
        $row + $col * $_self.rows
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
/// use grape_ml::math::Matrix;
/// let m1 = Matrix::new(1, 2);
/// ```
///
/// Create a 1x2 matrix and filling with data
/// ```
/// // [1.0, 2.0]
/// // [3.0, 4.0]
/// use grape_ml::math::Matrix;
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
    /// use grape_ml::math::Matrix;
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
    /// use grape_ml::math::Matrix;
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
    /// use grape_ml::math::Matrix;
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
    /// use grape_ml::math::Matrix;
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
    /// use grape_ml::math::Matrix;
    /// let m1 = Matrix::new(2, 3);
    /// assert_eq!(m1.len(), 6);
    /// ```
    pub fn len(&self) -> usize {
        self.rows * self.columns
    }

    /// Returns the rows of the matrix
    ///
    /// ```
    /// use grape_ml::math::Matrix;
    /// let m1 = Matrix::new(2, 3);
    /// assert_eq!(m1.rows(), 2);
    /// ```
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the columns of the matrix
    ///
    /// ```
    /// use grape_ml::math::Matrix;
    /// let m1 = Matrix::new(2, 3);
    /// assert_eq!(m1.columns(), 3);
    /// ```
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Fill the entire matrix with the passed value
    ///
    /// ```
    /// use grape_ml::math::Matrix;
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
    /// use grape_ml::math::Matrix;
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
    /// use grape_ml::math::{Matrix, MatrixMapFunc};
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
    /// use grape_ml::math::{Matrix, MatrixMapFunc1Arg};
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
    /// use grape_ml::math::{Matrix, MatrixMapFunc};
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
    /// use grape_ml::math::{Matrix, MatrixMapFunc};
    ///
    /// let m = Matrix::new_with(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(m.get(1, 0), 3.0);
    /// ```
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[cell_id!(row, col, self)]
    }

    /// Returns a new transposed `Matrix`.
    ///
    /// ```
    /// use grape_ml::math::Matrix;
    ///
    /// // Example 1
    /// let t = Matrix::new_with(2, 1, vec![0.0, 1.0]);
    ///
    /// assert_eq!(t.transposed(), Matrix::new_with(1, 2, vec![0.0, 1.0]));
    ///
    /// // Exampple 2
    /// let t = Matrix::new_with(3, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    ///
    /// assert_eq!(t.transposed(), Matrix::new_with(3, 3, vec![0.0, 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0, 8.0]));
    ///
    /// // Exampple 3
    /// let t1 = Matrix::new_with(3, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    /// let t2 = t1.clone();
    ///
    /// assert_eq!(t1.transposed().transposed(), t2);
    /// ```
    pub fn transposed(&self) -> Self {
        let mut m = self.clone();
        m.transpose();
        m
    }

    /// Transpose the `Matrix`.
    ///
    /// ```
    /// use grape_ml::math::Matrix;
    ///
    /// // Example 1
    /// let mut t = Matrix::new_with(2, 1, vec![0.0, 1.0]);
    /// t.transpose();
    ///
    /// assert_eq!(t, Matrix::new_with(1, 2, vec![0.0, 1.0]));
    ///
    /// // Exampple 2
    /// let mut t = Matrix::new_with(3, 3, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    /// t.transpose();
    ///
    /// assert_eq!(t, Matrix::new_with(3, 3, vec![0.0, 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0, 8.0]));
    /// ```
    pub fn transpose(&mut self) {
        let mut data = Vec::with_capacity(self.rows * self.columns);
        unsafe {
            data.set_len(self.rows * self.columns); // Avoid reallocation
        }

        for r in 0..self.rows {
            for c in 0..self.columns {
                data[transposed_cell_id!(r, c, self)] =
                    self.data[cell_id!(r, c, self)];
            }
        }

        std::mem::swap(&mut self.rows, &mut self.columns);
        self.data = data;
    }

    /// Perform the summation of the matrix.
    ///
    /// ```
    /// use grape_ml::math::Matrix;
    /// let m = Matrix::new_with(2, 2, vec![1.0, 2.0, 6.0, 2.0]);
    ///
    /// assert_eq!(m.summation(), 11.0);
    /// ```
    pub fn summation(&self) -> f32 {
        self.data.iter().fold(0.0, |acc, v| acc + v)
    }

    /// Performs the summation of the exponential value of each element.
    ///
    /// ```
    /// use grape_ml::math::Matrix;
    /// let m = Matrix::new_with(2, 2, vec![1.0, 2.0, 6.0, 2.0]);
    ///
    /// assert_eq!(m.summation_exp().floor(), 420.0);
    /// ```
    pub fn summation_exp(&self) -> f32 {
        self.data.iter().fold(0.0, |acc, v| acc + v.exp())
    }

    /// Perfom an element wise multiplication. This function is optimized
    /// for large sized `Matrix`.
    ///
    /// Is required that both the `Matrix` have the same sized.
    /// ```
    /// use grape_ml::math::Matrix;
    ///
    /// // Element wise multiplication of small sized Matrix.
    ///
    /// let mut m1 = Matrix::new(10, 10);
    /// m1.fill_with(2.0);
    ///
    /// let mut m2 = Matrix::new(10, 10);
    /// m2.fill_with(3.0);
    ///
    /// let res = m1.element_wise_mul(m2);
    ///
    /// for r in 0..res.rows(){
    ///     for c in 0..res.columns(){
    ///         assert_eq!(6.0, res.get(r, c));
    ///     }
    /// }
    /// ```
    pub fn element_wise_mul(mut self, right_matrix: Matrix) -> Matrix {
        self.element_wise_mul_with(&right_matrix);
        self
    }

    /// Perfom an element wise multiplication. This function is optimized
    /// for large sized `Matrix`.
    ///
    /// Is required that both the `Matrix` have the same sized.
    ///
    /// ```
    /// use grape_ml::math::Matrix;
    ///
    /// // Element wise multiplication with large sized table.
    ///
    /// let mut m1 = Matrix::new(20_333, 2);
    /// m1.fill_with(2.0);
    ///
    /// let mut m2 = Matrix::new(20_333, 2);
    /// m2.fill_with(3.0);
    ///
    /// m1.element_wise_mul_with(&mut m2);
    ///
    /// for r in 0..m1.rows(){
    ///     for c in 0..m1.columns(){
    ///         assert_eq!(6.0, m1.get(r, c));
    ///     }
    /// }
    /// ```
    pub fn element_wise_mul_with(&mut self, right_matrix: &Matrix) {
        if self.columns != right_matrix.columns {
            println!("Can't perform an element wise multiplication on two Matrix with different columns count. {} x {}", self, right_matrix);
            return;
        }

        if self.rows != right_matrix.rows {
            println!("Can't perform an element wise multiplication on two Matrix with different rows count. {} x {}", self, right_matrix);
            return;
        }

        // It's not required explicitly writes SIMD operations since this code is
        // so trivial that is sure that the compiler is able to vectorize it.
        // However an optimization can be done, by using many threads for this
        // operation.
        if self.len() > ELEMENT_WISE_MUL_MT_THRESHOLD {
            rayon::scope(|scope| {
                // Calculates the slot size per each thread
                let slot_size = self.len() / ELEMENT_WISE_MUL_MT_THREADS;
                let spare_slot_size =
                    self.len() - (slot_size * ELEMENT_WISE_MUL_MT_THREADS);

                for t in 0..ELEMENT_WISE_MUL_MT_THREADS {
                    // It's safe to use AtomicPtr to pass pointer to the threads
                    // because this algorithm is doesn't suffer of data race and
                    // all the spawn threads end their work within this function.
                    let res_ptr = AtomicPtr::new(self);
                    scope.spawn(move |_| {
                        unsafe {
                            let res = &mut *res_ptr.load(Ordering::Relaxed);
                            let first_element = t * slot_size;
                            let last_element = {
                                let mut le = first_element + slot_size;
                                if t == (ELEMENT_WISE_MUL_MT_THREADS - 1) {
                                    // On the last thread add the spare slot
                                    le += spare_slot_size;
                                }
                                le
                            };

                            for i in first_element..last_element {
                                res.data[i] *= right_matrix.data[i];
                            }
                        }
                    });
                }
            });
        } else {
            for i in 0..self.len() {
                self.data[i] *= right_matrix.data[i];
            }
        }
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
    /// This version add a name, and it is useful to differentiate the matrix.
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

    /// Internal function used only for testing purposes.
    ///
    /// __Please don't use this.__
    pub fn i_slow_not_optimized_mul(self, other: Matrix) -> Matrix {
        if self.columns != other.rows {
            println!(
                "This matrix multiplication can't be performed: {} x {}",
                self, other
            );
            return Matrix::new(0, 0);
        }

        let mut res = Matrix::new(self.rows, other.columns);

        for other_c in 0..other.columns {
            for r in 0..self.rows {
                let mut sum = 0.0;
                for c in 0..self.columns {
                    sum += self.get(r, c) * other.get(c, other_c);
                }
                res.set(r, other_c, sum);
            }
        }

        res
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

/// The multiplication function is using an optimized algorithm that allows to
/// use many thread and SIMD operations to compute the result.
///
/// ```
/// use grape_ml::math::{Matrix, MatrixMapFunc};
///
/// let mut m_a = Matrix::new(100, 100);
/// let mut m_b = Matrix::new(100, 100);
/// m_a.fill_rand(0.0, 2.0);
/// m_b.fill_rand(0.0, 2.0);
///
/// let mut correct_res = m_a.clone().i_slow_not_optimized_mul(m_b.clone());
/// let mut may_be_wrong_res = m_a * m_b;
///
/// // Flooring the result just to avoid floating point precision loss.
/// correct_res.map(|v| v.floor());
/// may_be_wrong_res.map(|v| v.floor());
///
/// assert_eq!(correct_res, may_be_wrong_res);
/// ```
impl Mul for Matrix {
    type Output = Matrix;

    fn mul(self, mut right_matrix: Matrix) -> Matrix {
        // Performs the multiplication between two `Matrix` using SIMD operations.
        //
        // This algorithm doen't use the "mathematical" way of doing it: A_rows * B_columns.
        //
        // Rather it:
        // - Creates the result `Matrix` with all cells set to 0.
        // - For each element in the Matrix A perform a multiplication against the elements of the same
        // rows in the Matrix B, advancing per each column in the Matrix A a row in the Matrix B.
        // - Sum the result directly to the result `Matrix`.
        //
        // In this way it is possible to use SIMD APIs and benefit from the CPU cache.
        //
        // TODO try to find an even better algorithm to do the multiplication!

        let mut left_matrix = self;

        if left_matrix.columns != right_matrix.rows {
            println!(
                "This matrix multiplication can't be performed: {} x {}",
                left_matrix, right_matrix
            );
            return Matrix::new(0, 0);
        }

        let mut result_matrix = Matrix::new_with(
            left_matrix.rows,
            right_matrix.columns,
            vec![0.0; left_matrix.rows * right_matrix.columns],
        );

        let left_matrix_rows = left_matrix.rows;

        // To share the matrices accross the thread I've to use the type `Arc`
        // but since it will be shared accross some threads spawned and ended before
        // the end of this function I don't need all the power provided by `Arc`
        // and the unsafe `AtomicPtr` is much better.
        let left_matrix_ptr = AtomicPtr::new(&mut left_matrix);
        let right_matrix_ptr = AtomicPtr::new(&mut right_matrix);
        // Even if this is mutated, it's safe use an AtomicPtr because the algorithm
        // doesn't suffer of data race.
        let res_ptr = AtomicPtr::new(&mut result_matrix);

        // For each row of the right `Matrix` spawn a thread.
        rayon::scope(move |scope| {
            for r in 0usize..left_matrix_rows {
                unsafe {
                    let lm = &*left_matrix_ptr.load(Ordering::Relaxed);
                    let rm = &*right_matrix_ptr.load(Ordering::Relaxed);
                    let res = &mut *res_ptr.load(Ordering::Relaxed);
                    // TODO spawn a new thread per each row is wrong and not performant.
                    // Please spawn a small quantity of threads that will do the entire job
                    // instead to spawn a thread per each row.
                    scope.spawn(move |_| {
                        internal_multi_row_runtime_select(lm, rm, r, res)
                    });
                }
            }
        });

        result_matrix
    }
}

// I'm using the most significant bit to make it cross architecture; since AVX2 stores
// when the most significant bit is used while all the others doesn't store when 0 is set.
const ON: i32 = 1_i32 << 31;
const OFF: i32 = 0;

#[rustfmt::skip]
simd_runtime_generate!(
    fn internal_multi_row(left_matrix: &Matrix, right_matrix: &Matrix, left_m_row: usize, res: &mut Matrix) {
        for left_m_column in 0usize..left_matrix.columns {

            let left_val = left_matrix.data[cell_id!(left_m_row, left_m_column, left_matrix)];
            let simd_left_value = S::set1_ps(left_val);

            for right_m_column in (0..right_matrix.columns).step_by(S::VF32_WIDTH) {
                let res_cell_id = cell_id!(left_m_row, right_m_column, res);

                let simd_current_value = S::loadu_ps(&res.data[res_cell_id]);
                let simd_right_matrix = S::loadu_ps(
                    &right_matrix.data[cell_id!(left_m_column, right_m_column, right_matrix)],
                );

                let simd_mul_res = S::mul_ps(simd_left_value, simd_right_matrix);
                let simd_res = S::add_ps(simd_current_value, simd_mul_res);

                // Depending on the Matrix size is necessary to not overflow during the storing.
                // TODO How to improve this?
                let used_elements = right_matrix.columns - right_m_column;
                let mut mask_array = [ON; 8];
                mask_array
                    .iter_mut()
                    .skip(used_elements)
                    .for_each(|v| *v = OFF);
                let mask = S::load_epi32(&mask_array[0]);

                S::maskstore_ps(&mut res.data[res_cell_id], mask, simd_res);
            }
        }
    }
);
