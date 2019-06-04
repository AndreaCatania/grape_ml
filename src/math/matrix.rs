use rand::Rng;

pub struct Matrix {
    rows: usize,
    columns: usize,
    data: Vec<f32>,
}

impl Clone for Matrix {
    fn clone(&self) -> Matrix {
        Matrix::new_cloning(self.rows, self.columns, &self.data)
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

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Matrix {
        Matrix::new_with(rows, columns, Vec::with_capacity(rows * columns))
    }

    pub fn new_cloning(rows: usize, columns: usize, data: &Vec<f32>) -> Matrix {
        Matrix::new_with(rows, columns, data.clone())
    }

    pub fn new_with(rows: usize, columns: usize, data: Vec<f32>) -> Matrix {
        let mut data = data;
        if data.len() != rows * columns {
            data.resize(rows * columns, 0.0);
        }

        Matrix {
            rows,
            columns,
            data,
        }
    }
}

impl Matrix {
    pub fn len(&self) -> usize {
        (self.rows * self.columns) as usize
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn data(&self) -> &Vec<f32> {
        &self.data()
    }

    pub fn fill_with(&mut self, val: f32) {
        self.data.iter_mut().map(|x| *x = val).count();
    }

    pub fn fill_rand(&mut self, range_min: f32, range_max: f32) {
        let mut rng = rand::thread_rng();
        // gen_range is exclusive on max side. (This is not necessary but has 0 cost)
        let range_max = range_max + 0.01;
        self.data.iter_mut().map(|x| *x = rng.gen_range(range_min, range_max)).count();
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[col * self.rows + row]
    }

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

// Private
impl Matrix {
    fn init(&mut self) -> &mut Matrix {
        self.data.resize(self.len(), 0.0);
        self
    }
}
