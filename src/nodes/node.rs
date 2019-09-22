use mathru::algebra::linear::Matrix;

pub trait Node {
    fn node_type(&self) -> NodeType;
    fn process(&mut self, input: &Matrix<f32>);
    fn output(&self) -> &Matrix<f32>;
    // TODO down cast function?
}

pub enum NodeType {
    FullyConnectedNN,
    PartiallyConnectedNN,
}
