use mathru::algebra::linear::Matrix;

use crate::nodes::{Node, NodeType};

pub struct PartiallyConnectedNNNode {}

impl Node for PartiallyConnectedNNNode {
    fn node_type(&self) -> NodeType {
        NodeType::PartiallyConnectedNN
    }

    fn process(&mut self, input: &Matrix<f32>) {
        panic!();
    }

    fn output(&self) -> &Matrix<f32> {
        panic!();
    }
}
