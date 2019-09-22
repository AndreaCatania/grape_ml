use mathru::algebra::linear::Matrix;

use crate::{
    nodes::{Node, NodeType},
    utils::neural_network::Activation,
};

#[derive(Debug)]
pub struct FullyConnectedNNBuilder {
    input_size: usize,
    hidden_layers_size: Vec<usize>,
    output_size: usize,
    weight_min_range: f32,
    weight_max_range: f32,
    activations: Vec<Activation>,
}

impl FullyConnectedNNBuilder {
    pub fn with_input_size(mut self, size: usize) -> Self {
        self.input_size = size;
        self
    }

    pub fn with_hidden_layers_count(mut self, count: usize) -> Self {
        self.hidden_layers_size.resize(count, 0usize);
        self
    }

    pub fn with_hidden_layer_size(mut self, layer: usize, size: usize) -> Self {
        self.hidden_layers_size[layer] = size;
        self
    }

    pub fn with_output_size(mut self, size: usize) -> Self {
        self.output_size = size;
        self
    }

    pub fn with_random_weights(
        mut self,
        min_range: f32,
        max_range: f32,
    ) -> Self {
        self.weight_min_range = min_range;
        self.weight_max_range = max_range;
        self
    }

    pub fn build(self) -> FullyConnectedNNNode {
        let layer_count = 2 + self.hidden_layers_size.len();

        let mut weights = Vec::with_capacity(layer_count);
        let mut biases = Vec::with_capacity(layer_count);

        for i in 0..layer_count {
            let layer_weight = Matrix::<f32>::new()
            weights.push()
        }

        FullyConnectedNNNode {
            activations: self.activations,
            weights,
            biases,
        }
    }
}

#[derive(Debug)]
pub struct FullyConnectedNNNode {
    activations: Vec<Activation>,
    weights: Vec<Matrix<f32>>,
    biases: Vec<Matrix<f32>>,
}

impl FullyConnectedNNNode {
    pub fn new() -> FullyConnectedNNBuilder {
        FullyConnectedNNBuilder {
            input_size: 0,
            hidden_layers_size: Vec::new(),
            output_size: 0,
            weight_min_range: -1.0,
            weight_max_range: 1.0,
        }
    }
}

impl Node for FullyConnectedNNNode {
    fn node_type(&self) -> NodeType {
        NodeType::FullyConnectedNN
    }

    fn process(&mut self, input: &Matrix<f32>) {
        panic!();
    }

    fn output(&self) -> &Matrix<f32> {
        panic!();
    }
}
