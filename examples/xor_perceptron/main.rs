use grape_ml::nodes::FullyConnectedNNNode;

fn main() {
    let fnn = FullyConnectedNNNode::new()
        .with_input_size(2)
        .with_hidden_layers_count(1)
        .with_hidden_layer_size(0, 2)
        .with_output_size(2)
        .build();

    println!("Fully connected: {:?}", fnn);
}
