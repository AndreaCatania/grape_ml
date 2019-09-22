//! This module contains all the Nodes already implemented in Grape.

pub use fully_connected_nn::FullyConnectedNNNode;
pub use node::{Node, NodeType};
pub use partially_connected_nn::PartiallyConnectedNNNode;

mod fully_connected_nn;
mod node;
mod partially_connected_nn;
