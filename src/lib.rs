//! # Grape Machine learning
//! Grape Machine learning is a framework for data analisys that allow to combine
//! different methods in a pipeline.
//!
//! Since the human brain is composed by many areas, with different structure,
//! that receive different inputs from different source, the graph pipeline try
//! to mimic this behaviour by combining nodes and taking the input from many sources.
//!
//! A node maybe a fully connected neural network, or a CNN, or a simple pre process
//! algorithm that optimize the data for the next node. A node may run in parallel
//! with other nodes.
//! Each node takes the input from the parent node and perform some operations;
//! then the output is given to the the child nodes.

#![warn(
    missing_debug_implementations,
    missing_docs,
    rust_2018_idioms,
    rust_2018_compatibility,
    clippy::all
)]

pub mod nodes;
pub mod utils;
