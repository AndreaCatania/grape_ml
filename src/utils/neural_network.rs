//! Neural Network utilities

/// Neural Network activation functions
#[derive(Debug)]
pub enum Activation {
    /// The identity activation function returns always the same argument value.
    Identity,
    /// Sigmoid activation function
    Sigmoid,
    /// REctified Linear Unit activation function
    Relu,
    /// Leaky REctified Linear Unit activation function
    LeakyRelu,
    /// Tanh activation function
    Tanh,
    /// Linear activation function
    Linear,
    /// Binary activation function
    Binary,
    /// Softmax activation function
    Softmax,
}
