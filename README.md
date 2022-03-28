# Multilayer Perceptron

Creating MLP neural network from scratch. Using iris dataset.

Created MLP neural network with input, one hidden, output layer.  
Optimizer coded: mini-batch GD, Batch GD, Adam. Mini-batch and Batch can be used with momentum.  
Activation function: tahn, sigmoid, relu. Tahn is computation expensive training is much slower then using sigmoid or relu.
ReLu was showed to be much faster and optimizers can converge.
Best model why hyper parameters:
- neurons hidden layer: 20
- epochs: 300
- learning rate: 0.01
- batch size: 64
- hidden layer activation function: relu    
- output layer activation function: sigmoid
- weight distribution: Uniform 
- momentum: true
