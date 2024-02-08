import numpy as np


# Unit 8: forward_pass.py
class Linear:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

        print(self.weights)

    def forward(self, inputs):

        # Column of the weight matrix match the number of features in input (row)
        self.output = np.dot(inputs, self.weights) + self.bias


# Unit 9: activation_function_relu.py
class Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Unit 10: activation_function_softmax.py
class Softmax:
    def forward(self, inputs):

        # By subtracting the maximum value in each set of inputs (for each data point if you're working with batches),
        # you ensure that the largest number in the exponentiation operation is 0 (exp(0) = 1).
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
