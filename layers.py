import numpy as np
from loss import CCE


# Unit 8: forward_pass.py
class Linear:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

        print(self.weights)

    def forward(self, inputs):
        self.inputs = inputs
        # Column of the weight matrix match the number of features in input (row)
        self.output = np.dot(inputs, self.weights) + self.bias

    def backward(self, dvalues):

        # Gradients parameters.
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient value.
        self.dinputs = np.dot(dvalues, self.weights.T)


# Unit 9: activation_function_relu.py
class Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Make a copy to not modify the targeted variable.
        self.dinputs = dvalues.copy()

        # Zero where the gradiant is negative.
        self.dinputs[self.inputs <= 0] = 0


# Unit 10: activation_function_softmax.py
class Softmax:
    def forward(self, inputs):

        # By subtracting the maximum value in each set of inputs (for each data point if you're working with batches),
        # you ensure that the largest number in the exponentiation operation is 0 (exp(0) = 1).
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Gradiant are computed for each sample that's why we need to iterate through.
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class SoftmaxCCE:
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Softmax()
        self.loss = CCE()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient, to make it invariant to the number of sample.
        self.dinputs = self.dinputs / samples
