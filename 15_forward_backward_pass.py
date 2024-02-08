import numpy as np

# Example gradient from next layer.
dvalues = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])


# Example inputs.
inputs = np.array([[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2], [-1.5, 2.7, 3.3, -0.8]])

# Example weights.
weights = np.array(
    [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
).T

# Example biases.
biases = np.array([[2, 3, 0.5]])


# Forward pass.
layer_out = np.dot(inputs, weights) + biases  # Linear output.
relu_out = np.maximum(0, layer_out)  # relu output

# Backward Pass.
drelu = relu_out.copy()
drelu[layer_out <= 0] = 0

dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, drelu)
dbiases = np.sum(drelu, axis=0, keepdims=True)

# Update parameters.
weights += -0.001 * dweights
biases += -0.001 * dbiases

print(weights)
print(biases)
