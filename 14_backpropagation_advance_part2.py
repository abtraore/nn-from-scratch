import numpy as np

# Example gradient from next layer.
dvalues = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

# Example inputs.
inputs = np.array([[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2], [-1.5, 2.7, 3.3, -0.8]])

# Example biases.
biases = np.array([[2, 3, 0.5]])

# Example weights.
weights = np.array(
    [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
).T

print(f"{dvalues=}\n")
print(f"{weights=}\n")
print(f"{weights.T=}\n")

# dinput computation.
dinputs = np.dot(dvalues, weights.T)
print(f"{dinputs=}\n")

# dweights computation.
dweights = np.dot(inputs.T, dvalues)
print(f"{dweights=}\n")

# dbiases computation.
dbiases = np.sum(dvalues, axis=0, keepdims=True)
print(f"{dbiases=}\n")

# New dvalue example.
dvalues = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Example layer output.
z = np.array([[1, 2, -3, -4], [2, -7, -1, 3], [-1, 2, 5, -1]])

# ReLU activation's derivative (original).
drelu = np.zeros_like(z)
drelu[z > 0] = 1
drelu *= dvalues

# ReLU activation's derivative (simplified).
drelu = dvalues.copy()
drelu[z <= 0] = 0

print(f"{drelu=}\n")
