import numpy as np

inputs = np.array([[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])

weights = np.array(
    [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
)

biases = [2, 3, 0.5]

print(f"Inputs=\n{inputs}\n")
print(f"Weights=\n{weights.T}\n")
print(f"Baises=\n{biases}\n")

outputs = np.dot(inputs, weights.T)


print(f"Outputs=\n{outputs}")
