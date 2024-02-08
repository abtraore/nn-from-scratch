import numpy as np

softmax_output = [0.7, 0.1, 0.2]

softmax_output = np.array(softmax_output).reshape(-1, 1)

dsoftmax = np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T)
print(dsoftmax)
