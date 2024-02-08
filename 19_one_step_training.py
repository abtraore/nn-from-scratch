import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from layers import Linear, Relu, Softmax, SoftmaxCCE
from loss import CCE

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

l1 = Linear(2, 3)
l2 = Linear(3, 3)

relu1 = Relu()

softmax = Softmax()
loss_fn = SoftmaxCCE()

l1.forward(X)
relu1.forward(l1.output)
l2.forward(relu1.output)

loss = loss_fn.forward(l2.output, y)

predictions = np.argmax(loss_fn.output, axis=1)

accuracy = np.mean(predictions == y)

print(f"{loss=}")
print(f"{accuracy=}")

# Backward pass
loss_fn.backward(loss_fn.output, y)
l2.backward(loss_fn.dinputs)
relu1.backward(l2.dinputs)
l1.backward(relu1.dinputs)

# Print gradients
print(l1.dweights)
print(l1.dbiases)
print(l2.dweights)
print(l2.dbiases)
