import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from layers import Linear, Relu, Softmax
from loss import CCE

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

l1 = Linear(2, 3)
l2 = Linear(3, 3)

relu1 = Relu()
relu2 = Relu()

softmax = Softmax()

loss_fn = CCE()

l1.forward(X)
relu1.forward(l1.output)
l2.forward(relu1.output)
softmax.forward(l2.output)

loss = loss_fn.calculate(softmax.output, y)

predictions = np.argmax(softmax.output, axis=1)

accuracy = np.mean(predictions == y)

print(f"{loss=}")
print(f"{accuracy=}")
