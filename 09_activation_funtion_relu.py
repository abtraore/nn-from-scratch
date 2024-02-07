import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from layers import Linear, Relu

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

l1 = Linear(2, 3)
l1.forward(X)
relu1 = Relu()
relu1.forward(l1.output)
print(relu1.output[:5])
