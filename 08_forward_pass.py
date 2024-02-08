import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from layers import Linear

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

l1 = Linear(2, 3)
l1.forward(X)

print(l1.output.shape)
