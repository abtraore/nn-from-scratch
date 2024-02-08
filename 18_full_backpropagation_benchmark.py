import numpy as np
from timeit import timeit
import nnfs

from layers import *
from loss import *

nnfs.init()
softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
class_targets = np.array([0, 1, 1])


def f1():
    softmax_loss = SoftmaxCCE()
    softmax_loss.backward(softmax_outputs, class_targets)
    dvalues1 = softmax_loss.dinputs


def f2():
    activation = Softmax()
    activation.output = softmax_outputs
    loss = CCE()
    loss.backward(softmax_outputs, class_targets)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs


t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)
print(f"f2 is {t2 / t1:.2} time slower than f1.")
