import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from layers import Linear, Relu, SoftmaxCCE
from optimizers import SGD

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

l1 = Linear(2, 64)
l2 = Linear(64, 3)
relu1 = Relu()

loss_fn = SoftmaxCCE()

optim = SGD(learning_rate=1, decay=1e-65)

for epoch in range(20001):

    l1.forward(X)
    relu1.forward(l1.output)
    l2.forward(relu1.output)

    loss = loss_fn.forward(l2.output, y)

    predictions = np.argmax(loss_fn.output, axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(
            f"epoch: {epoch} , "
            + f"acc: {accuracy:.3} , "
            + f"loss: {loss:.3} "
            + f"lr: {optim.current_learning_rate}"
        )

    # Backward pass
    loss_fn.backward(loss_fn.output, y)
    l2.backward(loss_fn.dinputs)
    relu1.backward(l2.dinputs)
    l1.backward(relu1.dinputs)

    # Optimization step
    optim.pre_update_params()
    optim.update_params(l1)
    optim.update_params(l2)
    optim.post_update_params()
