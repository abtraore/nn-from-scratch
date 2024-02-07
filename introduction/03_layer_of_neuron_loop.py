inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]

bias = [2, 3, 0.5]

outputs = []


for n_weights, n_bias in zip(weights, bias):

    n_outputs = 0
    for n_inputs, weight in zip(inputs, n_weights):
        n_outputs += n_inputs * weight

    n_outputs += n_bias

    outputs.append(n_outputs)

print(outputs)
