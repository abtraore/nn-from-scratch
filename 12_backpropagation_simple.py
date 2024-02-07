x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

# Multiply inputs by weights.
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print(xw0, xw1, xw2, b)

# Adding weighted inputs and bias.
z = xw0 + xw1 + xw2 + b

# ReLU activation.
y = max(0, z)

# Example derivative fron the next layer.
d_value = 1.0

# Derivative of ReLU and the chain rule.
d_relu_dz = d_value * (1.0 if z > 0 else 0.0)
print(d_relu_dz)

# Partial derivative w.r.t wiri with in i range [0,3] and chain rule [SUM].

dsum_dx0w0 = 1  # Partial derivative result (sum property).
drelu_dx0w0 = d_relu_dz * dsum_dx0w0  # Chain rule.

dsum_dx1w1 = 1  # Partial derivative result (sum property).
drelu_dx1w1 = d_relu_dz * dsum_dx1w1  # Chain rule.

dsum_dx2w2 = 1  # Partial derivative result (sum property).
drelu_dx2w2 = d_relu_dz * dsum_dx2w2  # Chain rule.


dsum_db = 1  # Partial derivative result (sum property).
drelu_db = d_relu_dz * dsum_db  # Chain rule.


# Partial derivative w.r.t xi with in i range [0,3[ and chain rule [MUL].
dmul_dx0 = w[0]  # Partial derivative result (mul property).
drelu_dx0 = drelu_dx0w0 * dmul_dx0  # Chain rule.

dmul_dx1 = w[1]  # Partial derivative result (mul property).
drelu_dx1 = drelu_dx1w1 * dmul_dx1  # Chain rule.

dmul_dx2 = w[2]  # Partial derivative result (mul property).
drelu_dx2 = drelu_dx2w2 * dmul_dx2  # Chain rule.

# Partial derivative w.r.t wi with in i range [0,3[ and chain rule [MUL].
dmul_dw0 = x[0]  # Partial derivative result (mul property).
drelu_dw0 = drelu_dx0w0 * dmul_dw0  # Chain rule.

dmul_dw1 = x[1]  # Partial derivative result (mul property).
drelu_dw1 = drelu_dx1w1 * dmul_dw1  # Chain rule.

dmul_dw2 = w[2]  # Partial derivative result (mul property).
drelu_dw2 = drelu_dx2w2 * dmul_dw2  # Chain rule.

dx = [drelu_dx0, drelu_dx1, drelu_dx2]
dw = [drelu_dw0, drelu_dw1, drelu_dw2]
db = drelu_db


w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]

b += -0.001 * db

print(w, b)


# Multiply inputs by weights.
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print(xw0, xw1, xw2, b)

# Adding weighted inputs and bias.
z = xw0 + xw1 + xw2 + b

# ReLU activation.
y = max(0, z)

print(y)
