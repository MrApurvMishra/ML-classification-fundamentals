# import required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
import math


# forward propagation
def forward(X, params):
    G = np.dot(X, params['U'].T) + params['b']
    H = np.tanh(G)
    z = np.dot(H, params['W'].T) + params['c']
    y = sigmoid(z)

    return y


# Gradient descent using the derivatives calculated above
def grad_descent(X, t, params):
    N = X.shape[0]

    # Perform forwards computation.
    G = np.dot(X, params['U'].T) + params['b']
    H = np.tanh(G)
    z = np.dot(H, params['W'].T) + params['c']
    y = sigmoid(z)

    # average loss or error
    E = (1. / N) * np.sum(-t * np.log(y) - (1 - t) * np.log(1 - y))

    # Perform backwards computation.
    z_bar = (1. / N) * (y - t)
    W_bar = np.dot(H.T, z_bar)
    c_bar = np.dot(z_bar, np.ones(N))
    H_bar = np.outer(z_bar, params['W'].T)
    G_bar = H_bar * (1 - np.tanh(G) ** 2)
    U_bar = np.dot(G_bar.T, X)
    b_bar = np.dot(G_bar.T, np.ones(N))

    # Wrap our gradients in a dictionary.
    grads = {}
    grads['U'] = U_bar
    grads['b'] = b_bar
    grads['W'] = W_bar
    grads['c'] = c_bar

    return grads, E


# generating data
data = np.asarray([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
labels = np.asarray([0, 1, 1, 0])

# set seed value
np.random.seed(0)

# First, initialize our neural network parameters.
params = {}
params['U'] = np.random.randn(3, 2)
params['b'] = np.zeros(3)
params['W'] = np.random.randn(3)
params['c'] = 0

num_steps = 5000
alpha = 1
for step in range(num_steps):
    grads, loss = grad_descent(data, labels, params)
    for k in params:
        params[k] -= alpha * grads[k]

    # Print loss every so often.
    if step % 1000 == 0:
        print("Step {:3d} | Loss {:3.2f}".format(step, loss))

# plotting
num_data = 500
num_points = 200
x1s = np.linspace(-0.5, 1.5, num_points)
x2s = np.linspace(-0.5, 1.5, num_points)

points = np.transpose([np.tile(x1s, len(x2s)), np.repeat(x2s, len(x1s))])
Y = forward(points, params).reshape(num_points, num_points)
X1, X2 = np.meshgrid(x1s, x2s)

plt.figure()
plt.pcolormesh(X1, X2, Y, cmap=plt.cm.get_cmap('YlGn'), shading='auto')
plt.colorbar()
plt.scatter(data[:num_data//2, 0], data[:num_data//2, 1], color='red')
plt.scatter(data[num_data//2:, 0], data[num_data//2:, 1], color='blue')
plt.title("employing tanh", fontsize=15)
plt.show()
