import numpy as np
import matplotlib.pyplot as plt

# sigmoid function, continuous
def sigmoid(x, derivate=False):
    if derivate:
        return np.exp(-x)/((1+np.exp(-x))**2)
    else:
        return 1/(1+np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)
print(x) # [-10.  -9.79  -9.59   -9.39 
plt.plot(x, y)
plt.show()


# step function, discrete
def step(x):
    return np.piecewise(x, [x < 0.0, x >= 0.0], [0, 1])

x = np.linspace(-10, 10, 100)
y = step(x)
print(x)
plt.plot(x, y)
plt.show()


# ReLU function, continuous
def relu(x, derivate=False):
    if derivate:
        x[x<=0] = 0
        x[x>0] = 1
        return x
    else:
        return np.piecewise(x, [x < 0.0, x >= 0.0], [0, lambda x: x])

x = np.linspace(-10, 10, 100)
y = relu(x)
plt.plot(x, y)
plt.show()

# Tanh function, continuous
def tanh(x):
    return np.tanh(x)

x = np.linspace(-10, 10, 100)
y = tanh(x)
plt.plot(x, y)
plt.show()


# mse mean squared error, loss function
def mse(y, y_hat, derivate=False):
    if derivate:
        return 2 * (y_hat - y) / y.size
    return np.mean((y - y_hat)**2)

real = np.array([1, 2, 3, 4, 5])
prediction = np.array([1, 2, 3, 4, 3])
print(mse(real, prediction)) # 0.8







