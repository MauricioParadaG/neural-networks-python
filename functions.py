import numpy as np
import matplotlib.pyplot as plt

# sigmoid function, continuous
def sigmoid(x):
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
def relu(x):
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







