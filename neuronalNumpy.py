import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

n = 1000
guassion_quantiles =  make_gaussian_quantiles(mean=None, cov=0.1, n_samples=n, n_features=2, n_classes=2, shuffle=True, random_state=None)

x, y = guassion_quantiles # x is the data, y is the label

print(x.shape) # (1000, 2)
print(y.shape) # (1000,)

y = y[: , np.newaxis] # (1000, 1)

plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=40, cmap=plt.cm.Spectral)
plt.show()

def initialize_parameters(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(0, L-1):
        parameters['W' + str(l+1)] = (np.random.rand(layers_dims[l], layers_dims[l+1]) * 2) - 1 
        parameters['b' + str(l+1)] = (np.zeros((1, layers_dims[l+1])) * 2) - 1
    return parameters

layers_dims = [2, 4, 8, 1] # 2 inputs, 4 neurons, 8 neurons, 1 output
parameters = initialize_parameters(layers_dims)
print(parameters)

