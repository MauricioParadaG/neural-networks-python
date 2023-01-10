import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from functions import relu, sigmoid, mse

n = 1000
guassion_quantiles =  make_gaussian_quantiles(mean=None, cov=0.1, n_samples=n, n_features=2, n_classes=2, shuffle=True, random_state=None)

x_data, y_label = guassion_quantiles

print(x_data.shape) # (1000, 2)
print(y_label.shape) # (1000,)

y_label = y_label[: , np.newaxis] # (1000, 1)

plt.scatter(x_data[:, 0], x_data[:, 1], c=y_label[:, 0], s=40, cmap=plt.cm.Spectral)
# plt.show()

def initialize_parameters(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(0, L-1):
        parameters['W' + str(l+1)] = (np.random.rand(layers_dims[l], layers_dims[l+1]) * 2) - 1 
        parameters['b' + str(l+1)] = (np.random.rand(1, layers_dims[l+1]) * 2) - 1
    return parameters

""" layers_dims = [2, 4, 8, 1] # 2 inputs, 4 neurons, 8 neurons, 1 output
parameters = initialize_parameters(layers_dims)
# print(parameters)

print(parameters['W1'].shape) # (2, 4)

np.matmul(x_data, parameters['W1']) # (1000, 4) """


def train(x_data, learning_rate, parameters, training=True):
# Forward propagation

    parameters['A0'] = x_data
        
    parameters['Z1'] = np.matmul(parameters['A0'],parameters['W1']) + parameters['b1']
    parameters['A1'] = relu(parameters['Z1'])
    
    parameters['Z2'] = np.matmul(parameters['A1'],parameters['W2']) + parameters['b2']
    parameters['A2'] = relu(parameters['Z2'])
       
    parameters['Z3'] = np.matmul(parameters['A2'],parameters['W3']) + parameters['b3']
    parameters['A3'] = sigmoid(parameters['Z3'])
  
    output = parameters['A3']
    
    if training:
    # Backpropagation
    
        parameters['dZ3'] =  mse(y_label,output,True) * sigmoid(parameters['A3'],True)
        parameters['dW3'] = np.matmul(parameters['A2'].T,parameters['dZ3'])
        
        parameters['dZ2'] = np.matmul(parameters['dZ3'],parameters['W3'].T) * relu(parameters['A2'],True)
        parameters['dW2'] = np.matmul(parameters['A1'].T,parameters['dZ2'])
        
        parameters['dZ1'] = np.matmul(parameters['dZ2'],parameters['W2'].T) * relu(parameters['A1'],True)
        parameters['dW1'] = np.matmul(parameters['A0'].T,parameters['dZ1'])

        
         # Gradient descent - update W and b
           
        parameters['W3'] = parameters['W3'] - parameters['dW3'] * learning_rate
        parameters['b3'] = parameters['b3'] - (np.mean(parameters['dZ3'],axis=0, keepdims=True)) * learning_rate
        
        parameters['W2'] = parameters['W2'] - parameters['dW2'] * learning_rate
        parameters['b2'] = parameters['b2'] - (np.mean(parameters['dZ2'],axis=0, keepdims=True)) * learning_rate
        
        parameters['W1'] = parameters['W1'] -parameters['dW1'] * learning_rate
        parameters['b1'] = parameters['b1'] - (np.mean(parameters['dZ1'],axis=0, keepdims=True)) * learning_rate
    
    return output

  # Train neuronal network

layers_dims = [2, 4, 8, 1] # 2 inputs, 4 neurons, 8 neurons, 1 output
parameters = initialize_parameters(layers_dims)

errors = []
for i in range(50000):
    output = train(x_data, 0.001, parameters)
    if i % 10000 == 0:
        print(mse(y_label,output))
        errors.append(mse(y_label,output))

plt.plot(errors)
plt.show()















