import numpy as np

scalar = np.array(5)
print("scalar dim: ", scalar.ndim) # 0
print("scalar shape: ", scalar.shape) # ()

vector = np.array([1,2,3])
print("vector dim: ", vector.ndim) # 1
print("vector shape: ", vector.shape) # (3,)

matrix = np.array([[1,2,3],[4,5,6]])
print("matrix dim: ", matrix.ndim) # 2
print("matrix shape: ", matrix.shape) # (2,3)

tensor = np.array([
  [[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print("tensor dim: ", tensor.ndim) # 3
print("tensor shape: ", tensor.shape) # (2,2,3)

#reshape a tensor
tensor1 = np.array([[0,1],
                    [2,3],
                    [4,5],
                    [6,7],
                    ])
print(tensor1.shape) # (4,2)

toReshape = tensor1.reshape((8,1))
print(toReshape) # [[0] [1] [2] [3] [4] [5] [6] [7]]
toReshape = tensor1.reshape((2,4))
print(toReshape) # [[0 1 2 3] [4 5 6 7]]


# transpose a tensor
tensor2 = np.transpose(tensor1)
print(tensor2) # [[0 2 4 6] [1 3 5 7]]






