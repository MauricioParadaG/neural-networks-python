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
toReshape = np.array([[0,1],
                    [2,3],
                    [4,5],
                    [6,7],
                    ])
print(toReshape.shape) # (4,2)

toReshape = toReshape.reshape((8,1))
print(toReshape.shape) # (8,1)






