import numpy as np

A = np.mat([[19,6], [20,10]])
print(np.linalg.det(A))
print(np.sqrt(np.linalg.det(A)))
print([np.mat(np.random.random((2,2))) for _ in range(4)])

print(np.mean(A,axis=0)[0,:])
print(np.argmax(A, axis=1))
print(np.exp(0))