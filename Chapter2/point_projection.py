import numpy as np
X = np.array([20, 30, 40]).T
z = X[2]
K = np.array([[10, 0, 20],
              [0, 20, 40],
              [0, 0, 1]])

uv_norm = X / z
print(uv_norm)
uv = K @ uv_norm
print(uv)