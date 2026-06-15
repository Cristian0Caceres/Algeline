import numpy as np

M = np.array([
    [0,   0, 1],
    [0.5, 0, 0],
    [0.5, 1, 0]
])

v = np.array([
    [1],
    [0],
    [0]
])

for i in range(5):
    v = M @ v
    print(f"Paso {i+1}")
    print(v)