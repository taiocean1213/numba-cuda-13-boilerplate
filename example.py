import numpy as np
from numba import cuda

print("CUDA available:", cuda.is_available())

@cuda.jit
def add_kernel(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

if cuda.is_available():
    N = 100000
    a = np.arange(N, dtype=np.float32)
    b = np.arange(N, dtype=np.float32)
    c = np.zeros(N, dtype=np.float32)

    # Launch kernel with enough threads
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    add_kernel[blocks_per_grid, threads_per_block](a, b, c)

    # Check if computation is correct
    expected = a + b
    if np.allclose(c, expected):
        print("GPU computation successful!")
    else:
        print("GPU computation failed!")
else:
    print("No CUDA GPU available.")
