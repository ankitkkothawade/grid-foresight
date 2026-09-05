

import numpy as np

print("NumPy version:", np.__version__)
print("\n--- NumPy BLAS/LAPACK build config ---")
np.show_config()

print("\n--- Testing matmul on clean synthetic random data ---")
print("(If any RuntimeWarning prints below this line, that's the real bug --")
print(" there is no legitimate reason for one on random, well-behaved data.)\n")

a = np.random.rand(2000, 50)
b = np.random.rand(50, 2000)
result = np.matmul(a, b)

print("Result contains NaN:", np.isnan(result).any())
print("Result contains Inf:", np.isinf(result).any())
print("Done -- if nothing warned above, matmul itself is fine and the bug")
print("is somewhere else after all.")