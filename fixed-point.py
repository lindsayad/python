import numpy as np

# Define the matrix A and vector b
I = np.array([[1, 0], [0, 1]])
A = np.array([[2, -1], [1, 2]])
b = np.array([1, 3])
Minv = np.array([[0.5, 0], [0, 0.5]])

# Function to perform fixed-point iteration
def fixed_point_iteration(A, b, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        x_new = (I - Minv @ A) @ x + Minv @ b
        if np.linalg.norm(x_new - x) < tol:
            return x_new  # Convergence achieved
        x = x_new
        print("Iteration ", i + 1, " complete")
        print("Current solution: ", x)
    return x  # Return the last computed value if max_iter reached

# Initial guess
x0 = np.array([0, 0])

# Perform fixed-point iteration
solution = fixed_point_iteration(A, b, x0)

print("Solution using fixed-point iteration:", solution)
