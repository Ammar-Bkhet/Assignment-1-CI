import numpy as np

# Define the function F(x, y)
def F(x, y):
    return 3 * x**4 + 3 * x**2 * y**2 + x**2 + 2 * y**4

# Define the gradient of F with respect to x and y
def gradient(x, y):
    dF_dx = 12 * x**3 + 6 * x * y**2 + 2 * x
    dF_dy = 6 * x**2 * y + 8 * y**3
    return np.array([dF_dx, dF_dy])

# Gradient descent parameters
learning_rate = 0.01
epsilon = 1e-5
max_iterations = 100000  # To prevent infinite loops if convergence fails

# Starting point
point = np.array([1.0, 1.0])  # P0(1,1)

# Gradient descent loop
for i in range(max_iterations):
    grad = gradient(point[0], point[1])
    grad_norm = np.sqrt(grad[0]**2 + grad[1]**2)
    
    # Stop if the gradient norm is smaller than epsilon
    if grad_norm < epsilon:
        print(f"Converged in {i} iterations.")
        break
    
    # Update the point
    point = point - learning_rate * grad

# Results
print("Local minimum point:", point)
print("Function value at local minimum:", F(point[0], point[1]))

