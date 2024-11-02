import numpy as np

# Define the original function F(x, y)
def F(x, y):
    return 3 * x**4 + 3 * x**2 * y**2 + x**2 + 2 * y**4

# Define the gradient of F with respect to x and y
def gradient(x, y):
    dF_dx = 12 * x**3 + 6 * x * y**2 + 2 * x
    dF_dy = 6 * x**2 * y + 8 * y**3
    return np.array([dF_dx, dF_dy])

# Define F as a function of alpha only, along the descent direction
def F_alpha(alpha, point, grad):
    # Substitute x and y in terms of alpha
    x, y = point - alpha * grad
    return F(x, y)

# First derivative of F(alpha) with respect to alpha
def dF_dalpha(alpha, point, grad):
    # Use finite differences to compute the derivative
    h = 1e-6
    return (F_alpha(alpha + h, point, grad) - F_alpha(alpha, point, grad)) / h

# Second derivative of F(alpha) with respect to alpha
def d2F_dalpha2(alpha, point, grad):
    # Use finite differences to compute the second derivative
    h = 1e-6
    return (dF_dalpha(alpha + h, point, grad) - dF_dalpha(alpha, point, grad)) / h

# Steepest Descent with Optimal Learning Rate via Line Search
epsilon = 1e-6
max_outer_iterations = 10000
max_inner_iterations = 100

# Starting point
point = np.array([1.0, 1.0])

# Steepest Descent main loop
for i in range(max_outer_iterations):
    grad = gradient(point[0], point[1])
    grad_norm = np.sqrt(grad[0]**2 + grad[1]**2)
    
    # Stop if the gradient norm is smaller than epsilon
    if grad_norm < epsilon:
        print(f"Converged in {i} iterations.")
        break

    # Initialize alpha for optimal learning rate search
    alpha = 1.0
    for j in range(max_inner_iterations):
        # Compute the first and second derivatives of F(alpha)
        df_dalpha = dF_dalpha(alpha, point, grad)
        d2f_dalpha2 = d2F_dalpha2(alpha, point, grad)
        
        # Update alpha using Newton's method
        if abs(d2f_dalpha2) < epsilon:
            print("Second derivative near zero; skipping alpha update.")
            break
        alpha = alpha - df_dalpha / d2f_dalpha2
        
        # Stop inner loop if df_dalpha is small (optimal alpha found)
        if abs(df_dalpha) < epsilon:
            break

    # Update the point with the optimized learning rate alpha
    point = point - alpha * grad

# Results
print("Local minimum point:", point)
print("Function value at local minimum:", F(point[0], point[1]))
