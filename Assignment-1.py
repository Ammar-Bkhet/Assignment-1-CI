import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, diff, hessian, solve

x, y = symbols('x y')

f_xy = x*3 + 2 * x2 + x*y + y*2

gradient = np.array([diff(f_xy, x), diff(f_xy, y)])

critical_points = solve(gradient, (x, y))
print("Critical points:", critical_points)

H = hessian(f_xy, (x, y))

for point in critical_points:
    hessian_at_point = H.subs({x: point[0], y: point[1]}).evalf()
    print(f"Hessian at {point}:")
    print(hessian_at_point)

# Prepare to plot the function
x_vals = np.linspace(-3, 3, 100)
y_vals = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X*3 + 2 * X2 + X * Y + Y*2

# Create a figure for 3D and contour plots
fig = plt.figure(figsize=(14, 6))

# 3D Plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
ax1.set_title('3D Surface Plot of f(x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')

# Contour Plot
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour, ax=ax2, label='f(x, y)')

# Plot the critical points
for point in critical_points:
    ax2.plot(float(point[0]), float(point[1]), 'ro', markersize=10)  # Mark critical points in red
    ax2.text(float(point[0]), float(point[1]), f' {point}', fontsize=12, color='white')

ax2.set_title('Contour Plot of f(x, y) with Critical Points')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.grid()

plt.tight_layout()
plt.show()
