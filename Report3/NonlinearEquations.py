import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize

#  Problems 1 and 2 definitions
def F_prob1(x):
    return np.array([
        x[0]**3 + 3*x[0]**2 + 3*x[0] - x[1],
        x[0]**2 + 2*x[0] - x[1] + 1
    ])

def F_prob2(x):
    return np.array([
        x[0]**3 - 3*x[0]**2 + 3*x[0] - x[1] - 3,
        x[0]**2 - 2*x[0] - x[1]
    ])

def cost_func(x, F_func):
    return 0.5 * np.sum(F_func(x)**2)

# Problem 3 definition
def F_prob3(x):
    return np.array([
        2*x[0] - x[1] - np.exp(-x[0]),
        -x[0] + 2*x[1] - np.exp(-x[1])
    ])

def solve_and_plot(F_func, bounds, title, true_points=None):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[2], bounds[3], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = cost_func([X[j, i], Y[j, i]], F_func)

    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=np.logspace(-2, 3, 20), cmap='viridis')
    plt.clabel(cp, inline=True, fontsize=8)
    plt.title(f"Contour lines of Cost Function: {title}")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    if true_points:
        for label, pt in true_points.items():
            plt.plot(pt[0], pt[1], 'ro')
            plt.text(pt[0], pt[1], f" {label}", color='red')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

pts1 = {
    "Global Min": [0.46557, 2.1479],
    "Local Min": [-1, -0.5],
    "Saddle": [-1/3, -7/54]
}

print("--- Problem 1 Analysis ---")
for name, pt in pts1.items():
    res = cost_func(pt, F_prob1)
    print(f"{name} at {pt} | Cost: {res:.6f}")

solve_and_plot(F_prob1, [-2.5, 1.5, -1.5, 3], "Problem 1", pts1)

print("\n--- Problem 2 Solution ---")
res2 = least_squares(F_prob2, x0=[0, 0])
print(f"Optimal Solution: {res2.x}")
print(f"Residual: {res2.fun}")
solve_and_plot(F_prob2, [-1, 4, -4, 2], "Problem 2")

print("\n--- Problem 3 Solution ---")
res3 = least_squares(F_prob3, x0=[-5, -5])
print(f"Starting from [-5, -5], Solution: {res3.x}")
solve_and_plot(F_prob3, [-1, 2, -1, 2], "Problem 3")