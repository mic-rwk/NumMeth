import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Function Definitions
def griewank(x):
    part_sum = np.sum(x**2) / 4000
    part_prod = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return part_sum - part_prod + 1

def rastrigin(x):
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

# Evolutionary Solver with History Tracking

def solve_with_history(func, bounds, n_dim):
    history = []
    
    # Callback to capture the best fitness at each generation
    def callback(xk, convergence):
        history.append(func(xk))
    
    res = differential_evolution(func, [bounds] * n_dim, callback=callback, tol=1e-6)
    return res, history

# Visualization

def plot_benchmark(func, bounds, title):
    # Create Grid for 2D/3D plots
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([func(np.array([x_val, y_val])) for x_val, y_val in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    fig = plt.figure(figsize=(18, 5))

    # 1. 3D Surface Plot
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_title(f'3D Surface: {title}')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # 2. 2D Contour Plot
    ax2 = fig.add_subplot(1, 3, 2)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='magma')
    ax2.set_title(f'2D Contour: {title}')
    fig.colorbar(contour, ax=ax2)

    # 3. Evolutionary Process (Fitness)
    # Solving for n=2 to match plots, but algorithm works for any n
    result, history = solve_with_history(func, bounds, n_dim=2)
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(history, color='blue', linewidth=2)
    ax3.set_yscale('log')
    ax3.set_title(f'Fitness vs Evolution ({title})')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Best Fitness (Log Scale)')
    ax3.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.show()
    
    print(f"Results for {title}:")
    print(f"  Global Minimum found at: {result.x}")
    print(f"  Function value at minimum: {result.fun:.10f}\n")

# Execution

# Griewank usually analyzed in range [-5, 5] or wider
plot_benchmark(griewank, [-10, 10], "Griewank Function")

# Rastrigin usually analyzed in range [-5.12, 5.12]
plot_benchmark(rastrigin, [-5.12, 5.12], "Rastrigin Function")