import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def solve_problem_1():
    x1, x2 = sp.symbols('x1 x2')

    f_sym = 100 * (x2 - x1**2)**2 + (1 - x1)**2

    grad_f = [sp.diff(f_sym, x1), sp.diff(f_sym, x2)]

    Hessian = sp.Matrix([[sp.diff(f_sym, x1, x1), sp.diff(f_sym, x1, x2)],
                        [sp.diff(f_sym, x2, x1), sp.diff(f_sym, x2, x2)]])

    point = {x1: 1, x2: 1}

    grad_eval = [float(g.subs(point)) for g in grad_f]
    hess_eval = np.array(Hessian.subs(point), dtype=float)

    print("--- Problem 1 ---")
    print(f"Gradient w punkcie [1, 1]T: {np.round(grad_eval, 4)}")
    print(f"Hessian w punkcie [1, 1]T:\n {np.round(hess_eval, 4)}")
    print("-" * 30)

    X = np.linspace(-2, 2, 500)
    Y = np.linspace(-1, 3, 500)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z = 100 * (Y_grid - X_grid**2)**2 + (1 - X_grid)**2

    plt.figure(figsize=(7, 5))
    levels = [0.1, 1, 10, 50, 100, 500, 2000, 5000]
    cp = plt.contour(X_grid, Y_grid, Z, levels=levels, cmap='jet')
    plt.clabel(cp, inline=1, fontsize=8)
    plt.plot(1, 1, 'r*', markersize=12, label='Optimum [1, 1]')
    plt.title("Wykres poziomicowy funkcji Rosenbrocka")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid(True)
    plt.show()

def solve_problem_2():
    x1, x2 = sp.symbols('x1 x2')

    f_quad = 2 * x1**2 - x1 * x2 + x2**2 - 3 * x1 + 3.5

    grad_quad = [sp.diff(f_quad, x1), sp.diff(f_quad, x2)]

    stationary_points = sp.solve(grad_quad, (x1, x2))

    H_quad = sp.Matrix([[sp.diff(f_quad, x1, x1), sp.diff(f_quad, x1, x2)],
                        [sp.diff(f_quad, x2, x1), sp.diff(f_quad, x2, x2)]])

    print("--- Problem 2 ---")
    print(f"Punkt stacjonarny: {stationary_points}")
    print(f"Hessian:\n {H_quad}")
    print(f"Określoność: macierz jest dodatnio określona (wszystkie minory główne > 0)")
    print("-" * 30)

def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_f(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])

def gradient_descent_backtracking(x0, alpha_0=1.0, rho=0.5, c=1e-4, max_iter=20000):
    x = np.array(x0, dtype=float)
    alpha = alpha_0
    
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < 1e-6:
            break
        
        alpha = alpha_0
        direction = -g
        while f(x + alpha * direction) > f(x) + c * alpha * np.dot(g, direction):
            alpha *= rho
        
        x += alpha * direction
        
        if k % 100 == 0 or k < 5:
            print(f"Iteracja {k}: alfa = {alpha:.5f}, f(x) = {f(x):.6f}")
            
    return x

solve_problem_1()
solve_problem_2()

print("--- Problem 5: Przypadek 1 (x0 = [1.2, 1.2]) ---")
x_opt1 = gradient_descent_backtracking([1.2, 1.2])
print(f"Optimum: {np.round(x_opt1, 6)}")

print("\n--- Problem 5: Przypadek 2 (x0 = [-1.2, 1.0]) ---")
x_opt2 = gradient_descent_backtracking([-1.2, 1.0])
print(f"Optimum: {np.round(x_opt2, 6)}")