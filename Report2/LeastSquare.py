import numpy as np

def solve_problem_1():
    print("=== PROBLEM 1 ===\n")
    A = np.array([[3, -1], [1, 2], [2, 1]])
    b = np.array([4, 0, 1])

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"Rozwiązanie (x1, x2): {x}")

def solve_problem_2():
    print("=== PROBLEM 2 ===\n")
    x_data = np.array([0, 1, 1, -1])
    y_data = np.array([3, 0, -1, 2])

    A = np.column_stack([
        np.ones_like(x_data),
        x_data**2,
        np.sin(np.pi * x_data / 2)
    ])

    coeffs = np.linalg.lstsq(A, y_data, rcond=None)[0]
    print(f"Współczynniki (a0, a1, a2): {coeffs}")

def solve_problem_6():
    print("=== PROBLEM 6 ===\n")
    x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    y = np.array([2, 7, 9, 12, 13, 14, 14, 13, 10, 8, 4])

    # Linear method
    A1 = np.column_stack([np.ones_like(x), x])
    c1 = np.linalg.lstsq(A1, y, rcond=None)[0]
    norm1 = np.linalg.norm(A1 @ c1 - y)

    # Quadratic method
    A2 = np.column_stack([np.ones_like(x), x, x**2])
    c2 = np.linalg.lstsq(A2, y, rcond=None)[0]
    norm2 = np.linalg.norm(A2 @ c2 - y)

    print(f"Norma l2 (linia): {norm1:.4f}")
    print(f"Norma l2 (kwadrat): {norm2:.4f}")
    print("Wybieramy model z mniejszą normą.")

solve_problem_1()
solve_problem_2()
solve_problem_6()