import numpy as np

def jacobi(A, b, x0, tol=1e-6, max_iter=100):
    D = np.diag(np.diag(A))
    LU = A - D
    x = x0.copy()
    history = [x.copy()]
    for i in range(max_iter):
        x_new = np.linalg.solve(D, b - np.dot(LU, x))
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, i + 1, history
        x = x_new
        history.append(x.copy())
    return x, max_iter, history

def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
    L = np.tril(A)
    U = A - L
    x = x0.copy()
    history = [x.copy()]
    for i in range(max_iter):
        x_new = np.linalg.solve(L, b - np.dot(U, x))
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, i + 1, history
        x = x_new
        history.append(x.copy())
    return x, max_iter, history

# Define the system
A = np.array([[1, 1, 1],
              [1, 1, 2],
              [1, 2, 2]], dtype=float)
b = np.array([1, 2, 1], dtype=float)
x0 = np.zeros(3)

# Exact solution check
try:
    x_exact = np.linalg.solve(A, b)
except np.linalg.LinAlgError:
    x_exact = "Singular matrix"

# Check convergence criteria (Spectral radius)
D = np.diag(np.diag(A))
L = np.tril(A, -1)
U = np.triu(A, 1)

# Jacobi iteration matrix: T_j = -D^-1 (L + U)
T_j = -np.linalg.inv(D) @ (L + U)
rho_j = max(abs(np.linalg.eigvals(T_j)))

# Gauss-Seidel iteration matrix: T_gs = -(D+L)^-1 U
T_gs = -np.linalg.inv(D + L) @ U
rho_gs = max(abs(np.linalg.eigvals(T_gs)))

print(f"Exact Solution: {x_exact}")
print(f"Spectral Radius Jacobi: {rho_j}")
print(f"Spectral Radius Gauss-Seidel: {rho_gs}")