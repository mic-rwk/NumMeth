import numpy as np
from scipy.linalg import eig, det, inv

# Problem 1
def solve_problem_1():
    A1 = np.array([[1, 0, 0], [2, 1, 0], [0, 0, 3]])
    
    matrices = [A1]
    results = []
    
    for i, A in enumerate(matrices):
        evals, evecs = np.linalg.eig(A)
        trace_A = np.trace(A)
        sum_evals = np.sum(evals)
        det_A = np.linalg.det(A)
        prod_evals = np.prod(evals)
        is_singular = np.isclose(det_A, 0)
        
        results.append({
            "Matrix": f"A{i+1}",
            "Eigenvalues": evals,
            "Eigenvectors": evecs,
            "Trace": trace_A,
            "Sum Evals": sum_evals,
            "Det": det_A,
            "Prod Evals": prod_evals,
            "Singular": is_singular
        })
    return results

# Problem 2
def power_iteration(A, num_simulations=100):
    n = A.shape[0]
    b_k = np.random.rand(n)
    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)
        # re-normalize the vector
        b_k = b_k1 / np.linalg.norm(b_k1)
    
    # Rayleigh quotient
    eigenvalue = np.dot(b_k.T, np.dot(A, b_k)) / np.dot(b_k.T, b_k)
    return eigenvalue, b_k

def inverse_power_iteration(A, shift=0, num_simulations=100):
    n = A.shape[0]
    I = np.eye(n)
    A_shifted = A - shift * I
    b_k = np.random.rand(n)
    
    try:
        A_inv = np.linalg.inv(A_shifted)
    except np.linalg.LinAlgError:
        return shift, b_k # shift is an eigenvalue
        
    for _ in range(num_simulations):
        b_k1 = np.dot(A_inv, b_k)
        b_k = b_k1 / np.linalg.norm(b_k1)
        
    eigenvalue = np.dot(b_k.T, np.dot(A, b_k)) / np.dot(b_k.T, b_k)
    return eigenvalue, b_k

def solve_problem_2():
    A = np.array([[4, 2, 0, 0],
                  [1, 4, 1, 0],
                  [0, 1, 4, 1],
                  [0, 0, 2, 4]])
    
    max_eval, _ = power_iteration(A)
    # For smallest, use inverse power with shift 0
    min_eval, _ = inverse_power_iteration(A, shift=0)
    
    return max_eval, min_eval

# Problem 3
def solve_problem_3():
    P = np.array([[4, -5], [2, -3]])
    u0 = np.array([8, 5])
    
    evals, evecs = np.linalg.eig(P)
    # Solve u0 = c1*v1 + c2*v2 -> [v1 v2] [c1 c2]^T = u0
    C = np.linalg.solve(evecs, u0)
    
    return evals, evecs, C

res1 = solve_problem_1()
res2 = solve_problem_2()
res3 = solve_problem_3()

print("PROBLEM 1 RESULTS:")
for r in res1:
    print(f"--- {r['Matrix']} ---")
    print("Eigenvalues:", r['Eigenvalues'])
    print("Trace:", r['Trace'], "Sum:", r['Sum Evals'])
    print("Det:", r['Det'], "Prod:", r['Prod Evals'])
    print("Singular:", r['Singular'])

print("\nPROBLEM 2 RESULTS:")
print("Largest Eigenvalue (Power Method):", res2[0])
print("Smallest Eigenvalue (Inverse Power Method):", res2[1])

print("\nPROBLEM 3 RESULTS:")
print("Eigenvalues of P:", res3[0])
print("Eigenvectors of P:\n", res3[1])
print("Coefficients C:", res3[2])