import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os
import json

NX, NY = 1000, 1000  
LX, LY = 1.0, 1.0  
DX, DY = LX / (NX - 1), LY / (NY - 1) 

x = np.linspace(0, LX, NX)
y = np.linspace(0, LY, NY)
X, Y = np.meshgrid(x, y)

def source_term(x, y):
    """
    Function for the source term in the Poisson equation.
    Returns the source term values based on x and y.
    """
    return 2 * ((x**4) * (3 * y - 2) + 
                (x**3) * (4 - 6 * y) + 
                (x**2) * (6 * y**3 - 12 * y**2 + 9 * y - 2) - 
                6 * x * ((y - 1)**2) * y + 
                (y - 1)**2 * y)

f_vec = source_term(X, Y).flatten()
N = NX * NY
A = sp.lil_matrix((N, N))

for i in range(N):
    xi, yi = i % NX, i // NX

    if xi == 0 or xi == NX - 1 or yi == 0 or yi == NY - 1:
        A[i, i] = 1  
        f_vec[i] = 0  
    else:
        A[i, i] = -2 / DX**2 - 2 / DY**2
        A[i, i - 1] = 1 / DX**2
        A[i, i + 1] = 1 / DX**2
        A[i, i - NX] = 1 / DY**2
        A[i, i + NX] = 1 / DY**2

A = A.tocsr()
u = spla.spsolve(A, f_vec)
u_fdm = u.reshape((NY, NX))

def analytic_solution(x, y):
    """
    Analytic solution to the Poisson equation for comparison.
    """
    return (x**2) * ((x - 1)**2) * y * ((y - 1)**2)

u_true = analytic_solution(X, Y)

numerator = np.linalg.norm(u_fdm - u_true, ord=2)  
denominator = np.linalg.norm(u_true, ord=2)        
l2_relative_error = numerator / denominator
ordered_points = np.vstack((X.flatten(), Y.flatten())).T 

results = {
    "evaluation_points": ordered_points.tolist(),
    "u_fdm": u_fdm.tolist(),
    'u_true': u_true.tolist(),
    "l2_rel": f"{l2_relative_error:.2e}"
}


os.makedirs("results", exist_ok=True)
output_file = os.path.join("results", "fdm_2d_poisson.json")
with open(output_file, "w") as json_file:
    json.dump(results, json_file, indent=4)
    
print(f"L2 Relative Error: {l2_relative_error:.2e}")
print(f"Results saved to '{output_file}'.")