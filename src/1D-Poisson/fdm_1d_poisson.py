import os
import numpy as np
import json

def u_e(x):
    return x * np.exp(-(x**4))

def f(x):
    return 16 * np.exp(-x**4) * x**7 - 20 * np.exp(-x**4) * x**3

u_L, u_R = 0, np.exp(-1)
ordered_points = np.linspace(0, 1, 512).reshape(-1, 1)
n = 2048
x = np.linspace(0, 1, n + 1)
h = x[1] - x[0]

A = (np.diag(-2 * np.ones(n - 1)) +
     np.diag(np.ones(n - 2), k=1) +
     np.diag(np.ones(n - 2), k=-1)) / h**2

b = f(x[1:-1])
b[0] -= u_L / h**2
b[-1] -= u_R / h**2

u_inner = np.linalg.solve(A, b)
u_numeric = np.concatenate(([u_L], u_inner, [u_R]))
u_fdm = np.interp(ordered_points.flatten(), x, u_numeric)
u_true = u_e(ordered_points.flatten())

error = u_fdm - u_true
l2_norm_error = np.linalg.norm(error, ord=2)
l2_norm_true = np.linalg.norm(u_true, ord=2)
l2_relative_error = l2_norm_error / l2_norm_true

results = {
    "evaluation_points": ordered_points.flatten().tolist(),
    "u_fdm": u_fdm.tolist(),
    'u_true': u_true.tolist(),
    "l2_rel": f"{l2_relative_error:.2e}"
}

os.makedirs("results", exist_ok=True)
output_file = os.path.join("results", "fdm_1d_poisson.json")
with open(output_file, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"L2 Relative Error: {l2_relative_error:.2e}")
print(f"Results saved to '{output_file}'.")