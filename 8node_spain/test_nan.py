import numpy as np

n = 8
prices = np.zeros((n, n))
demand = np.zeros((n, n))
f = np.zeros((n, n))
fext = np.zeros((n, n))
alfa_od = np.ones((n, n))
beta_od = np.ones((n, n))
term_fij = 0.0
alt_utility = np.zeros((n, n))
n_airlines = 8

oo, dd = 0, 0

base = (prices[oo, dd] * demand[oo, dd]) ** beta_od[oo, dd]
print("base:", base)

base_beta = (alfa_od[oo, dd] + 1e-4) * ((beta_od[oo, dd] * demand[oo, dd] * prices[oo, dd]) ** (beta_od[oo, dd] - 1))
print("base_beta:", base_beta)

val1 = f[oo, dd] * (np.log(max(0, f[oo, dd]) + 1e-12) - 1)
val2 = fext[oo, dd] * (np.log(max(0, fext[oo, dd]) / n_airlines + 1e-12) - 1)

print("val1:", val1, "val2:", val2)

grad_beta_v = base_beta * (term_fij - alt_utility[oo, dd] * fext[oo, dd] + val1 + val2)

print("grad_beta_v:", grad_beta_v)
