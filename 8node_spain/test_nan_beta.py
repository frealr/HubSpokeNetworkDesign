import numpy as np

n = 8
prices = np.zeros((n, n))
demand = np.zeros((n, n))
alfa_od = np.ones((n, n))
beta_od = np.ones((n, n)) * 0.5

oo, dd = 0, 0

base_beta = (alfa_od[oo, dd] + 1e-4) * ((beta_od[oo, dd] * demand[oo, dd] * prices[oo, dd]) ** (beta_od[oo, dd] - 1))
print("base_beta with 0.5:", base_beta)

base_beta2 = (alfa_od[oo, dd] + 1e-4) * ((beta_od[oo, dd] * demand[oo, dd] * prices[oo, dd]) ** (0.5 - 1))
print("base_beta with -0.5 power:", base_beta2)

val1 = 0.0 * (np.log(max(0, 0.0) + 1e-12) - 1)
grad_beta_v = base_beta2 * (0.0 + val1)
print("grad_beta_v:", grad_beta_v)
