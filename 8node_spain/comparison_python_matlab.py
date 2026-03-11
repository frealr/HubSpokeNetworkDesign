import numpy as np
import scipy.io as sio

tol = 1e-4

mat = sio.loadmat("debug_matlab.mat", squeeze_me=True, struct_as_record=False)
py  = sio.loadmat("debug_python.mat", squeeze_me=True, struct_as_record=False)

debug_mat = mat["debug"]
debug_py  = py["debug"]

vars_to_check = [
    "a","f","fext","fij",
    "grad_alfa_v","grad_beta_v",
    "grad_alfa_f","grad_beta_f",
    "alfa_od","beta_od"
]

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x.squeeze()
    return np.array(x)

print("Comparando iteraciones...\n")

rows, cols = debug_mat.shape

k = 0

for i in range(rows):
    for j in range(cols):

        iter_py = debug_py[k].iter
        bliter_py = debug_py[k].bliter

        for v in vars_to_check:

            m = to_numpy(getattr(debug_mat[i,j], v))
            p = to_numpy(getattr(debug_py[k], v))

            if m.shape != p.shape:
                print("\nSHAPE DIFERENTE")
                print("iter:", iter_py)
                print("bliter:", bliter_py)
                print("variable:", v)
                print("MATLAB shape:", m.shape)
                print("Python shape:", p.shape)
                raise SystemExit

            err = np.max(np.abs(m.astype(float) - p.astype(float)))

            if err > tol:

                print("\nDIVERGENCIA DETECTADA")
                print("iter:", iter_py)
                print("bliter:", bliter_py)
                print("variable:", v)
                print("error máximo:", err)

                print("\nStats MATLAB")
                print("sum:", np.sum(m))
                print("max:", np.max(m))
                print("min:", np.min(m))

                print("\nStats Python")
                print("sum:", np.sum(p))
                print("max:", np.max(p))
                print("min:", np.min(p))

                raise SystemExit

        k += 1

print("\nTodo coincide dentro de la tolerancia.")