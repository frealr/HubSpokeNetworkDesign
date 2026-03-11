import scipy.io as sio
import glob
import numpy as np

np.set_printoptions(suppress=True, precision=4, linewidth=120)

f = './8node_hs_prueba_v0_blo/bud=50000_lam=4_alfa=1.000000e-01_mu_al=1.000000e-07_mu_bet=5.000000e-02.mat'
try:
    mat = sio.loadmat(f)
    print("MATLAB (mu_bet=5e-2) obj_hist:\n", mat['obj_hist'].flatten()[:10])
    print("MATLAB (mu_bet=5e-2) alfa_od:\n", mat['alfa_od'])
except Exception as e:
    pass

f = './8node_hs_prueba_v0_blo/bud=50000.0_lam=4_alfa=0.1_mu_al=1e-07_mu_bet=0.2_python.mat'
try:
    mat = sio.loadmat(f)
    print("PYTHON (mu_bet=0.2) obj_hist:\n", mat['obj_hist'].flatten()[:10])
    print("PYTHON (mu_bet=0.2) alfa_od:\n", mat['alfa_od'])
except Exception as e:
    pass
