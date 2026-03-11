import scipy.io as sio
import numpy as np
import glob

# Python result
py_file = './8node_hs_prueba_v0_blo/bud=50000.0_lam=4_alfa=0.1_mu_al=1e-07_mu_bet=0.2_python.mat'
py_data = sio.loadmat(py_file)

# MATLAB result (finding the best match)
mat_files = glob.glob('./8node_hs_prueba_v0_blo/bud=50000*alfa=1.000000e-01*mu_al=1.000000e-07*mu_bet=2.000000e-01.mat')
if mat_files:
    mat_file = mat_files[0]
    mat_data = sio.loadmat(mat_file)
    print(f"MATLAB Obj: {mat_data['obj_val_ll'][0][0]}")
    print(f"Python Obj: {py_data['obj_val_ll'][0][0]}")
    
    print(f"MATLAB pax_obj: {mat_data.get('pax_obj', [['N/A']])[0][0]}")
    print(f"Python pax_obj: {py_data.get('pax_obj', [['N/A']])[0][0]}")
    
    print(f"MATLAB op_obj: {mat_data.get('op_obj', [['N/A']])[0][0]}")
    print(f"Python op_obj: {py_data.get('op_obj', [['N/A']])[0][0]}")
    
    print(f"Sum f (PY): {np.sum(py_data['f'])}")
    if 'f' in mat_data:
        print(f"Sum f (MAT): {np.sum(mat_data['f'])}")
