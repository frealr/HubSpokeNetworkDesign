import scipy.io as sio
import glob
import numpy as np

mat_files = glob.glob('./8node_hs_prueba_v0_blo/bud=50000*alfa=1.000000e-01*mu_al=1.000000e-07*mu_bet=2.000000e-01.mat')
if mat_files:
    mat_file = mat_files[0]
    data = sio.loadmat(mat_file)
    print(f"Keys in {mat_file}: {data.keys()}")
    if 'obj_val_ll' in data:
        print(f"obj_val_ll: {data['obj_val_ll']}")
    if 'pax_obj' in data:
        print(f"pax_obj: {data['pax_obj']}")
    if 'demand' in data:
        print(f"Sum Demand: {np.sum(data['demand'])}")
        print(f"Mean Demand: {np.nanmean(data['demand'])}")
    if 'f' in data:
        print(f"Sum f: {np.sum(data['f'])}")
        print(f"Mean f: {np.nanmean(data['f'])}")
        non_zero_f = data['f'][data['f'] > 1e-4]
        print(f"Mean non-zero f: {np.nanmean(non_zero_f) if len(non_zero_f) > 0 else 0}")
        print(f"Max f: {np.max(data['f'])}")
else:
    print("No MATLAB file found.")
