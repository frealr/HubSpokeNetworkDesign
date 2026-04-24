import os
import sys
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import scipy.io as sio
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GAMS_EXE = '/opt/gams/gams49.6_linux_x64_64_sfx/gams'
GAMS_EXE = os.environ.get('GAMS_EXE', DEFAULT_GAMS_EXE)
os.chdir(SCRIPT_DIR)

def read_gams_csv_robust(file_path, symbol_name, max_retries=5, delay=1.0):
    for attempt in range(max_retries):
        try:
            # The file is an Excel file with a sheet for each symbol
            with pd.ExcelFile(file_path) as xls:
                df = pd.read_excel(xls, sheet_name=symbol_name)
            return df
        except Exception as e:
            if attempt == max_retries - 1:
                return pd.DataFrame()
            time.sleep(delay)

def read_excel_robust(file_path, sheet_name, max_retries=5, delay=1.0):
    for attempt in range(max_retries):
        try:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Warning: Failed to read {file_path} after {max_retries} attempts.")
                return pd.DataFrame()
            time.sleep(delay)

def write_matrix_csv(A, fn):
    df = pd.DataFrame(A, index=[f'i{i+1}' for i in range(A.shape[0])],
                      columns=[f'j{i+1}' for i in range(A.shape[1])])
    df.to_csv(fn)

def write_vector_csv(v, fn, prefix):
    v = np.ravel(v)
    df = pd.DataFrame({'idx': [f'{prefix}{i+1}' for i in range(len(v))], 'value': v})
    df.to_csv(fn, index=False)

def write_scalar_csv_append(name, val, fn):
    df = pd.DataFrame({'name': [name], 'value': [val]})
    df.to_csv(fn, mode='a', header=not os.path.exists(fn), index=False)

def write_gams_param_iii(filename, M):
    M = np.nan_to_num(M, nan=0.0, posinf=1e4, neginf=-1e4)
    n1, n2, n3 = M.shape
    with open(filename, 'w') as fid:
        for s in range(n1):
            for r in range(n2):
                for c in range(n3):
                    val = M[s, r, c]
                    fid.write(f'seg{s+1}.i{r+1}.i{c+1} {val:.12g}\n')

def write_gams_param_ii(filename, M):
    M = np.nan_to_num(M, nan=0.0, posinf=1e4, neginf=-1e4)
    n1, n2 = M.shape
    with open(filename, 'w') as fid:
        for r in range(n1):
            for c in range(n2):
                val = M[r, c]
                fid.write(f'i{r+1}.i{c+1} {val:.12g}\n')

def write_gams_param1d_full(filename, v):
    v = np.nan_to_num(np.ravel(v), nan=0.0, posinf=1e4, neginf=-1e4)
    with open(filename, 'w') as fid:
        for k in range(len(v)):
            fid.write(f'i{k+1} {v[k]:.12g}\n')

def split_and_accumarray(T, iU, jU, oU, dU):

    fij = np.zeros((len(iU), len(jU), len(oU), len(dU)))

    i_map = {v: k for k, v in enumerate(iU)}
    j_map = {v: k for k, v in enumerate(jU)}
    o_map = {v: k for k, v in enumerate(oU)}
    d_map = {v: k for k, v in enumerate(dU)}

    i_idx = T['i'].map(i_map).to_numpy()
    j_idx = T['j'].map(j_map).to_numpy()
    o_idx = T['o'].map(o_map).to_numpy()
    d_idx = T['d'].map(d_map).to_numpy()

    np.add.at(fij, (i_idx, j_idx, o_idx, d_idx), T['value'].to_numpy())

    return fij

def parameters_8node_network():
    n = 8
    n_airlines = 5
    candidates = np.ones((n, n)) - np.eye(n)
    
    # Needs actual context paths
    distance = pd.read_csv('distance.csv', header=None).values
    prices = pd.read_csv('prices.csv', header=None).values
    
    omega_t = -0.02
    omega_p = -0.02
    
    link_cost = 10.0 * distance
    np.fill_diagonal(link_cost, 1e4)
    link_cost[link_cost == 0] = 1e4
    
    station_cost = 3e3 * np.ones(n)
    hub_cost = 5e3 * np.ones(n)
    
    link_capacity_slope = 0.2 * link_cost
    station_capacity_slope = (5 * 5e2 + 4 * 50 * 8) * np.ones(n)
    
    demand = pd.read_csv('demand.csv', header=None).values

    demand = demand/365
    
    load_factor = 0.25 * np.ones(n)
    
    congestion_coef_stations = 0.1 * np.ones(n)
    congestion_coef_links = 0.1 * np.ones((n, n))
    takeoff_time = 20
    landing_time = 20
    taxi_time = 10
    cruise_time = 60 * distance / 800
    
    travel_time = cruise_time + takeoff_time + landing_time + taxi_time
    np.fill_diagonal(travel_time, 0)
    
    np.random.seed(123)
    p_escala = 0.4
    
    alt_utility = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            escala = np.random.rand(n_airlines) < p_escala
            alt_time_vec = travel_time[i, j] * (1 + 0.5 * escala) + 60 * escala
            alt_price_vec = prices[i, j] + 0.3 * prices[i, j] * (np.random.rand(n_airlines) - 0.5)
            
            alt_u = np.log(np.sum(np.exp(omega_p * alt_price_vec + omega_t * alt_time_vec))) - np.log(n_airlines)
            alt_utility[i, j] = alt_u
            alt_utility[j, i] = alt_u
            
    for i in range(n):
        alt_utility[i, i] = 0
        
    op_link_cost = 7600 * travel_time / 60
    
    a_nom = 171
    tau = 0.85
    eta = 0.3
    a_max = 1e9
    
    return (n, link_cost, station_cost, hub_cost, link_capacity_slope,
            station_capacity_slope, demand, prices, load_factor,
            op_link_cost, congestion_coef_stations, congestion_coef_links,
            travel_time, alt_utility, a_nom, tau, eta, a_max, candidates,
            omega_t, omega_p)

def parameters_6node_network():
    # Helper fallback from original file
    n = 8
    candidates = [[1, 2], [0, 2, 3], [0, 1, 3, 4], [1, 2, 4, 5], [2, 3, 5], [3, 4]]
    alt_cost = np.array([
        [0, 1.6, 0.8, 2, 1.6, 2.5],
        [2, 0, 0.9, 1.2, 1.5, 2.5],
        [1.5, 1.4, 0, 1.3, 0.9, 2],
        [1.9, 2, 1.9, 0, 1.8, 2],
        [3, 1.5, 2, 2, 0, 1.5],
        [2.1, 2.7, 2.2, 1, 1.5, 0]
    ])
    link_cost = (1e6 / (25 * 365.25)) * np.array([
        [0, 1.7, 2.7, 0, 0, 0],
        [1.7, 0, 2.1, 3, 0, 0],
        [2.7, 2.1, 0, 2.6, 1.7, 0],
        [0, 3, 2.6, 0, 2.8, 2.4],
        [0, 0, 1.7, 2.8, 0, 1.9],
        [0, 0, 0, 2.4, 1.9, 0]
    ])
    link_cost[link_cost == 0] = 1e4
    station_cost = (1e6 / (25 * 365.25)) * np.array([2, 3, 2.2, 3, 2.5, 1.3])
    hub_cost = (1e6 / (25 * 365.25)) * np.array([200, 100, 2, 300, 200, 100])
    link_capacity_slope = 0.04 * link_cost
    station_capacity_slope = 0.04 * station_cost
    demand = 1e3 * np.array([
        [0, 9, 26, 19, 13, 12],
        [11, 0, 14, 26, 7, 18],
        [30, 19, 0, 30, 24, 8],
        [21, 9, 11, 0, 22, 16],
        [14, 14, 8, 9, 0, 20],
        [26, 1, 22, 24, 13, 0]
    ])
    distance = 1e4 * np.ones((n, n))
    np.fill_diagonal(distance, 0)
    distance[0, 1] = 0.75
    distance[0, 2] = 0.7
    distance[1, 2] = 0.6
    distance[1, 3] = 1.1
    distance[2, 3] = 1.1
    distance[2, 4] = 0.5
    distance[3, 4] = 0.8
    distance[3, 5] = 0.7
    distance[4, 5] = 0.5
    for i in range(n):
        for j in range(i + 1, n):
            distance[j, i] = distance[i, j]
            
    load_factor = 0.25 * np.ones(n)
    op_link_cost = 4 * distance
    congestion_coef_stations = 0.1 * np.ones(n)
    congestion_coef_links = 0.1 * np.ones((n, n))
    prices = distance ** 0.7
    prices[prices > 10] = 1.2
    travel_time = 60 * distance / 30
    alt_time = 60 * alt_cost / 30
    alt_price = alt_cost ** 0.7
    
    a_nom = 588
    tau = 0.57
    eta = 0.25
    a_max = 1e9
    return (n, link_cost, station_cost, hub_cost, link_capacity_slope,
            station_capacity_slope, demand, prices, load_factor,
            op_link_cost, congestion_coef_stations, congestion_coef_links,
            travel_time, alt_time, alt_price, a_nom, tau, eta, a_max, candidates)

def get_budget(s, sh, a, n, station_cost, station_capacity_slope, hub_cost, link_cost, lam):
    budget = 0
    for i in range(n):
        if s[i] > 1e-2:
            budget += station_cost[i] + station_capacity_slope[i] * (s[i] + sh[i])
        if sh[i] > 1e-2:
            budget += lam * hub_cost[i]
    return budget

def get_entr_val(travel_time, prices, alt_time, alt_price, a_prim, delta_a,
                 s_prim, delta_s, fij, f, fext, demand, dm_pax, dm_op, n):
    pax_obj = 0
    for o in range(n):
        for d in range(n):
            pax_obj += 1e-6 * (demand[o, d] * np.sum((travel_time + prices) * fij[:, :, o, d]))
    pax_obj += 1e-6 * np.sum(demand * (alt_time + alt_price) * fext)
    # Using logical log
    f_log = f.copy()
    f_log[f_log <= 0] = 1e-12
    fext_log = fext.copy()
    fext_log[fext_log <= 0] = 1e-12
    pax_obj += 1e-6 * np.sum(demand * (-f_log * np.log(f_log) - f))
    pax_obj += 1e-6 * np.sum(demand * (-fext_log * np.log(fext_log) - fext))
    return 1e6 * pax_obj

def get_obj_val(op_link_cost, prices, a, f, demand):
    op_obj = np.sum(op_link_cost * a)          # costes operacionales
    pax_obj = np.sum(prices * demand * f)      # ingresos de pasajeros
    obj_val = -pax_obj + op_obj                # minimizar: -ingresos + costes
    return obj_val, pax_obj, op_obj

def get_linearization(n, nreg, alt_utility, vals_regs, n_airlines):
    dmax = np.zeros((nreg, n, n))
    dmin = np.zeros((nreg, n, n))
    lin_coef = np.zeros((nreg, n, n))
    bord = np.zeros((nreg, n, n))
    
    for o in range(n):
        for d in range(n):
            u = alt_utility[o, d]
            for r in range(nreg - 1):
                dmax[r, o, d] = min(0, u + np.log(n_airlines * vals_regs[r] / (1 - vals_regs[r])))
            dmax[nreg - 1, o, d] = 0
            dmin[0, o, d] = -3e1
            for r in range(1, nreg):
                dmin[r, o, d] = dmax[r - 1, o, d]
                
            for r in range(1, nreg - 1):
                if dmax[r, o, d] == dmin[r, o, d]:
                    lin_coef[r, o, d] = 0
                    bord[r, o, d] = vals_regs[r]
                else:
                    lin_coef[r, o, d] = (vals_regs[r] - vals_regs[r - 1]) / (dmax[r, o, d] - dmin[r, o, d])
                    bord[r, o, d] = vals_regs[r - 1]
            
            lin_coef[0, o, d] = vals_regs[0] / (dmax[0, o, d] - dmin[0, o, d])
            bord[0, o, d] = 0
            if dmin[nreg - 1, o, d] == 0:
                lin_coef[nreg - 1, o, d] = 0
            else:
                lin_coef[nreg - 1, o, d] = (1 - vals_regs[nreg - 2]) / (0 - dmin[nreg - 1, o, d])
            bord[nreg - 1, o, d] = vals_regs[nreg - 2]
            
    b = dmin
    return lin_coef, bord, b

def logit(x, omega_t, omega_p, time, price):
    return np.exp(x) / (np.exp(x) + np.exp(omega_t * time + omega_p * price))

def set_max_f(n, fij, n_airlines, travel_time, prices, alt_utility, omega_p, omega_t):
    cotas = np.zeros((n, n))
    for oo in range(n):
        for dd in range(n):
            utility = np.sum(travel_time[fij[:, :, oo, dd] > 1e-3]) * omega_t + prices[oo, dd] * omega_p
            cotas[oo, dd] = np.exp(utility) / (np.exp(utility) + n_airlines * np.exp(alt_utility[oo, dd]))
    write_gams_param_ii('./export_txt/f_bounds.txt', cotas)
    
def write_txt_param(name, val):
    with open(f"./export_txt/{name}.txt", 'w') as f:
        f.write(str(val))

def parse_matrix(output_csv, name, n):
    """Lee una matriz (n x n) desde el CSV/Excel de outputs de GAMS."""
    m_df = read_gams_csv_robust(output_csv, symbol_name=name)
    if m_df is None or len(m_df) == 0:
        return np.zeros((n, n))
    
    # If the dataframe has n rows and n+1 columns (first col is label), it's wide format
    if m_df.shape[1] >= n and m_df.shape[0] >= n:
        # Assuming first column is the row index (e.g., 'i1', 'i2', etc.)
        # and the next n columns are the values for j1, j2...
        # We can just extract the numeric values:
        m_vals = m_df.iloc[:, 1:n+1].values
        
        # Safely pad or crop to exactly (n, n)
        m = np.zeros((n, n))
        rows = min(n, m_vals.shape[0])
        cols = min(n, m_vals.shape[1])
        m[:rows, :cols] = m_vals[:rows, :cols]
        return m
    else:
        # Fallback to long format (i, j, value) in case we ever get that
        m_df = m_df.copy()
        if m_df.shape[1] >= 3:
            m_df.columns = ['i', 'j', 'value'] + list(m_df.columns[3:])
            try:
                m_df['i_idx'] = m_df['i'].astype(str).str.extract(r'(\d+)').astype(int) - 1
                m_df['j_idx'] = m_df['j'].astype(str).str.extract(r'(\d+)').astype(int) - 1
                m = np.zeros((n, n))
                valid_rows = (m_df['i_idx'] >= 0) & (m_df['i_idx'] < n) & (m_df['j_idx'] >= 0) & (m_df['j_idx'] < n)
                m[m_df.loc[valid_rows, 'i_idx'].values, m_df.loc[valid_rows, 'j_idx'].values] = m_df.loc[valid_rows, 'value'].values
                return m
            except Exception:
                return np.zeros((n, n))
            
        return np.zeros((n, n))

def compute_sim_MIP_entr(lam, beta, alfa, n, budget):
    # This was present but perhaps unused in primary loop; mapping for completeness
    pass

def compute_sim_cvx_blo(lam, alfa, n, budget, sh_prev_in):
    #revisar vs matlab
    (n, link_cost, station_cost, hub_cost, link_capacity_slope,
     station_capacity_slope, demand, prices, load_factor, op_link_cost,
     congestion_coef_stations, congestion_coef_links, travel_time,
     alt_utility, a_nom, tau, eta, a_max, candidasourcertes, omega_t, omega_p) = parameters_8node_network()
     
    #ok desde aqui
    niters = 20
    mu_alfa = 1e-7
    mu_beta = 2e-1

    debug = []
    
    alfa_od = np.ones((n, n))
    beta_od = np.ones((n, n))
    
    write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
    write_gams_param_ii('./export_txt/beta_od.txt', beta_od)
    gamma = 20
    
    logit_coef = 0.02
    n_airlines = 5
    #ok hasta aqui
    
    write_txt_param('niters', niters)
    write_txt_param('lam', lam)
    write_txt_param('alfa', alfa)
    write_txt_param('budget', budget)
    print('Este es el presupuesto:')
    print(budget)
    
    obj_hist_iters = {}
    s_traj = []
    sh_traj = []
    f_traj = []
    bliters = 30

    a_prev = 1e4 * np.ones((n, n))
    s_prev = 1e4 * np.ones(n)
    sh_prev = sh_prev_in.copy()
    
    comp_time = 0
    obj_val = 0
    obj_val_prev = 1e3
    
    _iter = 1
    while _iter <= niters:
        # alfa_od = ones(n);
        # beta_od = ones(n);
        write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
        write_gams_param_ii('./export_txt/beta_od.txt', beta_od)
        stop = 0
        obj_hist_this_iter = []
        for bliter in range(1, bliters + 1):
            if (abs((obj_val - obj_val_prev) / obj_val) <= 1e-3 if obj_val != 0 else False) and (bliter > 1):
                stop = 1
            elif stop == 0:
                write_gams_param_ii('./export_txt/a_prev.txt', a_prev)
                write_gams_param1d_full('./export_txt/s_prev.txt', s_prev)
                write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev)
                
                gmsFile = os.path.join(SCRIPT_DIR, 'cvx-ll.gms')
                gamsExe = GAMS_EXE
                cmd = f'"{gamsExe}" "{gmsFile}"'

                write_txt_param('current_iter', _iter)
                subprocess.run(cmd, shell=True, cwd=SCRIPT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                _df_t = read_gams_csv_robust('./output_all.xlsx', symbol_name='solver_time')
                comp_time += float(_df_t.columns[0]) if _df_t is not None and len(_df_t.columns) > 0 else 0.0
                
                sh_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='sh_level')
                sh = sh_df.to_numpy().flatten() if len(sh_df) > 0 else np.zeros(n)
                sh = np.maximum(sh, 1e-4)
                
                s_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='s_level')
                s = s_df.to_numpy().flatten() if len(s_df) > 0 else np.zeros(n)
                s = np.maximum(s, 1e-4)

                
                f = parse_matrix('./output_all.xlsx', 'f_level', n)
                print(f"f vale")
                print(f)
                
                a = parse_matrix('./output_all.xlsx', 'a_level', n)
                a = np.maximum(a, 1e-4)
                
                f = parse_matrix('./output_all.xlsx', 'f_level', n)
                fext = parse_matrix('./output_all.xlsx', 'fext_level', n)

                fij = np.zeros((n, n, n, n))
                if os.path.exists('fij_long.csv'):
                    T = pd.read_csv('fij_long.csv')
                    if len(T) > 0:
                        iU = T['i'].unique(); jU = T['j'].unique(); oU = T['o'].unique(); dU = T['d'].unique()
                        fij = split_and_accumarray(T, iU, jU, oU, dU)

                a[a < 1e-2] = 0
                f[f < 1e-2] = 0
                fij[fij < 1e-2] = 0
                fext[fext > 0.99] = 1

                a_ll = a.copy()
                f_ll = f.copy()
                s_ll = s.copy()
                sh_ll = sh.copy()
                
                grad_alfa_v = np.zeros((n, n))
                grad_beta_v = np.zeros((n, n))
                for oo in range(n):
                    for dd in range(n):
                        propio = ((prices[oo, dd] * demand[oo, dd]) ** beta_od[oo, dd]) * (logit_coef * prices[oo, dd] * f[oo, dd] + logit_coef * np.sum(fij[:, :, oo, dd] * travel_time))
                        externo = -((prices[oo, dd] * demand[oo, dd]) ** beta_od[oo, dd]) * (alt_utility[oo, dd] * fext[oo, dd])
                        log_propio = ((prices[oo, dd] * demand[oo, dd]) ** beta_od[oo, dd]) * (f[oo, dd] * (np.log(np.maximum(0, f[oo, dd]) + 1e-12) - 1))
                        log_ext = ((prices[oo, dd] * demand[oo, dd]) ** beta_od[oo, dd]) * (fext[oo, dd] * (np.log(np.maximum(0, fext[oo, dd]) / n_airlines + 1e-12) - 1))
            
                        grad_alfa_v[oo, dd] = propio + externo + log_propio + log_ext
                        grad_beta_v[oo, dd] = (alfa_od[oo, dd] + 1e-4) * ((beta_od[oo, dd] * demand[oo, dd] * prices[oo, dd]) ** (beta_od[oo, dd] - 1)) * \
                                              (logit_coef * prices[oo, dd] * f[oo, dd] + logit_coef * np.sum(fij[:, :, oo, dd] * travel_time) - \
                                              alt_utility[oo, dd] * fext[oo, dd] + f[oo, dd] * (np.log(np.maximum(0, f[oo, dd]) + 1e-12) - 1) + \
                                              fext[oo, dd] * (np.log(np.maximum(0, fext[oo, dd]) / n_airlines + 1e-12) - 1))
                                              
                used_budget = get_budget(s, sh, a, n, station_cost, station_capacity_slope, hub_cost, link_cost, lam)
                
                set_max_f(n, fij, n_airlines, travel_time, prices, alt_utility, omega_p, omega_t)
                
                gmsFile = os.path.join(SCRIPT_DIR, 'cvx-sl.gms')
                gamsExe = GAMS_EXE
                cmd = f'"{gamsExe}" "{gmsFile}"'
                subprocess.run(cmd, shell=True, cwd=SCRIPT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                _df_t2 = read_gams_csv_robust('./output_all.xlsx', symbol_name='solver_time')
                comp_time += float(_df_t2.columns[0]) if _df_t2 is not None and len(_df_t2.columns) > 0 else 0.0
                
                sh_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='sh_level')
                sh = sh_df.to_numpy().flatten() if len(sh_df) > 0 else np.zeros(n)
                sh = np.maximum(sh, 1e-4)
                
                s_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='s_level')
                s = s_df.to_numpy().flatten() if len(s_df) > 0 else np.zeros(n)
                s = np.maximum(s, 1e-4)
                
                a = parse_matrix('./output_all.xlsx', 'a_level', n)
                a = np.maximum(a, 1e-4)
                
                f = parse_matrix('./output_all.xlsx', 'f_level', n)
                fext = parse_matrix('./output_all.xlsx', 'fext_level', n)

                fij = np.zeros((n, n, n, n))
                if os.path.exists('fij_long.csv'):
                    T = pd.read_csv('fij_long.csv')
                    if len(T) > 0:
                        iU = T['i'].unique(); jU = T['j'].unique(); oU = T['o'].unique(); dU = T['d'].unique()
                        fij = split_and_accumarray(T, iU, jU, oU, dU)

                a[a < 1e-2] = 0
                f[f < 1e-2] = 0
                fext[fext > 0.99] = 1
                fij[fij < 1e-2] = 0
                
                obj_val_ll, pax_obj, op_obj = get_obj_val(op_link_cost, prices, a_ll, f_ll, demand)
                used_budget = get_budget(s, sh, a, n, station_cost, station_capacity_slope, hub_cost, link_cost, lam)
                
                print(obj_val_ll)
                
                f_sl = f.copy()
                a_sl = a.copy()
                
                grad_alfa_f = np.zeros((n, n))
                grad_beta_f = np.zeros((n, n))
                for oo in range(n):
                    for dd in range(n):
                        term_f = (logit_coef * prices[oo, dd] * f[oo, dd] + logit_coef * np.sum(fij[:, :, oo, dd] * travel_time) - \
                                  alt_utility[oo, dd] * fext[oo, dd] + f[oo, dd] * (np.log(f[oo, dd] + 1e-12) - 1) + \
                                  fext[oo, dd] * (np.log(fext[oo, dd] / n_airlines + 1e-12) - 1))
                                  
                        grad_alfa_f[oo, dd] = gamma * (((demand[oo, dd] * prices[oo, dd]) ** beta_od[oo, dd]) * term_f - grad_alfa_v[oo, dd])
                        grad_beta_f[oo, dd] = gamma * ((alfa_od[oo, dd] + 1e-4) * ((beta_od[oo, dd] * demand[oo, dd] * prices[oo, dd]) ** (beta_od[oo, dd] - 1)) * term_f - grad_beta_v[oo, dd])
                
                beta_od = beta_od - mu_beta * grad_beta_f
                alfa_od = alfa_od - mu_alfa * grad_alfa_f
                
                beta_od = np.maximum(0.5, beta_od)
                beta_od = np.minimum(2, beta_od)
                
                alfa_od = np.maximum(1, alfa_od)
                alfa_od = np.minimum(9, alfa_od)

                np.fill_diagonal(beta_od, 1)
                np.fill_diagonal(alfa_od, 1)
                
                write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
                write_gams_param_ii('./export_txt/beta_od.txt', beta_od)
                
                print('beta')
                print(beta_od)
                print('alfa')
                print(alfa_od)
                print('obj_val')
                print(obj_val_ll)
                
                obj_hist_this_iter.append(obj_val_ll)
                
                obj_val_prev = obj_val
                obj_val = obj_val_ll

                debug.append({
                    "iter": _iter,
                    "bliter": bliter,
                    "a": a.copy(),
                    "f": f.copy(),
                    "fext": fext.copy(),
                    "fij": fij.copy(),
                    "grad_alfa_v": grad_alfa_v.copy(),
                    "grad_beta_v": grad_beta_v.copy(),
                    "grad_alfa_f": grad_alfa_f.copy(),
                    "grad_beta_f": grad_beta_f.copy(),
                    "alfa_od": alfa_od.copy(),
                    "beta_od": beta_od.copy()
                })
                
        if (used_budget - budget) / budget < 0.05:
            if _iter < (niters - 1):
                print('cumplo presupuesto')
            # break

        obj_hist_iters[_iter] = obj_hist_this_iter.copy()

        s_traj.append(s_ll.copy())
        sh_traj.append(sh_ll.copy())
        f_traj.append(f_ll.copy())

        s_prev = s_ll.copy()
        sh_prev = sh_ll.copy()
        a_prev = a_ll.copy()
        stop = 0

        _iter += 1

    s_traj = np.array(s_traj) if s_traj else np.zeros((0, n))
    sh_traj = np.array(sh_traj) if sh_traj else np.zeros((0, n))
    f_traj = np.array(f_traj) if f_traj else np.zeros((0, n, n))

    sio.savemat("debug_python.mat", {"debug": debug})
    return s, sh, a, f, fext, fij, comp_time, used_budget, pax_obj, op_obj, obj_val_ll, alfa_od, beta_od, obj_hist_iters, s_traj, sh_traj, f_traj


def compute_sim_MIP(lam, beta, alfa, n, budget):
    (n, link_cost, station_cost, hub_cost, link_capacity_slope,
     station_capacity_slope, demand, prices, load_factor, op_link_cost,
     congestion_coef_stations, congestion_coef_links, travel_time,
     alt_utility, a_nom, tau, eta, a_max, candidasourcertes, omega_t, omega_p) = parameters_8node_network()
     
    dm_pax = 0.01
    dm_op = 0.008
    
    write_txt_param('lam', lam)
    write_txt_param('alfa', alfa)
    write_txt_param('beta', beta)
    write_txt_param('budget', budget)
    print(budget)
    
    a_prev = 1e4 * np.ones((n, n))
    s_prev = 1e4 * np.ones(n)
    
    write_gams_param_ii('./export_txt/a_prev.txt', a_prev)
    write_gams_param1d_full('./export_txt/s_prev.txt', s_prev)
    write_txt_param('dm_pax', dm_pax)
    write_txt_param('dm_op', dm_op)
    
    gmsFile = os.path.join(SCRIPT_DIR, 'mip.gms')
    gamsExe = GAMS_EXE
    cmd = f'"{gamsExe}" "{gmsFile}"'
    subprocess.run(cmd, shell=True, cwd=SCRIPT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Read outputs
    if os.path.exists('./output_all.xlsx'):
        sh_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='sh_level')
        sh = sh_df.iloc[:, -1].values.flatten() if len(sh_df)>0 else np.zeros(n)
        
        s_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='s_level')
        s = s_df.iloc[:, -1].values.flatten() if len(s_df)>0 else np.zeros(n)
        sprim = s.copy()
        deltas = np.zeros(n)
        a = parse_matrix('./output_all.xlsx', 'a_level', n)
        f = parse_matrix('./output_all.xlsx', 'f_level', n)
        fext = parse_matrix('./output_all.xlsx', 'fext_level', n)
        
        mipgap_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='mip_opt_gap')
        mipgap = mipgap_df.iloc[:, -1].values.flatten() if len(mipgap_df)>0 else [0]
        
        ctime_vals = read_gams_csv_robust('./output_all.xlsx', symbol_name='solver_time').values.flatten()
        comp_time = ctime_vals[-1] if len(ctime_vals) > 0 else 0
    else:
        sh = np.zeros(n); s = np.zeros(n); sprim = np.zeros(n); deltas = np.zeros(n); a = np.zeros((n,n)); f = np.zeros((n,n)); fext = np.zeros((n,n)); mipgap = [0]; comp_time = 0
        
    if os.path.exists('fij_long.csv'):
        T = pd.read_csv('fij_long.csv')
        iU = T['i'].unique(); jU = T['j'].unique(); oU = T['o'].unique(); dU = T['d'].unique()
        fij = split_and_accumarray(T, iU, jU, oU, dU)
    else:
        fij = np.zeros((n, n, n, n))
        
    obj_val, pax_obj, op_obj = get_obj_val(op_link_cost, prices, a, f, demand)
    req_bud = budget
    budget_used = get_budget(s, sh, a, n, station_cost, station_capacity_slope, hub_cost, link_cost, lam)
    
    filename = f'./8node_hs_prueba_v0/2h_budget={req_bud}_lam={lam}.mat'
    if not os.path.exists('./8node_hs_prueba_v0'):
        os.makedirs('./8node_hs_prueba_v0')
    sio.savemat(filename, {'s': s, 'sprim': sprim, 'deltas': deltas, 'a': a, 'f': f, 'fext': fext, 'fij': fij,
                           'comp_time': comp_time, 'budget': budget_used, 'pax_obj': pax_obj, 'op_obj': op_obj,
                           'obj_val': obj_val, 'mipgap': mipgap})
                           
    return s, sh, a, f, fext, fij


if __name__ == '__main__':
    # Initialize basic setup
    basedir = 'export_csv'
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    if not os.path.exists('export_txt'):
        os.makedirs('export_txt')
        
    (n, link_cost, station_cost, hub_cost, link_capacity_slope,
     station_capacity_slope, demand, prices, load_factor, op_link_cost,
     congestion_coef_stations, congestion_coef_links, travel_time, alt_utility,
     a_nom, tau, eta, a_max, candidasourcertes, omega_t, omega_p) = parameters_8node_network()
     
    # demand ya viene dividida por 365 desde parameters_8node_network()
    M = 1e4
    nreg = 10
    eps = 1e-3
    vals_regs = np.linspace(0.005, 0.995, nreg - 1)
    n_airlines = 5
    
    lin_coef, bord, b = get_linearization(n, nreg, alt_utility, vals_regs, n_airlines)
    
    # candidasourcertes ya es una matriz (n x n) de 0s y 1s desde parameters_8node_network()
    candidates = np.zeros((n, n))
    candidates[candidasourcertes > 0] = 1
        
    alfa_od = np.ones((n, n))
    beta_od = np.ones((n, n))
    gamma = 20
    
    write_gams_param_iii('./export_txt/lin_coef.txt', lin_coef)
    write_gams_param_iii('./export_txt/b.txt', b)
    write_gams_param_iii('./export_txt/bord.txt', bord)
    
    write_gams_param_ii('./export_txt/demand.txt', demand)
    write_gams_param_ii('./export_txt/travel_time.txt', travel_time)
    write_gams_param_ii('./export_txt/alt_utility.txt', alt_utility)
    write_gams_param_ii('./export_txt/link_cost.txt', link_cost)
    write_gams_param_ii('./export_txt/link_capacity_slope.txt', link_capacity_slope)
    write_gams_param_ii('./export_txt/prices.txt', prices)
    write_gams_param_ii('./export_txt/op_link_cost.txt', op_link_cost)
    write_gams_param_ii('./export_txt/candidates.txt', candidates)
    write_gams_param_ii('./export_txt/congestion_coefs_links.txt', congestion_coef_links)
    
    write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
    write_gams_param_ii('./export_txt/beta_od.txt', beta_od)
    write_txt_param('gamma', gamma)
    
    write_gams_param1d_full('./export_txt/station_cost.txt', station_cost)
    write_gams_param1d_full('./export_txt/hub_cost.txt', hub_cost)
    write_gams_param1d_full('./export_txt/station_capacity_slope.txt', station_capacity_slope)
    write_gams_param1d_full('./export_txt/congestion_coefs_stations.txt', congestion_coef_stations)
    
    a_prev = 1e4 * np.ones((n, n))
    s_prev = 1e4 * np.ones(n)
    sh_prev = 1e-3 * s_prev
    
    write_gams_param_ii('./export_txt/a_prev.txt', a_prev)
    write_gams_param1d_full('./export_txt/s_prev.txt', s_prev)
    write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev)
    
    # Run the configurations
    alfa = 0.5
    budgets = [3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5]
    lam = 4
    
    #for bud in budgets:
    #    compute_sim_MIP(lam, 1, alfa, n, bud)
        
    for bud in budgets:
        filename = f'./8node_hs_prueba_v0/budget={bud}_lam={lam}.mat'
        if os.path.exists(filename):
            data = sio.loadmat(filename)
            print(bud, data['obj_val'][0][0])
            
    alfas = [0.1]
    write_txt_param('gamma', 20)

    if not os.path.exists('./8node_hs_prueba_v0_blo'):
        os.makedirs('./8node_hs_prueba_v0_blo')

    for bud in budgets:
        for al in alfas:
            alfa_od = np.ones((n, n))
            beta_od = np.ones((n, n))
            write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
            write_gams_param_ii('./export_txt/beta_od.txt', beta_od)

            sh_prev_list = 1e4 * np.ones((n + 1, n))
            for idx_ms in range(n):
                sh_prev_list[idx_ms + 1, idx_ms] = 2

            best_obj_ms = 1
            best_res_ms = {'f': np.zeros((n, n)), 'a': np.zeros((n, n))}
            total_comp_time_ms = 0

            for start_idx in range(n + 1):
                sh_prev_in = sh_prev_list[start_idx, :]
                print('empiezo multistart con sh = ')
                print(sh_prev_in)
                res = compute_sim_cvx_blo(lam, al, n, bud, sh_prev_in)
                (s_curr, sh_curr, a_curr, f_curr, fext_curr, fij_curr, comp_time_curr, used_budget_curr, pax_obj_curr, op_obj_curr, obj_val_ll_curr, alfa_od_curr, beta_od_curr, obj_hist_iters_curr, s_traj_curr, sh_traj_curr, f_traj_curr) = res
                total_comp_time_ms += comp_time_curr

                f_curr[np.abs(f_curr - best_res_ms['f']) < 2e-2] = best_res_ms['f'][np.abs(f_curr - best_res_ms['f']) < 2e-2]
                print(f_curr)
                a_curr[np.abs(a_curr - best_res_ms['a']) < 5e-2] = best_res_ms['a'][np.abs(a_curr - best_res_ms['a']) < 5e-2]
                print(a_curr)

                obj_curr = np.sum(f_curr * demand * prices) - np.sum(op_link_cost * a_curr)
                print(obj_curr)

                if (obj_curr > best_obj_ms) and (np.sum(s_curr) > 2e-2):
                    print('el mejor hasta ahora')
                    print('f sale:')
                    print(f_curr)
                    best_obj_ms = obj_curr
                    best_res_ms.update({'s': s_curr, 'sh': sh_curr, 'a': a_curr, 'f': f_curr, 'fext': fext_curr, 'fij': fij_curr,
                                        'used_budget': used_budget_curr, 'pax_obj': pax_obj_curr,
                                        'op_obj': op_obj_curr, 'obj_val_ll': obj_val_ll_curr, 'alfa_od': alfa_od_curr,
                                        'beta_od': beta_od_curr, 'obj_hist_iters': obj_hist_iters_curr,
                                        's_traj': s_traj_curr, 'sh_traj': sh_traj_curr, 'f_traj': f_traj_curr})

            # Tiempo total de cómputo = suma de todas las inicializaciones
            best_res_ms['comp_time'] = total_comp_time_ms

            # Convertir obj_hist_iters (dict Python) a cell array para savemat
            if 'obj_hist_iters' in best_res_ms:
                ohi = best_res_ms['obj_hist_iters']
                n_done = len(ohi)
                obj_hist_cell = np.empty((1, n_done), dtype=object)
                for k in range(n_done):
                    obj_hist_cell[0, k] = np.array(ohi.get(k + 1, []))
                best_res_ms['obj_hist_iters'] = obj_hist_cell

            mu_alfa = 1e-7
            mu_beta = 2e-1
            filename = f'./8node_hs_prueba_v0_blo/bud={bud}_lam={lam}_alfa={al}_mu_al={mu_alfa}_mu_bet={mu_beta}_python-euler.mat'
            sio.savemat(filename, best_res_ms)
