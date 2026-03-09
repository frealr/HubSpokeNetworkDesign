import os
import sys
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
import scipy.io as sio
import time

def read_gams_csv_robust(file_path, symbol_name, max_retries=5, delay=1.0):
    for attempt in range(max_retries):
        try:
            df = pd.read_csv(file_path)
            # Find the rows associated with the symbol. The symbol name is usually in the first column or 'Dim1' if header
            # Or GAMS CSV writes them concatenated. The format usually has the Symbol name in the first column.
            # Let's inspect typical structure or filter the first column blindly if it equals the symbol.
            # actually if header=True was used, columns are ['name', 'dim1', ..., 'value']
            # if name is not there, check first col.
            if 'name' in df.columns:
                sub_df = df[df['name'] == symbol_name]
            else:
                sub_df = df[df.iloc[:, 0] == symbol_name]
            
            # Values are in the rest
            return sub_df.iloc[:, 1:]
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

def split_and_accumarray(fij_df, iU, jU, oU, dU):
    fij = np.zeros((len(iU), len(jU), len(oU), len(dU)))
    i_map = {val: idx for idx, val in enumerate(iU)}
    j_map = {val: idx for idx, val in enumerate(jU)}
    o_map = {val: idx for idx, val in enumerate(oU)}
    d_map = {val: idx for idx, val in enumerate(dU)}
    for _, row in fij_df.iterrows():
        i_idx = i_map[row['i']]
        j_idx = j_map[row['j']]
        o_idx = o_map[row['o']]
        d_idx = d_map[row['d']]
        fij[i_idx, j_idx, o_idx, d_idx] += row['value']
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
        for j in range(n):
            if a[i, j] > 1e-2:
                budget += link_cost[i, j]
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
    n = a.shape[0]
    pax_obj = 0
    op_obj = np.sum(op_link_cost * a)
    for o in range(n):
        for d in range(n):
            pax_obj -= prices[o, d] * demand[o, d] * f[o, d]
    return pax_obj + op_obj, pax_obj, op_obj

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

def compute_sim_MIP_entr(lam, beta, alfa, n, budget):
    # This was present but perhaps unused in primary loop; mapping for completeness
    pass

def compute_sim_cvx_blo(lam, alfa, n, budget):
    (n, link_cost, station_cost, hub_cost, link_capacity_slope,
     station_capacity_slope, demand, prices, load_factor, op_link_cost,
     congestion_coef_stations, congestion_coef_links, travel_time,
     alt_utility, a_nom, tau, eta, a_max, candidasourcertes, omega_t, omega_p) = parameters_8node_network()
     
    niters = 10
    mu_alfa = 1e-7
    mu_beta = 2e-1
    
    alfa_od = np.ones((n, n))
    beta_od = np.ones((n, n))
    
    write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
    write_gams_param_ii('./export_txt/beta_od.txt', beta_od)
    gamma = 20
    
    logit_coef = 0.02
    n_airlines = 5
    
    write_txt_param('niters', niters)
    write_txt_param('lam', lam)
    write_txt_param('alfa', alfa)
    write_txt_param('budget', budget)
    print('Este es el presupuesto:')
    print(budget)
    
    obj_hist = np.zeros(30)
    bliters = 30
    
    a_prev = 1e4 * np.ones((n, n))
    s_prev = 1e4 * np.ones(n)
    sh_prev = s_prev.copy()
    
    comp_time = 0
    obj_val = 0
    obj_val_prev = 1e3
    
    for _iter in range(1, niters + 1):
        write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
        write_gams_param_ii('./export_txt/beta_od.txt', beta_od)
        stop = 0
        
        for bliter in range(1, bliters + 1):
            if abs((obj_val - obj_val_prev) / (obj_val + 1e-12)) <= 1e-3 and bliter > 1:
                stop = 1
            elif stop == 0:
                write_gams_param_ii('./export_txt/a_prev.txt', a_prev)
                write_gams_param1d_full('./export_txt/s_prev.txt', s_prev)
                write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev)
                
                gmsFile = r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain\cvx-ll.gms'
                gamsExe = r'C:\GAMS\50\gams.exe'
                cmd = f'"{gamsExe}" "{gmsFile}"'
                
                write_txt_param('current_iter', 1)
                subprocess.run(cmd, shell=True, cwd=r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain')
                
                # Reading results
                if os.path.exists('./output_all.csv'):
                    ctime_vals = read_gams_csv_robust('./output_all.csv', symbol_name='solver_time').values.flatten()
                    if len(ctime_vals) > 0: comp_time += ctime_vals[-1] # Usually last column is value
                    
                    sh_df = read_gams_csv_robust('./output_all.csv', symbol_name='sh_level')
                    sh = np.maximum(sh_df.iloc[:, -1].values.flatten() if len(sh_df)>0 else np.zeros(n), 1e-4)
                    
                    s_df = read_gams_csv_robust('./output_all.csv', symbol_name='s_level')
                    s = np.maximum(s_df.iloc[:, -1].values.flatten() if len(s_df)>0 else np.zeros(n), 1e-4)
                    
                    # For matrices, the CSV is in long format: i, j, value. We need to pivot or reshape
                    # But Python expects a full (n,n) array. We can use pivot to guarantee shape
                    def parse_matrix(name, n):
                        m_df = read_gams_csv_robust('./output_all.csv', symbol_name=name)
                        if len(m_df) == 0: return np.zeros((n,n))
                        # m_df columns: [dim1, dim2, value]
                        m_df.columns = ['i', 'j', 'value']
                        # map 'i1' back to 0-indexed int
                        try:
                            m_df['i_idx'] = m_df['i'].str.extract(r'(\d+)').astype(int) - 1
                            m_df['j_idx'] = m_df['j'].str.extract(r'(\d+)').astype(int) - 1
                            m = np.zeros((n,n))
                            m[m_df['i_idx'].values, m_df['j_idx'].values] = m_df['value'].values
                            return m
                        except:
                            return np.zeros((n,n))
                            
                    f = parse_matrix('f_level', n)
                    a = np.maximum(parse_matrix('a_level', n), 1e-4)
                    fext = parse_matrix('fext_level', n)
                else:
                    # fallback if file doesn't exist
                    sh = np.ones(n)*1e-4; s = np.ones(n)*1e-4; f = np.zeros((n,n)); a = np.ones((n,n))*1e-4; fext = np.zeros((n,n))
                    
                if os.path.exists('fij_long.csv'):
                    T = pd.read_csv('fij_long.csv')
                    iU = T['i'].unique(); jU = T['j'].unique(); oU = T['o'].unique(); dU = T['d'].unique()
                    fij = split_and_accumarray(T, iU, jU, oU, dU)
                else:
                    fij = np.zeros((n, n, n, n))
                
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
                        if demand[oo, dd] < 1e-3 or prices[oo, dd] < 1e-3: continue
                        propio = ((prices[oo, dd] * demand[oo, dd] + 1e-4) ** beta_od[oo, dd]) * (logit_coef * prices[oo, dd] * f[oo, dd] + logit_coef * np.sum(fij[:, :, oo, dd] * travel_time))
                        externo = -((prices[oo, dd] * demand[oo, dd] + 1e-4) ** beta_od[oo, dd]) * (alt_utility[oo, dd] * fext[oo, dd])
                        log_propio = ((prices[oo, dd] * demand[oo, dd] + 1e-4) ** beta_od[oo, dd]) * (f[oo, dd] * (np.log(max(0, f[oo, dd]) + 1e-12) - 1))
                        log_ext = ((prices[oo, dd] * demand[oo, dd] + 1e-4) ** beta_od[oo, dd]) * (fext[oo, dd] * (np.log(max(0, fext[oo, dd]) / n_airlines + 1e-12) - 1))
                        
                        grad_alfa_v[oo, dd] = propio + externo + log_propio + log_ext
                        grad_beta_v[oo, dd] = (alfa_od[oo, dd] + 1e-4) * ((beta_od[oo, dd] * demand[oo, dd] * prices[oo, dd] + 1e-4) ** (beta_od[oo, dd] - 1)) * \
                                              (logit_coef * prices[oo, dd] * f[oo, dd] + logit_coef * np.sum(fij[:, :, oo, dd] * travel_time) - \
                                              alt_utility[oo, dd] * fext[oo, dd] + f[oo, dd] * (np.log(max(0, f[oo, dd]) + 1e-12) - 1) + \
                                              fext[oo, dd] * (np.log(max(0, fext[oo, dd]) / n_airlines + 1e-12) - 1))
                
                used_budget = get_budget(s, sh, a, n, station_cost, station_capacity_slope, hub_cost, link_cost, lam)
                set_max_f(n, fij, n_airlines, travel_time, prices, alt_utility, omega_p, omega_t)
                
                gmsFile = r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain\cvx-sl.gms'
                cmd = f'"{gamsExe}" "{gmsFile}"'
                subprocess.run(cmd, shell=True, cwd=r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain')
                
                # Reading sl results
                if os.path.exists('./output_all.csv'):
                    ctime_vals = read_gams_csv_robust('./output_all.csv', symbol_name='solver_time').values.flatten()
                    if len(ctime_vals) > 0: comp_time += ctime_vals[-1]
                    
                    sh_df = read_gams_csv_robust('./output_all.csv', symbol_name='sh_level')
                    sh = np.maximum(sh_df.iloc[:, -1].values.flatten() if len(sh_df)>0 else np.zeros(n), 1e-4)
                    
                    s_df = read_gams_csv_robust('./output_all.csv', symbol_name='s_level')
                    s = np.maximum(s_df.iloc[:, -1].values.flatten() if len(s_df)>0 else np.zeros(n), 1e-4)
                    
                    f = parse_matrix('f_level', n)
                    a = np.maximum(parse_matrix('a_level', n), 1e-4)
                    fext = parse_matrix('fext_level', n)
                
                if os.path.exists('fij_long.csv'):
                    T = pd.read_csv('fij_long.csv')
                    iU = T['i'].unique(); jU = T['j'].unique(); oU = T['o'].unique(); dU = T['d'].unique()
                    fij = split_and_accumarray(T, iU, jU, oU, dU)
                else:
                    fij = np.zeros((n, n, n, n))
                    
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
                        if demand[oo, dd] < 1e-3 or prices[oo, dd] < 1e-3: continue
                        term_f = (logit_coef * prices[oo, dd] * f[oo, dd] + logit_coef * np.sum(fij[:, :, oo, dd] * travel_time) - \
                                  alt_utility[oo, dd] * fext[oo, dd] + f[oo, dd] * (np.log(f[oo, dd] + 1e-12) - 1) + \
                                  fext[oo, dd] * (np.log(fext[oo, dd] / n_airlines + 1e-12) - 1))
                                  
                        grad_alfa_f[oo, dd] = gamma * (((demand[oo, dd] * prices[oo, dd] + 1e-4) ** beta_od[oo, dd]) * term_f - grad_alfa_v[oo, dd])
                        grad_beta_f[oo, dd] = gamma * ((alfa_od[oo, dd] + 1e-4) * ((beta_od[oo, dd] * demand[oo, dd] * prices[oo, dd] + 1e-4) ** (beta_od[oo, dd] - 1)) * term_f - grad_beta_v[oo, dd])
                
                beta_od = beta_od - mu_beta * grad_beta_f
                alfa_od = alfa_od - mu_alfa * grad_alfa_f
                
                beta_od = np.clip(beta_od, 0.5, 2)
                alfa_od = np.clip(alfa_od, 1, 9)
                
                write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
                write_gams_param_ii('./export_txt/beta_od.txt', beta_od)
                
                print('beta', beta_od)
                print('alfa', alfa_od)
                print('obj_val', obj_val_ll)
                
                obj_hist[bliter - 1] = obj_val_ll
                obj_val_prev = obj_val
                obj_val = obj_val_ll
                
        if abs(used_budget - budget) / budget < 0.05:
            print('cumplo presupuesto')
            break
            
        s_prev = s_ll
        sh_prev = sh_ll
        a_prev = a_ll
        stop = 0

    a_prev = 1e4 * np.ones((n, n))
    s_prev = 0.1 * np.ones(n)
    sh_prev = s_prev.copy()
    
    gmsFile = r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain\cvx-mip.gms'
    gamsExe = r'C:\GAMS\50\gams.exe'
    cmd = f'"{gamsExe}" "{gmsFile}"'
    
    for _iter in range(1, niters + 1):
        write_gams_param_ii('./export_txt/a_prev.txt', a_prev)
        write_gams_param1d_full('./export_txt/s_prev.txt', s_prev)
        write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev)
        write_txt_param('current_iter', _iter)
        
        subprocess.run(cmd, shell=True, cwd=r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain')
        
        if os.path.exists('./output_all.csv'):
            ctime_vals = read_gams_csv_robust('./output_all.csv', symbol_name='solver_time').values.flatten()
            if len(ctime_vals) > 0: comp_time += ctime_vals[-1]
            
            sh_df = read_gams_csv_robust('./output_all.csv', symbol_name='sh_level')
            sh = np.maximum(sh_df.iloc[:, -1].values.flatten() if len(sh_df)>0 else np.zeros(n), 1e-4)
            
            s_df = read_gams_csv_robust('./output_all.csv', symbol_name='s_level')
            s = np.maximum(s_df.iloc[:, -1].values.flatten() if len(s_df)>0 else np.zeros(n), 1e-4)
            
            f = parse_matrix('f_level', n)
            a = np.maximum(parse_matrix('a_level', n), 1e-4)
            fext = parse_matrix('fext_level', n)
        
        if os.path.exists('fij_long.csv'):
            T = pd.read_csv('fij_long.csv')
            iU = T['i'].unique(); jU = T['j'].unique(); oU = T['o'].unique(); dU = T['d'].unique()
            fij = split_and_accumarray(T, iU, jU, oU, dU)
        else:
            fij = np.zeros((n, n, n, n))
            
        a[a < 1e-2] = 0
        f[f < 1e-2] = 0
        fij[fij < 1e-2] = 0
        fext[fext > 0.99] = 1
        a_prev = a.copy()
        s_prev = s.copy()
        sh_prev = sh.copy()
        
    filename = f'./8node_hs_prueba_v0_blo/bud={budget}_lam={lam}_alfa={alfa}_mu_al={mu_alfa}_mu_bet={mu_beta}_python.mat'
    if not os.path.exists('./8node_hs_prueba_v0_blo'):
        os.makedirs('./8node_hs_prueba_v0_blo')
    sio.savemat(filename, {'s': s, 'sh': sh, 'a': a, 'f': f, 'fext': fext, 'fij': fij,
                           'comp_time': comp_time, 'used_budget': used_budget,
                           'pax_obj': pax_obj, 'op_obj': op_obj, 'obj_val_ll': obj_val_ll,
                           'alfa_od': alfa_od, 'beta_od': beta_od, 'obj_hist': obj_hist})
    return s, sh, a, f, fext, fij


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
    
    gmsFile = r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain\mip.gms'
    gamsExe = r'C:\GAMS\50\gams.exe'
    cmd = f'"{gamsExe}" "{gmsFile}"'
    subprocess.run(cmd, shell=True, cwd=r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\8node_spain')
    
    # Read outputs
    if os.path.exists('./output_all.csv'):
        sh_df = read_gams_csv_robust('./output_all.csv', symbol_name='sh_level')
        sh = sh_df.iloc[:, -1].values.flatten() if len(sh_df)>0 else np.zeros(n)
        
        s_df = read_gams_csv_robust('./output_all.csv', symbol_name='s_level')
        s = s_df.iloc[:, -1].values.flatten() if len(s_df)>0 else np.zeros(n)
        sprim = s.copy()
        deltas = np.zeros(n)
        a = parse_matrix('a_level', n)
        f = parse_matrix('f_level', n)
        fext = parse_matrix('fext_level', n)
        
        mipgap_df = read_gams_csv_robust('./output_all.csv', symbol_name='mip_opt_gap')
        mipgap = mipgap_df.iloc[:, -1].values.flatten() if len(mipgap_df)>0 else [0]
        
        ctime_vals = read_gams_csv_robust('./output_all.csv', symbol_name='solver_time').values.flatten()
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
     
    demand = demand / 365
    M = 1e4
    nreg = 10
    eps = 1e-3
    vals_regs = np.linspace(0.005, 0.995, nreg - 1)
    n_airlines = 5
    
    lin_coef, bord, b = get_linearization(n, nreg, alt_utility, vals_regs, n_airlines)
    
    candidates = np.zeros((n, n))
    for i in range(n):
        candidates[i, candidasourcertes[i, :] > 0] = 1
        
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
    budgets = [3e4]
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
    for bud in budgets:
        for al in alfas:
            alfa_od = np.ones((n, n))
            beta_od = np.ones((n, n))
            write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
            write_gams_param_ii('./export_txt/beta_od.txt', beta_od)
            compute_sim_cvx_blo(lam, al, n, bud)
            
    best_obj = 1e3 * np.ones(len(budgets))
    best_alfa = np.zeros(len(budgets))
    used_bud = np.zeros(len(budgets))
    for bb, bud in enumerate(budgets):
        for al in alfas:
            filename = f'./8node_hs_prueba_v0_blo/bud={bud}_lam={lam}_alfa={al}.mat'
            # (Matches generated output depending on mu_alfa/mu_beta config logic)
            # In MATLAB, the script calls `load` without matching mu_a/b, so it assumes unique.
            # Here we just iterate directly if possible.
            import glob
            files = glob.glob(f'./8node_hs_prueba_v0_blo/bud={bud}_lam={lam}_alfa={al}*.mat')
            for f in files:
                data = sio.loadmat(f)
                obj_val_ll = data['obj_val_ll'][0][0]
                ubb = data['used_budget'][0][0]
                if obj_val_ll < best_obj[bb]:
                    best_obj[bb] = obj_val_ll
                    best_alfa[bb] = al
                    used_bud[bb] = ubb
