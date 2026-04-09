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
            with pd.ExcelFile(file_path) as xls:
                df = pd.read_excel(xls, sheet_name=symbol_name)
            return df
        except Exception as e:
            if attempt == max_retries - 1:
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

    if len(T) > 0:
        i_idx = T['i'].map(i_map).to_numpy()
        j_idx = T['j'].map(j_map).to_numpy()
        o_idx = T['o'].map(o_map).to_numpy()
        d_idx = T['d'].map(d_map).to_numpy()
        np.add.at(fij, (i_idx, j_idx, o_idx, d_idx), T['value'].to_numpy())
    return fij

def write_txt_param(name, val):
    with open(f"./export_txt/{name}.txt", 'w') as f:
        f.write(str(val))

def parse_matrix(output_csv, name, n):
    m_df = read_gams_csv_robust(output_csv, symbol_name=name)
    if m_df is None or len(m_df) == 0:
        return np.zeros((n, n))
    
    if m_df.shape[1] >= n and m_df.shape[0] >= n:
        m_vals = m_df.iloc[:, 1:n+1].values
        m = np.zeros((n, n))
        rows = min(n, m_vals.shape[0])
        cols = min(n, m_vals.shape[1])
        m[:rows, :cols] = m_vals[:rows, :cols]
        return m
    else:
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

def get_budget(s, sh, a, n, station_cost, station_capacity_slope, hub_cost, link_cost, lam):
    budget = 0
    for i in range(n):
        if s[i] > 1e-2:
            budget += station_cost[i] + station_capacity_slope[i] * (s[i] + sh[i])
        if sh[i] > 1e-2:
            budget += lam * hub_cost[i]
    return budget

def get_obj_val(op_link_cost, prices, a, f, demand):
    op_obj = np.sum(op_link_cost * a)
    pax_obj = np.sum(prices * demand * f)
    obj_val = -pax_obj + op_obj
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

def parameters_4node_network():
    n = 4
    n_airlines = 5
    candidates = np.ones((n, n)) - np.eye(n)
    
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

def compute_sim_cvx_blo(lam, alfa, n, budget, mu_alfa, mu_beta, sh_prev_in):
    (n, link_cost, station_cost, hub_cost, link_capacity_slope,
     station_capacity_slope, demand, prices, load_factor, op_link_cost,
     congestion_coef_stations, congestion_coef_links, travel_time,
     alt_utility, a_nom, tau, eta, a_max, candidates, omega_t, omega_p) = parameters_4node_network()
     
    niters = 40

    debug = []
    
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
    print('este es el presupuesto:')
    print(budget)
    
    obj_hist = np.zeros(30)
    bliters = 30
    
    a_prev = 1e4 * np.ones((n, n))
    s_prev = 1e4 * np.ones(n)
    sh_prev = sh_prev_in.copy()
    
    comp_time = 0
    obj_val = 0
    obj_val_prev = 1e3
    
    _iter = 1
    while _iter <= niters:
        write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
        write_gams_param_ii('./export_txt/beta_od.txt', beta_od)
        stop = 0
        for bliter in range(1, bliters + 1):
            if (abs((obj_val - obj_val_prev) / (obj_val + 1e-4)) <= 1e-3) and (bliter > 1):
                stop = 1
            elif stop == 0:
                write_gams_param_ii('./export_txt/a_prev.txt', a_prev)
                write_gams_param1d_full('./export_txt/s_prev.txt', s_prev)
                write_gams_param1d_full('./export_txt/sh_prev.txt', sh_prev)
                
                gmsFile = r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain\cvx-ll.gms'
                gamsExe = r'C:\GAMS\50\gams.exe'
                cmd = f'"{gamsExe}" "{gmsFile}"'
                
                write_txt_param('current_iter', _iter)
                subprocess.run(cmd, shell=True, cwd=r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                ctime_vals = read_gams_csv_robust('./output_all.xlsx', symbol_name='solver_time')
                if ctime_vals is not None and len(ctime_vals) > 0:
                    comp_time += ctime_vals.to_numpy().flatten()[-1]
                
                sh_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='sh_level')
                sh = sh_df.to_numpy().flatten() if sh_df is not None and len(sh_df) > 0 else np.zeros(n)
                sh = np.maximum(sh, 1e-4)
                
                s_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='s_level')
                s = s_df.to_numpy().flatten() if s_df is not None and len(s_df) > 0 else np.zeros(n)
                s = np.maximum(s, 1e-4)
                
                print(s)
                print(sh)

                f = parse_matrix('./output_all.xlsx', 'f_level', n)
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
                fext_ll = fext.copy()
                fij_ll = fij.copy()
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
                
                gmsFile = r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain\cvx-sl.gms'
                cmd = f'"{gamsExe}" "{gmsFile}"'
                subprocess.run(cmd, shell=True, cwd=r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                ctime_vals = read_gams_csv_robust('./output_all.xlsx', symbol_name='solver_time')
                if ctime_vals is not None and len(ctime_vals) > 0:
                    comp_time += ctime_vals.to_numpy().flatten()[-1]
                
                sh_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='sh_level')
                sh = sh_df.to_numpy().flatten() if sh_df is not None and len(sh_df) > 0 else np.zeros(n)
                sh = np.maximum(sh, 1e-4)
                
                s_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='s_level')
                s = s_df.to_numpy().flatten() if s_df is not None and len(s_df) > 0 else np.zeros(n)
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
                beta_od = np.minimum(2.1, beta_od)
                
                alfa_od = np.maximum(1, alfa_od)
                alfa_od = np.minimum(9, alfa_od)

                np.fill_diagonal(beta_od, 1)
                np.fill_diagonal(alfa_od, 1)
                
                write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
                write_gams_param_ii('./export_txt/beta_od.txt', beta_od)
                
                obj_hist[bliter - 1] = obj_val_ll
                
                obj_val_prev = obj_val
                obj_val = obj_val_ll

                debug.append({
                    "a": a_ll.copy(),
                    "f": f_ll.copy(),
                    "fext": fext_ll.copy(),
                    "fij": fij_ll.copy(),
                    "grad_alfa_v": grad_alfa_v.copy(),
                    "grad_beta_v": grad_beta_v.copy(),
                    "grad_alfa_f": grad_alfa_f.copy(),
                    "grad_beta_f": grad_beta_f.copy(),
                    "alfa_od": alfa_od.copy(),
                    "beta_od": beta_od.copy()
                })
                
        if (used_budget - budget) / budget < 0.05:
            pass
            
        s_prev = s_ll.copy()
        sh_prev = sh_ll.copy()
        a_prev = a_ll.copy()
        stop = 0
        
        _iter += 1

    sio.savemat("debug_matlab.mat", {"debug": debug})
    return s, sh, a, f, fext, fij, comp_time, used_budget, pax_obj, op_obj, obj_val_ll, alfa_od, beta_od, obj_hist

def compute_sim_MIP(lam, budget):
    (n, link_cost, station_cost, hub_cost, link_capacity_slope,
     station_capacity_slope, demand, prices, load_factor, op_link_cost,
     congestion_coef_stations, congestion_coef_links, travel_time,
     alt_utility, a_nom, tau, eta, a_max, candidates, omega_t, omega_p) = parameters_4node_network()
     
    write_txt_param('lam', lam)
    write_txt_param('budget', budget)
    print(budget)
    
    a_prev = 1e4 * np.ones((n, n))
    s_prev = 1e4 * np.ones(n)
    write_gams_param_ii('./export_txt/a_prev.txt', a_prev)
    write_gams_param1d_full('./export_txt/s_prev.txt', s_prev)
    
    gmsFile = r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain\mip.gms'
    gamsExe = r'C:\GAMS\50\gams.exe'
    cmd = f'"{gamsExe}" "{gmsFile}"'
    subprocess.run(cmd, shell=True, cwd=r'C:\Users\freal\Desktop\HubSpokeNetworkDesign\4node_spain', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists('./output_all.xlsx'):
        sh_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='sh_level')
        sh = sh_df.to_numpy().flatten() if sh_df is not None and len(sh_df) > 0 else np.zeros(n)
        
        s_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='s_level')
        s = s_df.to_numpy().flatten() if s_df is not None and len(s_df) > 0 else np.zeros(n)
        sprim = s.copy()
        deltas = np.zeros(n)
        
        a = parse_matrix('./output_all.xlsx', 'a_level', n)
        f = parse_matrix('./output_all.xlsx', 'f_level', n)
        fext = parse_matrix('./output_all.xlsx', 'fext_level', n)
        
        mipgap_df = read_gams_csv_robust('./output_all.xlsx', symbol_name='mip_opt_gap')
        mipgap = mipgap_df.to_numpy().flatten() if mipgap_df is not None and len(mipgap_df) > 0 else [0]
        
        ctime_vals = read_gams_csv_robust('./output_all.xlsx', symbol_name='solver_time')
        if ctime_vals is not None and len(ctime_vals) > 0:
            comp_time = ctime_vals.values.flatten()[-1]
        else:
            comp_time = 0
    else:
        sh = np.zeros(n); s = np.zeros(n); sprim = np.zeros(n); deltas = np.zeros(n); a = np.zeros((n,n)); f = np.zeros((n,n)); fext = np.zeros((n,n)); mipgap = [0]; comp_time = 0
        
    fij = np.zeros((n, n, n, n))
    if os.path.exists('fij_long.csv'):
        T = pd.read_csv('fij_long.csv')
        if len(T) > 0:
            iU = T['i'].unique(); jU = T['j'].unique(); oU = T['o'].unique(); dU = T['d'].unique()
            fij = split_and_accumarray(T, iU, jU, oU, dU)
            
    obj_val, pax_obj, op_obj = get_obj_val(op_link_cost, prices, a, f, demand)
    req_bud = budget
    budget_used = get_budget(s, sh, a, n, station_cost, station_capacity_slope, hub_cost, link_cost, lam)
    
    filename = f'./4node_hs_prueba_v0/2h_budget={req_bud}_lam={lam}.mat'
    if not os.path.exists('./4node_hs_prueba_v0'):
        os.makedirs('./4node_hs_prueba_v0')
    sio.savemat(filename, {'s': s, 'sprim': sprim, 'deltas': deltas, 'a': a, 'f': f, 'fext': fext, 'fij': fij,
                           'comp_time': comp_time, 'budget': budget_used, 'pax_obj': pax_obj, 'op_obj': op_obj,
                           'obj_val': obj_val, 'mipgap': mipgap})
                           
    return s, sh, a, f, fext, fij


if __name__ == '__main__':
    basedir = 'export_csv'
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    if not os.path.exists('export_txt'):
        os.makedirs('export_txt')
        
    (n, link_cost, station_cost, hub_cost, link_capacity_slope,
     station_capacity_slope, demand, prices, load_factor, op_link_cost,
     congestion_coef_stations, congestion_coef_links, travel_time, alt_utility,
     a_nom, tau, eta, a_max, candidasourcertes, omega_t, omega_p) = parameters_4node_network()
     
    M = 1e4
    nreg = 20
    eps = 1e-3
    vals_regs = np.linspace(0.005, 0.995, nreg - 1)
    n_airlines = 5
    
    lin_coef, bord, b = get_linearization(n, nreg, alt_utility, vals_regs, n_airlines)
    
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
    
    budgets = [3e4, 3.5e4, 4e4, 4.5e4, 5e4]
    lam = 4
    for bud in budgets:
        compute_sim_MIP(lam, bud)
        
    for bud in budgets:
        filename = f'./4node_hs_prueba_v0/2h_budget={bud}_lam={lam}.mat'
        if os.path.exists(filename):
            data = sio.loadmat(filename)
            print(bud)
            print(data['obj_val'][0][0])
            
    alfas = [0.1]
    gamma = 20
    write_txt_param('gamma', gamma)
    mus_alfa = [1e-5]
    mus_beta = [5e-2]
    
    if not os.path.exists('./4node_hs_prueba_v0_blo'):
        os.makedirs('./4node_hs_prueba_v0_blo')
    """        
    for bud in budgets:
        for al in alfas:
            for mu_al in mus_alfa:
                for mu_bet in mus_beta:
                    alfa_od = np.ones((n, n))
                    beta_od = np.ones((n, n))
                    write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
                    write_gams_param_ii('./export_txt/beta_od.txt', beta_od)
                    
                    sh_prev = 5 * np.ones(n)
                    res = compute_sim_cvx_blo(lam, al, n, bud, mu_al, mu_bet, sh_prev)
                    (s, sh, a, f, fext, fij, comp_time, used_budget, pax_obj, op_obj, obj_val_ll, alfa_od, beta_od, obj_hist) = res
                    
                    obj = np.sum(f * demand * prices) - np.sum(op_link_cost * a)
                    filename = f'./4node_hs_prueba_v0_blo/bud={bud}_lam={lam}_alfa={al}_mu_al={mu_al}_mu_bet={mu_bet}_python.mat'
                    sio.savemat(filename, {'s': s, 'sh': sh, 'a': a, 'f': f, 'fext': fext, 'fij': fij,
                                           'comp_time': comp_time, 'used_budget': used_budget,
                                           'pax_obj': pax_obj, 'op_obj': op_obj, 'obj_val_ll': obj_val_ll,
                                           'alfa_od': alfa_od, 'beta_od': beta_od, 'obj_hist': obj_hist})
    """   
    for bud in budgets:
        for al in alfas:
            for mu_al in mus_alfa:
                for mu_bet in mus_beta:
                    alfa_od = np.ones((n, n))
                    beta_od = np.ones((n, n))
                    write_gams_param_ii('./export_txt/alfa_od.txt', alfa_od)
                    write_gams_param_ii('./export_txt/beta_od.txt', beta_od)
                    
                    sh_prev_list = 5 * np.ones((n + 1, n))
                    for idx_ms in range(n):
                        sh_prev_list[idx_ms + 1, idx_ms] = 2
                        
                    best_obj_ms = 1
                    best_res_ms = {'f': np.zeros((n, n)), 'a': np.zeros((n, n))}
                    
                    for start_idx in range(n + 1):
                        sh_prev_in = sh_prev_list[start_idx, :]
                        print('empieazo multistart con sh = ')
                        print(sh_prev_in)
                        res = compute_sim_cvx_blo(lam, al, n, bud, mu_al, mu_bet, sh_prev_in)
                        (s_curr, sh_curr, a_curr, f_curr, fext_curr, fij_curr, comp_time_curr, used_budget_curr, pax_obj_curr, op_obj_curr, obj_val_ll_curr, alfa_od_curr, beta_od_curr, obj_hist_curr) = res
                        
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
                                                'comp_time': comp_time_curr, 'used_budget': used_budget_curr, 'pax_obj': pax_obj_curr,
                                                'op_obj': op_obj_curr, 'obj_val_ll': obj_val_ll_curr, 'alfa_od': alfa_od_curr,
                                                'beta_od': beta_od_curr, 'obj_hist': obj_hist_curr})
                                                
                    filename = f'./4node_hs_prueba_v0_blo/bud={bud}_lam={lam}_alfa={al}_mu_al={mu_al}_mu_bet={mu_bet}_python.mat'
                    sio.savemat(filename, best_res_ms)

    best_obj = 1e3 * np.ones(len(budgets))
    for bb, bud in enumerate(budgets):
        print(bud)
        best_obj_val = 0
        best_mu_alfa = 0
        best_mu_beta = 0
        filename_MIP = f'./4node_hs_prueba_v0/2h_budget={bud}_lam={lam}.mat'
        if os.path.exists(filename_MIP):
            data_MIP = sio.loadmat(filename_MIP)
            f_MIP = data_MIP['f']
            a_MIP = data_MIP['a']
        else:
            continue
            
        for al in alfas:
            for mu_al in mus_alfa:
                for mu_bet in mus_beta:
                    filename = f'./4node_hs_prueba_v0_blo/bud={bud}_lam={lam}_alfa={al}_mu_al={mu_al}_mu_bet={mu_bet}_python.mat'
                    if os.path.exists(filename):
                        data = sio.loadmat(filename)
                        f = data['f']
                        a = data['a']
                        obj = np.sum(f * prices * demand) - np.sum(a * op_link_cost)
                        if obj > best_obj_val:
                            best_obj_val = obj
                            best_f = f.copy()
                            best_a = a.copy()
                            best_mu_alfa = mu_al
                            best_mu_beta = mu_bet
                            
        f_MIP[np.abs(f_MIP - best_f) < 2e-2] = best_f[np.abs(f_MIP - best_f) < 2e-2]
        a_MIP[np.abs(f_MIP - best_f) < 2e-2] = best_a[np.abs(f_MIP - best_f) < 2e-2]
        
        obj_MIP = np.sum(f_MIP * prices * demand) - np.sum(a_MIP * op_link_cost)
        gap = 100 * (obj_MIP - best_obj_val) / obj_MIP if obj_MIP != 0 else 0
        print(f'budget = {bud}: best obj  = {best_obj_val}, mu_alfa = {best_mu_alfa}, mu_beta = {best_mu_beta}, MIP obj = {obj_MIP}, gap = {gap}')
