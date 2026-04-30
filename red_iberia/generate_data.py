"""
generate_data.py  –  red_iberia (99 nodos)

Adaptado de 6node_spain/generate_data_8nodemapto6.py.

Genera:
  - distance.csv   : matriz n×n de distancias haversine (km), sin cabecera
  - prices.csv     : matriz n×n de precios (yield × distancia, EUR/pax), sin cabecera
  - export_txt/    : todos los parámetros en formato GAMS .txt

Yield por mercado: media de las observaciones disponibles en yield.xlsx.
Mercados sin yield: se les asigna la media global (ver mercados_sin_yield.md).
"""
import os
import re
import sys
import time
import subprocess

import numpy as np
import pandas as pd
import scipy.io as sio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── helpers (idénticos al original) ─────────────────────────────────────────

def read_gams_csv_robust(file_path, symbol_name, max_retries=5, delay=1.0):
    for attempt in range(max_retries):
        try:
            with pd.ExcelFile(file_path) as xls:
                df = pd.read_excel(xls, sheet_name=symbol_name)
            return df
        except Exception:
            if attempt == max_retries - 1:
                return pd.DataFrame()
            time.sleep(delay)

def read_excel_robust(file_path, sheet_name, max_retries=5, delay=1.0):
    for attempt in range(max_retries):
        try:
            return pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception:
            if attempt == max_retries - 1:
                return pd.DataFrame()
            time.sleep(delay)

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

def matlab_sprintf_d(val):
    if val == int(val) and abs(val) < 1e15:
        return str(int(val))
    return f'{val:e}'

def write_txt_param(name, val):
    with open(os.path.join(BASE_DIR, f'export_txt/{name}.txt'), 'w') as f:
        f.write(matlab_sprintf_d(val))

def write_gams_param_ii_path(path, M):
    write_gams_param_ii(path, M)

def write_gams_param1d_path(path, v):
    write_gams_param1d_full(path, v)

def parse_matrix(output_xlsx, name, n):
    m_df = read_gams_csv_robust(output_xlsx, symbol_name=name)
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
                valid = ((m_df['i_idx'] >= 0) & (m_df['i_idx'] < n) &
                         (m_df['j_idx'] >= 0) & (m_df['j_idx'] < n))
                m[m_df.loc[valid, 'i_idx'].values,
                  m_df.loc[valid, 'j_idx'].values] = m_df.loc[valid, 'value'].values
                return m
            except Exception:
                return np.zeros((n, n))
        return np.zeros((n, n))

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

# ─── distancia haversine ──────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ─── construcción de la red ───────────────────────────────────────────────────

def assign_yield(y_market, global_mean):
    """Devuelve función que asigna yield a un par OD y su categoría.

    Prioridad:
      1. Yield directo del mercado en yield.xlsx
      2. Media del yield O→MAD y MAD→D (conexión vía Madrid)
      3. Media global de todos los yields disponibles
    """
    def get(o, d):
        if (o, d) in y_market.index:
            return y_market[(o, d)], 'directo'
        if (o, 'MAD') in y_market.index and ('MAD', d) in y_market.index:
            y = (y_market[(o, 'MAD')] + y_market[('MAD', d)]) / 2
            return y, 'via_mad'
        return global_mean, 'media_global'
    return get


def build_network():
    """Lee datos fuente y devuelve todos los parámetros de la red.

    Incluye todos los mercados con demanda > 0.
    Yield asignado por prioridad: directo > vía MAD > media global.
    """
    iata_re = re.compile(r'^[A-Z]{3}$')

    # ── Yield ─────────────────────────────────────────────────────────────────
    dy = pd.read_excel(os.path.join(BASE_DIR, 'yield.xlsx'), sheet_name='yield')
    y_market = dy.groupby(['Origen', 'Destino'])['Yield-PKT'].mean()
    global_mean = dy['Yield-PKT'].mean()
    get_yield = assign_yield(y_market, global_mean)

    # ── Demanda: todos los pares con demanda > 0 ──────────────────────────────
    dd = pd.read_excel(os.path.join(BASE_DIR, 'demanda.xlsx'),
                       sheet_name='demanda_completa')
    dd = dd.dropna(subset=['Origen', 'Destino'])
    dd = dd[dd['Origen'].apply(lambda x: bool(iata_re.match(str(x)))) &
            dd['Destino'].apply(lambda x: bool(iata_re.match(str(x))))]
    dd = dd[dd['Promedio de Suma de Demanda'] > 0]

    # Nodos activos (orden alfabético)
    active_nodes = sorted(set(dd['Origen'].tolist() + dd['Destino'].tolist()))

    # Actualizar airports.csv con los 99 nodos activos
    all_airports_full = pd.read_csv(os.path.join(BASE_DIR, 'airports_full.csv'))
    airports = all_airports_full[all_airports_full['iata'].isin(active_nodes)].sort_values('iata').reset_index(drop=True)
    airports.to_csv(os.path.join(BASE_DIR, 'airports.csv'), index=False)

    nodes = airports['iata'].tolist()
    n = len(nodes)
    idx = {iata: i for i, iata in enumerate(nodes)}

    lats = airports['lat'].values
    lons = airports['lon'].values

    # Estadísticas de asignación
    counts = {'directo': 0, 'via_mad': 0, 'media_global': 0}
    for _, row in dd.iterrows():
        _, cat = get_yield(row['Origen'], row['Destino'])
        counts[cat] += 1
    print(f'  Nodos: {n}  |  Mercados: {len(dd)}')
    print(f'  Yield directo: {counts["directo"]}  |  Vía MAD: {counts["via_mad"]}  |  Media global: {counts["media_global"]}')

    # ── Distancias (km) ───────────────────────────────────────────────────────
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = haversine_km(lats[i], lons[i], lats[j], lons[j])

    # ── Precios: yield asignado × distancia ──────────────────────────────────
    prices = np.zeros((n, n))
    for i, o in enumerate(nodes):
        for j, d in enumerate(nodes):
            if i == j:
                continue
            y, _ = get_yield(o, d)
            prices[i, j] = y * dist[i, j]

    # ── Demanda (pax/semana) ──────────────────────────────────────────────────
    demand = np.zeros((n, n))
    for _, row in dd.iterrows():
        o, d = row['Origen'], row['Destino']
        if o in idx and d in idx:
            demand[idx[o], idx[d]] = row['Promedio de Suma de Demanda']

    # ── Parámetros operativos (misma lógica que 6node / 8node) ───────────────
    omega_t = -0.02
    omega_p = -0.02
    n_airlines = 5

    link_cost = 10.0 * dist
    np.fill_diagonal(link_cost, 1e4)
    link_cost[link_cost == 0] = 1e4

    station_cost = 3e3 * np.ones(n)
    hub_cost     = 5e3 * np.ones(n)

    link_capacity_slope    = 0.2 * link_cost
    station_capacity_slope = (5 * 5e2 + 4 * 50 * 8) * np.ones(n)

    load_factor = 0.25 * np.ones(n)

    congestion_coef_stations = 0.1 * np.ones(n)
    congestion_coef_links    = 0.1 * np.ones((n, n))

    takeoff_time = 20
    landing_time = 20
    taxi_time    = 10
    cruise_time  = 60.0 * dist / 800.0

    travel_time = cruise_time + takeoff_time + landing_time + taxi_time
    np.fill_diagonal(travel_time, 0)

    # Utilidad alternativa (misma semilla que el original)
    np.random.seed(123)
    p_escala = 0.4
    alt_utility = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            escala = np.random.rand(n_airlines) < p_escala
            alt_time_vec  = travel_time[i, j] * (1 + 0.5 * escala) + 60 * escala
            alt_price_vec = prices[i, j] + 0.3 * prices[i, j] * (np.random.rand(n_airlines) - 0.5)
            alt_u = (np.log(np.sum(np.exp(omega_p * alt_price_vec + omega_t * alt_time_vec)))
                     - np.log(n_airlines))
            alt_utility[i, j] = alt_u
            alt_utility[j, i] = alt_u
    np.fill_diagonal(alt_utility, 0)

    op_link_cost = 7600.0 * travel_time / 60.0

    candidates = np.ones((n, n)) - np.eye(n)

    a_nom = 171
    tau   = 0.85
    eta   = 0.3
    a_max = 1e9

    return (n, nodes, idx, dist, link_cost, station_cost, hub_cost,
            link_capacity_slope, station_capacity_slope, demand, prices,
            load_factor, op_link_cost, congestion_coef_stations,
            congestion_coef_links, travel_time, alt_utility,
            a_nom, tau, eta, a_max, candidates, omega_t, omega_p)


# ─── helpers BLO (idénticos al original) ─────────────────────────────────────

def get_linearization(n, nreg, alt_utility, vals_regs, n_airlines):
    dmax     = np.zeros((nreg, n, n))
    dmin     = np.zeros((nreg, n, n))
    lin_coef = np.zeros((nreg, n, n))
    bord     = np.zeros((nreg, n, n))
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
                    bord[r, o, d]     = vals_regs[r]
                else:
                    lin_coef[r, o, d] = ((vals_regs[r] - vals_regs[r - 1]) /
                                         (dmax[r, o, d] - dmin[r, o, d]))
                    bord[r, o, d] = vals_regs[r - 1]
            lin_coef[0, o, d] = vals_regs[0] / (dmax[0, o, d] - dmin[0, o, d])
            bord[0, o, d] = 0
            if dmin[nreg - 1, o, d] == 0:
                lin_coef[nreg - 1, o, d] = 0
            else:
                lin_coef[nreg - 1, o, d] = ((1 - vals_regs[nreg - 2]) /
                                              (0 - dmin[nreg - 1, o, d]))
            bord[nreg - 1, o, d] = vals_regs[nreg - 2]
    b = dmin
    return lin_coef, bord, b


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'export_txt'), exist_ok=True)

    print('Construyendo red iberia (99 nodos)…')
    (n, nodes, idx, distance,
     link_cost, station_cost, hub_cost,
     link_capacity_slope, station_capacity_slope,
     demand, prices, load_factor,
     op_link_cost, congestion_coef_stations, congestion_coef_links,
     travel_time, alt_utility,
     a_nom, tau, eta, a_max, candidates,
     omega_t, omega_p) = build_network()

    print(f'  Nodos: {n}')
    print(f'  Pares OD con demanda>0: {int((demand > 0).sum())}')

    # ── Guardar distance.csv y prices.csv ─────────────────────────────────────
    pd.DataFrame(distance).to_csv(os.path.join(BASE_DIR, 'distance.csv'),
                                  header=False, index=False)
    pd.DataFrame(prices).to_csv(os.path.join(BASE_DIR, 'prices.csv'),
                                header=False, index=False)
    print('  Guardados: distance.csv, prices.csv')

    # ── Linearización PWL ────────────────────────────────────────────────────
    nreg      = 20
    vals_regs = np.linspace(0.005, 0.995, nreg - 1)
    n_airlines = 5
    lin_coef, bord, b = get_linearization(n, nreg, alt_utility, vals_regs, n_airlines)

    alfa_od = np.ones((n, n))
    beta_od = np.ones((n, n))
    gamma   = 20

    txt = os.path.join(BASE_DIR, 'export_txt')

    # ── Escribir export_txt ───────────────────────────────────────────────────
    write_gams_param_iii(f'{txt}/lin_coef.txt',  lin_coef)
    write_gams_param_iii(f'{txt}/b.txt',          b)
    write_gams_param_iii(f'{txt}/bord.txt',       bord)

    write_gams_param_ii(f'{txt}/demand.txt',                 demand)
    write_gams_param_ii(f'{txt}/travel_time.txt',            travel_time)
    write_gams_param_ii(f'{txt}/alt_utility.txt',            alt_utility)
    write_gams_param_ii(f'{txt}/link_cost.txt',              link_cost)
    write_gams_param_ii(f'{txt}/link_capacity_slope.txt',    link_capacity_slope)
    write_gams_param_ii(f'{txt}/prices.txt',                 prices)
    write_gams_param_ii(f'{txt}/op_link_cost.txt',           op_link_cost)
    write_gams_param_ii(f'{txt}/candidates.txt',             candidates)
    write_gams_param_ii(f'{txt}/congestion_coefs_links.txt', congestion_coef_links)
    write_gams_param_ii(f'{txt}/alfa_od.txt',                alfa_od)
    write_gams_param_ii(f'{txt}/beta_od.txt',                beta_od)

    write_gams_param1d_full(f'{txt}/station_cost.txt',           station_cost)
    write_gams_param1d_full(f'{txt}/hub_cost.txt',               hub_cost)
    write_gams_param1d_full(f'{txt}/station_capacity_slope.txt', station_capacity_slope)
    write_gams_param1d_full(f'{txt}/congestion_coefs_stations.txt', congestion_coef_stations)

    a_prev  = 1e4 * np.ones((n, n))
    s_prev  = 1e4 * np.ones(n)
    sh_prev = 1e-3 * s_prev

    write_gams_param_ii(f'{txt}/a_prev.txt',   a_prev)
    write_gams_param1d_full(f'{txt}/s_prev.txt',  s_prev)
    write_gams_param1d_full(f'{txt}/sh_prev.txt', sh_prev)

    write_txt_param('gamma',  gamma)
    write_txt_param('niters', 40)

    print('  export_txt/ generado.')
    print('Listo.')
