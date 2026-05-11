#!/usr/bin/env python3
"""
BLO para red_iberia_simple (25 nodos).

Para cada presupuesto se lanza un multistart sobre sh_prev con tres
inicializaciones y se conserva la mejor solución encontrada.
"""

import os
import sys
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
import scipy.io as sio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from generate_data import (
    build_network,
    get_competitor_airline_count,
    write_gams_param_ii,
    write_gams_param1d_full,
    write_gams_param_iii,
    write_txt_param,
    get_linearization,
    parse_matrix,
    read_gams_csv_robust,
)

GAMS_EXE = '/opt/gams/gams49.6_linux_x64_64_sfx/gams'
N_AIRLINES = get_competitor_airline_count()
NREG = 20
LOGIT_COEF = 0.02
GAMS_RUNTIME_FILES = [
    'cvx-ll.gms',
    'cvx-sl.gms',
    'param_definition.gms',
    'param_definition_cvx.gms',
    'param_definition_cvx-blo.gms',
    'mosek.opt',
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def get_budget(s, sh, n, station_cost, station_capacity_slope, hub_cost, lam):
    total = 0.0
    for i in range(n):
        if s[i] > 1e-2:
            total += station_cost[i] + station_capacity_slope[i] * (s[i] + sh[i])
        if sh[i] > 1e-2:
            total += lam * hub_cost[i]
    return total


def get_obj(op_link_cost, prices, demand, a, f):
    op_obj = float(np.sum(op_link_cost * a))
    pax_obj = float(np.sum(prices * demand * f))
    return -pax_obj + op_obj, pax_obj, op_obj


def write_txt_param_in_dir(work_dir, name, val):
    with open(os.path.join(work_dir, 'export_txt', f'{name}.txt'), 'w') as f:
        f.write(f'{val:e}')


def set_f_bounds(n, fij, travel_time, prices, alt_utility, omega_p, omega_t, work_dir):
    cotas = np.zeros((n, n))
    for oo in range(n):
        for dd in range(n):
            u = (np.sum(travel_time[fij[:, :, oo, dd] > 1e-3]) * omega_t
                 + prices[oo, dd] * omega_p)
            alt_u = alt_utility[oo, dd]
            umax = max(u, alt_u)
            num = np.exp(u - umax)
            den = num + N_AIRLINES * np.exp(alt_u - umax)
            cotas[oo, dd] = num / den
    write_gams_param_ii(os.path.join(work_dir, 'export_txt', 'f_bounds.txt'), cotas)


def build_sh_multistarts(nodes, high_value):
    """Devuelve las inicializaciones pedidas para sh_prev."""
    n = len(nodes)
    idx = {iata: i for i, iata in enumerate(nodes)}
    starts = []

    sh_all_high = high_value * np.ones(n)
    starts.append(('all_high', sh_all_high))

    sh_no_mad = high_value * np.ones(n)
    if 'MAD' in idx:
        sh_no_mad[idx['MAD']] = 0.0
    starts.append(('all_high_except_mad', sh_no_mad))

    sh_no_bcn = high_value * np.ones(n)
    if 'BCN' in idx:
        sh_no_bcn[idx['BCN']] = 0.0
    starts.append(('all_high_except_bcn', sh_no_bcn))

    return starts


def prepare_gams_run_dir():
    run_dir = Path(tempfile.mkdtemp(prefix='red_iberia_simple_multistart_', dir='/tmp'))
    export_dir = run_dir / 'export_txt'
    export_dir.mkdir(parents=True, exist_ok=True)
    base_export_dir = Path(BASE_DIR) / 'export_txt'

    for filename in GAMS_RUNTIME_FILES:
        src = Path(BASE_DIR) / filename
        if src.exists():
            shutil.copy2(src, run_dir / filename)

    if base_export_dir.exists():
        for src in base_export_dir.iterdir():
            if src.is_file():
                shutil.copy2(src, export_dir / src.name)

    return str(run_dir)


def run_single_start(start_name, sh_prev_init, lam, alfa, budget, mu_alfa, mu_beta,
                     net_params, niters=40, bliters=1, gamma=20):
    run_dir = prepare_gams_run_dir()
    try:
        result = compute_blo(
            lam, alfa, budget, mu_alfa, mu_beta, sh_prev_init, net_params,
            niters=niters, bliters=bliters, gamma=gamma, work_dir=run_dir
        )
        result['start_name'] = start_name
        result['sh_prev_init'] = sh_prev_init.copy()
        return result
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def run_budget_multistart(lam, alfa, budget, mu_alfa, mu_beta, net_params,
                          high_value, niters=40, bliters=1, gamma=20):
    """Ejecuta las tres inicializaciones en paralelo y conserva la mejor."""
    nodes = net_params[1]
    starts = build_sh_multistarts(nodes, high_value)
    all_results = []
    total_comp_time = 0.0

    with ThreadPoolExecutor(max_workers=len(starts)) as executor:
        future_to_start = {
            executor.submit(
                run_single_start,
                start_name,
                sh_prev_init,
                lam,
                alfa,
                budget,
                mu_alfa,
                mu_beta,
                net_params,
                niters,
                bliters,
                gamma,
            ): start_name
            for start_name, sh_prev_init in starts
        }

        for future in as_completed(future_to_start):
            result = future.result()
            all_results.append(result)
            total_comp_time += float(result['comp_time'])
            print(
                f'  multistart={result["start_name"]}  '
                f'obj_val={result["obj_val"]:.6g}  '
                f'used_budget={result["used_budget"]:.6g}  '
                f'comp_time={result["comp_time"]:.6g}',
                flush=True,
            )

    best_result = min(all_results, key=lambda result: float(result['obj_val']))
    best_result = dict(best_result)
    best_result['comp_time_single_start'] = float(best_result['comp_time'])
    best_result['comp_time'] = total_comp_time
    best_result['multistart_labels'] = np.array([name for name, _ in starts], dtype=object)
    best_result['selected_start'] = best_result['start_name']
    best_result['n_starts'] = len(starts)

    return best_result


def run_budget_case(budget, lam, alfa, mu_alfa, mu_beta, net_params,
                    sh_init_high_value, out_dir, niters=40, bliters=1, gamma=20):
    print(f'\n=== budget={budget:.0e} ===', flush=True)
    result = run_budget_multistart(
        lam, alfa, budget, mu_alfa, mu_beta, net_params,
        sh_init_high_value, niters=niters, bliters=bliters, gamma=gamma
    )

    fname = (f'bud={budget:.2e}_lam={lam}_alfa={alfa}'
             f'_mu_al={mu_alfa:.2e}_mu_bet={mu_beta:.2e}_python.mat')
    sio.savemat(os.path.join(out_dir, fname), result)
    print(
        f'  best_start={result["selected_start"]}  '
        f'obj_val={result["obj_val"]:.6g}  '
        f'used_budget={result["used_budget"]:.6g}  '
        f'comp_time_total={result["comp_time"]:.6g}',
        flush=True,
    )
    print(f'  Guardado: {fname}', flush=True)
    return budget, fname, result


def load_fij(n, work_dir):
    path = os.path.join(work_dir, 'fij_long.csv')
    fij = np.zeros((n, n, n, n))
    if not os.path.exists(path):
        return fij
    df = pd.read_csv(path)
    if len(df) == 0:
        return fij
    for col in ['i', 'j', 'o', 'd']:
        df[col] = df[col].astype(str).str.strip().str.extract(r'(\d+)').astype(int) - 1
    df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0.0)
    for row in df.itertuples(index=False):
        if 0 <= row.i < n and 0 <= row.j < n and 0 <= row.o < n and 0 <= row.d < n:
            fij[row.i, row.j, row.o, row.d] += row.value
    return fij


def run_gams(gms_file, work_dir):
    subprocess.run(
        [GAMS_EXE, gms_file],
        cwd=work_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def read_outputs(out_xlsx, n):
    sh_df = read_gams_csv_robust(out_xlsx, 'sh_level')
    sh = sh_df.to_numpy().flatten() if sh_df is not None and len(sh_df) > 0 else np.zeros(n)
    sh = np.maximum(sh, 1e-4)

    s_df = read_gams_csv_robust(out_xlsx, 's_level')
    s = s_df.to_numpy().flatten() if s_df is not None and len(s_df) > 0 else np.zeros(n)
    s = np.maximum(s, 1e-4)

    _df_t = read_gams_csv_robust(out_xlsx, 'solver_time')
    t = float(_df_t.columns[0]) if _df_t is not None and len(_df_t.columns) > 0 else 0.0

    a = parse_matrix(out_xlsx, 'a_level', n)
    a = np.maximum(a, 1e-4)
    f = parse_matrix(out_xlsx, 'f_level', n)
    fext = parse_matrix(out_xlsx, 'fext_level', n)
    return s, sh, a, f, fext, t


# ─── BLO ──────────────────────────────────────────────────────────────────────

def compute_blo(lam, alfa, budget, mu_alfa, mu_beta, sh_prev_init,
                net_params, niters=40, bliters=1, gamma=20, work_dir=None):
    (n, nodes, idx, distance,
     link_cost, station_cost, hub_cost,
     link_capacity_slope, station_capacity_slope,
     demand, prices, load_factor,
     op_link_cost, congestion_coef_stations, congestion_coef_links,
     travel_time, alt_utility,
     a_nom, tau, eta, a_max, candidates, omega_t, omega_p) = net_params

    work_dir = work_dir or BASE_DIR
    txt = os.path.join(work_dir, 'export_txt')
    out_ll_xlsx = os.path.join(work_dir, 'output_ll.xlsx')
    out_sl_xlsx = os.path.join(work_dir, 'output_sl.xlsx')
    ll_gms = os.path.join(work_dir, 'cvx-ll.gms')
    sl_gms = os.path.join(work_dir, 'cvx-sl.gms')

    alfa_od = np.ones((n, n))
    beta_od = np.ones((n, n))

    write_txt_param_in_dir(work_dir, 'gamma', gamma)
    write_txt_param_in_dir(work_dir, 'niters', niters)
    write_txt_param_in_dir(work_dir, 'n_airlines', N_AIRLINES)
    write_txt_param_in_dir(work_dir, 'lam', lam)
    write_txt_param_in_dir(work_dir, 'alfa', alfa)
    write_txt_param_in_dir(work_dir, 'budget', budget)
    write_gams_param_ii(f'{txt}/alfa_od.txt', alfa_od)
    write_gams_param_ii(f'{txt}/beta_od.txt', beta_od)

    a_prev = 1e4 * np.ones((n, n))
    s_prev = 1e4 * np.ones(n)
    sh_prev = sh_prev_init.copy()

    write_gams_param_ii(f'{txt}/a_prev.txt', a_prev)
    write_gams_param1d_full(f'{txt}/s_prev.txt', s_prev)
    write_gams_param1d_full(f'{txt}/sh_prev.txt', sh_prev)

    # f_bounds iniciales con fij vacío
    set_f_bounds(
        n, np.zeros((n, n, n, n)), travel_time, prices, alt_utility, omega_p, omega_t, work_dir
    )

    obj_val = 0.0
    obj_val_prev = 1e3
    comp_time = 0.0
    used_budget = 0.0
    pax_obj = 0.0
    op_obj = 0.0
    obj_val_ll = 0.0

    s = s_prev.copy()
    sh = sh_prev.copy()
    a = a_prev.copy()
    f = np.zeros((n, n))
    fext = np.zeros((n, n))
    fij = np.zeros((n, n, n, n))
    a_ll = a.copy(); f_ll = f.copy(); fext_ll = fext.copy()
    s_ll = s.copy(); sh_ll = sh.copy()

    s_traj = []
    sh_traj = []
    f_traj = []

    for _iter in range(1, niters + 1):
        print(f'  iter {_iter}/{niters}', flush=True)
        write_txt_param_in_dir(work_dir, 'current_iter', _iter)
        write_gams_param_ii(f'{txt}/alfa_od.txt', alfa_od)
        write_gams_param_ii(f'{txt}/beta_od.txt', beta_od)

        stop = 0
        for bliter in range(1, bliters + 1):
            if bliter > 1 and abs((obj_val - obj_val_prev) / (obj_val + 1e-4)) <= 1e-3:
                stop = 1
            if stop:
                break

            write_gams_param_ii(f'{txt}/a_prev.txt', a_prev)
            write_gams_param1d_full(f'{txt}/s_prev.txt', s_prev)
            write_gams_param1d_full(f'{txt}/sh_prev.txt', sh_prev)

            # ── Lower level ───────────────────────────────────────────────────
            run_gams(ll_gms, work_dir)
            s, sh, a, f, fext, t = read_outputs(out_ll_xlsx, n)
            comp_time += t
            fij = load_fij(n, work_dir)

            a[a < 1e-2] = 0; f[f < 1e-2] = 0
            fij[fij < 1e-2] = 0; fext[fext > 0.99] = 1

            a_ll = a.copy(); f_ll = f.copy(); fext_ll = fext.copy()
            s_ll = s.copy(); sh_ll = sh.copy()

            # Gradientes desde LL
            grad_alfa_v = np.zeros((n, n))
            grad_beta_v = np.zeros((n, n))
            for oo in range(n):
                for dd in range(n):
                    fij_od = fij[:, :, oo, dd]
                    term = (LOGIT_COEF * prices[oo, dd] * f[oo, dd]
                            + LOGIT_COEF * np.sum(fij_od * travel_time)
                            - alt_utility[oo, dd] * fext[oo, dd]
                            + f[oo, dd] * (np.log(np.maximum(0, f[oo, dd]) + 1e-12) - 1)
                            + fext[oo, dd] * (np.log(np.maximum(0, fext[oo, dd]) / N_AIRLINES + 1e-12) - 1))
                    coef_b = (prices[oo, dd] * demand[oo, dd]) ** beta_od[oo, dd]
                    grad_alfa_v[oo, dd] = coef_b * term
                    grad_beta_v[oo, dd] = ((alfa_od[oo, dd] + 1e-4)
                                           * (beta_od[oo, dd] * demand[oo, dd] * prices[oo, dd])
                                           ** (beta_od[oo, dd] - 1)
                                           * term)

            used_budget = get_budget(s, sh, n, station_cost, station_capacity_slope, hub_cost, lam)
            set_f_bounds(n, fij, travel_time, prices, alt_utility, omega_p, omega_t, work_dir)

            # ── Upper level ───────────────────────────────────────────────────
            run_gams(sl_gms, work_dir)
            s, sh, a, f, fext, t = read_outputs(out_sl_xlsx, n)
            comp_time += t
            fij = load_fij(n, work_dir)

            a[a < 1e-2] = 0; f[f < 1e-2] = 0
            fext[fext > 0.99] = 1; fij[fij < 1e-2] = 0

            obj_val_ll, pax_obj, op_obj = get_obj(op_link_cost, prices, demand, a_ll, f_ll)
            used_budget = get_budget(s, sh, n, station_cost, station_capacity_slope, hub_cost, lam)

            # Gradientes desde SL
            grad_alfa_f = np.zeros((n, n))
            grad_beta_f = np.zeros((n, n))
            for oo in range(n):
                for dd in range(n):
                    fij_od = fij[:, :, oo, dd]
                    term_f = (LOGIT_COEF * prices[oo, dd] * f[oo, dd]
                              + LOGIT_COEF * np.sum(fij_od * travel_time)
                              - alt_utility[oo, dd] * fext[oo, dd]
                              + f[oo, dd] * (np.log(f[oo, dd] + 1e-12) - 1)
                              + fext[oo, dd] * (np.log(fext[oo, dd] / N_AIRLINES + 1e-12) - 1))
                    grad_alfa_f[oo, dd] = gamma * (
                        ((demand[oo, dd] * prices[oo, dd]) ** beta_od[oo, dd]) * term_f
                        - grad_alfa_v[oo, dd])
                    grad_beta_f[oo, dd] = gamma * (
                        (alfa_od[oo, dd] + 1e-4)
                        * (beta_od[oo, dd] * demand[oo, dd] * prices[oo, dd])
                        ** (beta_od[oo, dd] - 1)
                        * term_f
                        - grad_beta_v[oo, dd])

            beta_od -= mu_beta * grad_beta_f
            alfa_od -= mu_alfa * grad_alfa_f
            beta_od = np.clip(beta_od, 0.5, 2.1)
            alfa_od = np.clip(alfa_od, 1.0, 9.0)
            np.fill_diagonal(beta_od, 1)
            np.fill_diagonal(alfa_od, 1)

            obj_val_prev = obj_val
            obj_val = obj_val_ll

        s_traj.append(s_ll.copy())
        sh_traj.append(sh_ll.copy())
        f_traj.append(f_ll.copy())

        s_prev = s_ll.copy()
        sh_prev = sh_ll.copy()
        a_prev = a_ll.copy()

    return {
        's': s_ll, 'sh': sh_ll, 'a': a_ll, 'f': f_ll,
        'fext': fext_ll, 'fij': fij,
        'comp_time': comp_time, 'used_budget': used_budget,
        'pax_obj': pax_obj, 'op_obj': op_obj, 'obj_val': obj_val_ll,
        'alfa_od': alfa_od, 'beta_od': beta_od,
        's_traj': np.array(s_traj),
        'sh_traj': np.array(sh_traj),
        'f_traj': np.array(f_traj),
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Construyendo red iberia simple (25 nodos)...')
    net_params = build_network()
    (n, nodes, idx, distance,
     link_cost, station_cost, hub_cost,
     link_capacity_slope, station_capacity_slope,
     demand, prices, load_factor,
     op_link_cost, congestion_coef_stations, congestion_coef_links,
     travel_time, alt_utility,
     a_nom, tau, eta, a_max, candidates, omega_t, omega_p) = net_params

    print(f'  n={n}, pares OD con demanda>0: {int((demand > 0).sum())}')

    # Datos estáticos (ya generados por generate_data.py, se sobreescriben por seguridad)
    txt = os.path.join(BASE_DIR, 'export_txt')
    vals_regs = np.linspace(0.005, 0.995, NREG - 1)
    lin_coef, bord, b = get_linearization(n, NREG, alt_utility, vals_regs, N_AIRLINES)

    alfa_od_init = np.ones((n, n))
    beta_od_init = np.ones((n, n))

    write_gams_param_iii(f'{txt}/lin_coef.txt', lin_coef)
    write_gams_param_iii(f'{txt}/b.txt', b)
    write_gams_param_iii(f'{txt}/bord.txt', bord)
    write_gams_param_ii(f'{txt}/demand.txt', demand)
    write_gams_param_ii(f'{txt}/travel_time.txt', travel_time)
    write_gams_param_ii(f'{txt}/alt_utility.txt', alt_utility)
    write_gams_param_ii(f'{txt}/link_cost.txt', link_cost)
    write_gams_param_ii(f'{txt}/link_capacity_slope.txt', link_capacity_slope)
    write_gams_param_ii(f'{txt}/prices.txt', prices)
    write_gams_param_ii(f'{txt}/op_link_cost.txt', op_link_cost)
    write_gams_param_ii(f'{txt}/a_nom.txt', a_nom)
    write_gams_param_ii(f'{txt}/candidates.txt', candidates)
    write_gams_param_ii(f'{txt}/congestion_coefs_links.txt', congestion_coef_links)
    write_gams_param_ii(f'{txt}/alfa_od.txt', alfa_od_init)
    write_gams_param_ii(f'{txt}/beta_od.txt', beta_od_init)
    write_gams_param1d_full(f'{txt}/station_cost.txt', station_cost)
    write_gams_param1d_full(f'{txt}/hub_cost.txt', hub_cost)
    write_gams_param1d_full(f'{txt}/station_capacity_slope.txt', station_capacity_slope)
    write_gams_param1d_full(f'{txt}/congestion_coefs_stations.txt', congestion_coef_stations)
    write_txt_param('gamma', 20)
    print('  export_txt/ actualizado.')

    # Parámetros BLO
    lam = 4
    alfa = 0.1
    #mu_alfa = 1e-7
    #mu_beta = 0.01
    mu_alfa = 0
    mu_beta = 0

    # 4 presupuestos por décadas
    budgets = [1e5,1.5e5,2e5,3e5]

    out_dir = os.path.join(BASE_DIR, 'hs_prueba_v0_blo')
    os.makedirs(out_dir, exist_ok=True)

    sh_init_high_value = 40.0

    with ThreadPoolExecutor(max_workers=len(budgets)) as executor:
        futures = [
            executor.submit(
                run_budget_case,
                budget,
                lam,
                alfa,
                mu_alfa,
                mu_beta,
                net_params,
                sh_init_high_value,
                out_dir,
            )
            for budget in budgets
        ]
        for future in as_completed(futures):
            future.result()

    print('\nListo.')
