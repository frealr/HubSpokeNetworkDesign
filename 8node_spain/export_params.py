"""
Escribe todos los archivos de export_txt/ que necesita GAMS para ejecutar
mip.gms, cvx-ll.gms y cvx-sl.gms.

No lanza ningún subproceso ni modelo. Solo genera datos.

Uso:
    python export_params.py --budget 40000 --lam 4 --alfa 0.1
    python export_params.py --budget 40000 --lam 4 --alfa 0.1 --dm-pax 0.01 --dm-op 0.008
"""

import argparse
import os

import numpy as np

from generate_data import (
    get_linearization,
    parameters_8node_network,
    write_gams_param1d_full,
    write_gams_param_ii,
    write_gams_param_iii,
    write_txt_param,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TXTDIR = os.path.join(SCRIPT_DIR, 'export_txt')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--budget',  type=float, required=True,
                        help='Presupuesto diario (€/día), e.g. 40000')
    parser.add_argument('--lam',     type=float, default=4,
                        help='Multiplicador coste hub (default: 4)')
    parser.add_argument('--alfa',    type=float, default=0.1,
                        help='Peso demanda en el objetivo BLO (default: 0.1)')
    parser.add_argument('--dm-pax',  type=float, default=0.01, dest='dm_pax',
                        help='Pendiente demanda pasajeros en MIP (default: 0.01)')
    parser.add_argument('--dm-op',   type=float, default=0.008, dest='dm_op',
                        help='Pendiente costes operativos en MIP (default: 0.008)')
    parser.add_argument('--gamma',   type=float, default=20,
                        help='Parámetro gamma del método BLO (default: 20)')
    parser.add_argument('--niters',  type=int,   default=20,
                        help='Iteraciones externas BLO (default: 20)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(TXTDIR, exist_ok=True)
    os.chdir(SCRIPT_DIR)

    # ------------------------------------------------------------------
    # Parámetros físicos de la red
    # ------------------------------------------------------------------
    (n, link_cost, station_cost, hub_cost, link_capacity_slope,
     station_capacity_slope, demand, prices, load_factor, op_link_cost,
     congestion_coef_stations, congestion_coef_links, travel_time, alt_utility,
     a_nom, tau, eta, a_max, candidates_raw, omega_t, omega_p) = parameters_8node_network()

    candidates = np.zeros((n, n))
    candidates[candidates_raw > 0] = 1

    # ------------------------------------------------------------------
    # Linealización PWL de la función logit
    # ------------------------------------------------------------------
    nreg = 10
    n_airlines = 5
    vals_regs = np.linspace(0.005, 0.995, nreg - 1)
    lin_coef, bord, b = get_linearization(n, nreg, alt_utility, vals_regs, n_airlines)

    # ------------------------------------------------------------------
    # Límite superior inicial de f (con fij = 0: ningún vuelo asignado aún)
    # ------------------------------------------------------------------
    f_bounds = np.zeros((n, n))
    for oo in range(n):
        for dd in range(n):
            utility = prices[oo, dd] * omega_p   # fij=0 => sin suma de tiempos
            f_bounds[oo, dd] = (np.exp(utility) /
                                (np.exp(utility) + n_airlines * np.exp(alt_utility[oo, dd])))

    # ------------------------------------------------------------------
    # Valores de arranque para variables de iteración
    # ------------------------------------------------------------------
    a_prev  = 1e4 * np.ones((n, n))
    s_prev  = 1e4 * np.ones(n)
    sh_prev = 1e-3 * s_prev
    alfa_od = np.ones((n, n))
    beta_od = np.ones((n, n))

    # ------------------------------------------------------------------
    # Escribir matrices (i,j) y (o,d)
    # ------------------------------------------------------------------
    write_gams_param_ii(f'{TXTDIR}/demand.txt',                demand)
    write_gams_param_ii(f'{TXTDIR}/travel_time.txt',           travel_time)
    write_gams_param_ii(f'{TXTDIR}/alt_utility.txt',           alt_utility)
    write_gams_param_ii(f'{TXTDIR}/link_cost.txt',             link_cost)
    write_gams_param_ii(f'{TXTDIR}/link_capacity_slope.txt',   link_capacity_slope)
    write_gams_param_ii(f'{TXTDIR}/prices.txt',                prices)
    write_gams_param_ii(f'{TXTDIR}/op_link_cost.txt',          op_link_cost)
    write_gams_param_ii(f'{TXTDIR}/candidates.txt',            candidates)
    write_gams_param_ii(f'{TXTDIR}/congestion_coefs_links.txt',congestion_coef_links)
    write_gams_param_ii(f'{TXTDIR}/alfa_od.txt',               alfa_od)
    write_gams_param_ii(f'{TXTDIR}/beta_od.txt',               beta_od)
    write_gams_param_ii(f'{TXTDIR}/a_prev.txt',                a_prev)
    write_gams_param_ii(f'{TXTDIR}/f_bounds.txt',              f_bounds)

    # ------------------------------------------------------------------
    # Escribir vectores (i)
    # ------------------------------------------------------------------
    write_gams_param1d_full(f'{TXTDIR}/station_cost.txt',           station_cost)
    write_gams_param1d_full(f'{TXTDIR}/hub_cost.txt',               hub_cost)
    write_gams_param1d_full(f'{TXTDIR}/station_capacity_slope.txt', station_capacity_slope)
    write_gams_param1d_full(f'{TXTDIR}/congestion_coefs_stations.txt', congestion_coef_stations)
    write_gams_param1d_full(f'{TXTDIR}/s_prev.txt',                 s_prev)
    write_gams_param1d_full(f'{TXTDIR}/sh_prev.txt',                sh_prev)

    # ------------------------------------------------------------------
    # Escribir linealización PWL (seg, o, d)
    # ------------------------------------------------------------------
    write_gams_param_iii(f'{TXTDIR}/lin_coef.txt', lin_coef)
    write_gams_param_iii(f'{TXTDIR}/b.txt',        b)
    write_gams_param_iii(f'{TXTDIR}/bord.txt',     bord)

    # ------------------------------------------------------------------
    # Escribir escalares de sesión (configurables por CLI)
    # ------------------------------------------------------------------
    write_txt_param('budget',       args.budget)
    write_txt_param('lam',          args.lam)
    write_txt_param('alfa',         args.alfa)
    write_txt_param('dm_pax',       args.dm_pax)
    write_txt_param('dm_op',        args.dm_op)
    write_txt_param('gamma',        args.gamma)
    write_txt_param('niters',       args.niters)
    write_txt_param('current_iter', 1)

    print(f'export_txt/ actualizado: budget={args.budget}, lam={args.lam}, alfa={args.alfa}')
    print(f'  dm_pax={args.dm_pax}, dm_op={args.dm_op}, gamma={args.gamma}, niters={args.niters}')
    print(f'  Archivos escritos en: {TXTDIR}')


if __name__ == '__main__':
    main()
