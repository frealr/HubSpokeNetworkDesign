"""
Rentabilidad potencial de cada mercado en vuelo directo – red_iberia_MAD
=======================================================================
Para cada mercado OD con MAD como origen o destino se calcula:
  - Demanda semanal total (ida + vuelta)
  - Ingresos semanales  = sum(demand * price)
  - Nº vuelos necesarios (por dirección) = ceil(demand / (a_nom * tau))
  - Coste operativo semanal = op_link_cost_por_vuelo * n_vuelos * 2 direcciones
  - Margen bruto semanal = ingresos - coste_op
  - Yield: ingreso por pax
  - CASK-equiv: coste operativo / (a_nom * tau * n_vuelos)  per-pax

Parámetros de aeronave:
  A321 (corto-medio): cap=171 pax,  op_cost=6600 EUR/bloque-hr
  B350 (transatlántico): cap=320 pax, op_cost=12350 EUR/bloque-hr
  Load factor objetivo tau=0.85
"""

import math
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TXT  = os.path.join(BASE, 'export_txt')

TRANSATLANTIC = {"EZE", "GRU", "JFK", "LIM", "MEX"}
A_NOM_NARROW  = 171.0
A_NOM_WIDE    = 320.0
OP_NARROW     = 7600.0 * 0.95   # EUR/hr bloque  (-5%)
OP_WIDE       = 12350.0 * 0.80  # EUR/hr bloque  (-20%)
TAU           = 0.85     # load factor


def read_matrix(fname, n):
    m = np.zeros((n, n))
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, val = line.split()
            parts = key.split('.')
            i = int(parts[0][1:]) - 1
            j = int(parts[1][1:]) - 1
            if 0 <= i < n and 0 <= j < n:
                m[i, j] = float(val)
    return m


# ── Aeropuertos ───────────────────────────────────────────────────────────────
airports = pd.read_csv(os.path.join(BASE, 'airports.csv'))
nodes  = airports['iata'].tolist()
n      = len(nodes)
idx    = {iata: i for i, iata in enumerate(nodes)}

# ── Matrices ─────────────────────────────────────────────────────────────────
demand      = read_matrix(os.path.join(TXT, 'demand.txt'),      n)
prices      = read_matrix(os.path.join(TXT, 'prices.txt'),      n)
travel_time = read_matrix(os.path.join(TXT, 'travel_time.txt'), n)  # minutos
dist_csv    = pd.read_csv(os.path.join(BASE, 'distance.csv'), header=None).values

# ── Calculo por mercado OD ────────────────────────────────────────────────────
rows = []
for j, dest in enumerate(nodes):
    if dest == 'MAD':
        continue
    # MAD es siempre i=0 (i1)
    i = idx['MAD']

    dem_out = demand[i, j]   # MAD → dest
    dem_in  = demand[j, i]   # dest → MAD
    total_dem = dem_out + dem_in

    if total_dem == 0:
        continue

    price_out = prices[i, j]
    price_in  = prices[j, i]

    revenue = dem_out * price_out + dem_in * price_in

    is_trans = dest in TRANSATLANTIC
    a_nom    = A_NOM_WIDE if is_trans else A_NOM_NARROW
    op_rate  = OP_WIDE    if is_trans else OP_NARROW

    tt_hr = travel_time[i, j] / 60.0   # minutos → horas

    # Coste operativo normalizado por pasajero = coste_vuelo / capacidad_avion
    cost_per_flight = op_rate * tt_hr
    cost_pax = cost_per_flight / a_nom

    # Margen por pasajero y total
    margin_pax_out = price_out - cost_pax
    margin_pax_in  = price_in  - cost_pax
    margin         = dem_out * margin_pax_out + dem_in * margin_pax_in

    rev_pax = revenue / total_dem if total_dem > 0 else 0

    dist_km = dist_csv[i, j] if dist_csv.shape[0] > i and dist_csv.shape[1] > j else 0

    rows.append({
        'Destino'        : dest,
        'Nombre'         : airports.loc[airports['iata'] == dest, 'name'].values[0],
        'Dist_km'        : round(dist_km),
        'Trans'          : is_trans,
        'Avion'          : 'B350' if is_trans else 'A321',
        'Cap_pax'        : int(a_nom),
        'TT_min'         : round(travel_time[i, j], 1),
        'Dem_MAD_dest'   : round(dem_out, 1),
        'Dem_dest_MAD'   : round(dem_in,  1),
        'Dem_total_sem'  : round(total_dem, 1),
        'Precio_EUR_pax' : round((price_out + price_in) / 2, 2),
        'Ingreso_sem_EUR': round(revenue),
        'Coste_op_sem_EUR': round(cost_pax * total_dem),
        'Margen_sem_EUR' : round(margin),
        'Rev_pax_EUR'    : round(rev_pax, 2),
        'Cost_pax_EUR'   : round(cost_pax, 2),
        'Margen_pax_EUR' : round(rev_pax - cost_pax, 2),
        'Margen_pct'     : round(100 * margin / revenue, 1) if revenue > 0 else 0,
    })

df = pd.DataFrame(rows).sort_values('Margen_sem_EUR', ascending=False).reset_index(drop=True)

# ── Mostrar ────────────────────────────────────────────────────────────────────
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 160)
pd.set_option('display.float_format', '{:,.0f}'.format)

print('\n' + '='*110)
print('  RENTABILIDAD POTENCIAL DE MERCADOS EN VUELO DIRECTO DESDE MAD')
print('='*110)
print(f'  Supuestos: A321 cap={A_NOM_NARROW:.0f} pax | B350 cap={A_NOM_WIDE:.0f} pax | '
      f'coste A321={OP_NARROW:,.0f} EUR/hr | coste B350={OP_WIDE:,.0f} EUR/hr | '
      f'coste/pax = op_cost_vuelo / capacidad_avion')
print('='*110)

cols_print = ['Destino', 'Rev_pax_EUR', 'Cost_pax_EUR', 'Cap_pax', 'Margen_pax_EUR', 'Margen_pct']

print(df[cols_print].to_string(index=True))

print('\n' + '-'*110)
print('TOTALES  (suma todos los mercados):')
print(f"  Demanda total semanal : {df['Dem_total_sem'].sum():>12,.0f} pax/sem")
print(f"  Ingresos totales      : {df['Ingreso_sem_EUR'].sum():>12,.0f} EUR/sem")
print(f"  Costes operativos     : {df['Coste_op_sem_EUR'].sum():>12,.0f} EUR/sem")
print(f"  Margen bruto          : {df['Margen_sem_EUR'].sum():>12,.0f} EUR/sem")
print(f"  Margen bruto %        : {100*df['Margen_sem_EUR'].sum()/df['Ingreso_sem_EUR'].sum():>11.1f}%")
print('-'*110)

# ── Guardar CSV ────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE, 'rentabilidad_directos.csv')
df.to_csv(out_path, index=False)
print(f'\nResultados guardados en: {out_path}')
