"""
Generate airports CSV and demand network map for red_iberia.

Reads demanda_completa sheet from demanda.xlsx, extracts all valid IATA nodes,
writes a CSV with coordinates, and plots a world map with great-circle arcs
for every OD pair with demand > 0.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Airport coordinates (IATA → (name, lat, lon)) ────────────────────────────
AIRPORTS = {
    'ACE': ('Lanzarote',            28.9455,  -13.6052),
    'AGP': ('Malaga',               36.6749,   -4.4991),
    'ALC': ('Alicante',             38.2822,   -0.5582),
    'AMM': ('Amman',                31.7226,   35.9932),
    'AMS': ('Amsterdam',            52.3086,    4.7639),
    'ARN': ('Stockholm',            59.6519,   17.9186),
    'ATH': ('Athens',               37.9364,   23.9445),
    'ATL': ('Atlanta',              33.6367,  -84.4281),
    'BCN': ('Barcelona',            41.2971,    2.0785),
    'BIO': ('Bilbao',               43.3010,   -2.9106),
    'BOG': ('Bogota',                4.7016,  -74.1469),
    'BOS': ('Boston',               42.3656,  -71.0096),
    'BRU': ('Brussels',             50.9014,    4.4844),
    'BUD': ('Budapest',             47.4298,   19.2611),
    'CAI': ('Cairo',                30.1219,   31.4056),
    'CCS': ('Caracas',              10.6031,  -66.9913),
    'CDG': ('Paris CDG',            49.0097,    2.5479),
    'CMN': ('Casablanca',           33.3675,   -7.5898),
    'CPH': ('Copenhagen',           55.6180,   12.6508),
    'CUZ': ('Cusco',               -13.5357,  -71.9388),
    'DFW': ('Dallas',               32.8998,  -97.0403),
    'DKR': ('Dakar',                14.7397,  -17.4902),
    'DME': ('Moscow Domodedovo',    55.4088,   37.9063),
    'DUB': ('Dublin',               53.4213,   -6.2701),
    'DUS': ('Dusseldorf',           51.2895,    6.7668),
    'EAS': ('San Sebastian',        43.3565,   -1.7906),
    'EDI': ('Edinburgh',            55.9500,   -3.3725),
    'EZE': ('Buenos Aires',        -34.8222,  -58.5358),
    'FCO': ('Rome',                 41.8003,   12.2389),
    'FRA': ('Frankfurt',            50.0333,    8.5706),
    'FUE': ('Fuerteventura',        28.4527,  -13.8638),
    'GIG': ('Rio de Janeiro',      -22.8099,  -43.2505),
    'GRU': ('Sao Paulo',           -23.4356,  -46.4731),
    'GRX': ('Granada',              37.1887,   -3.7774),
    'GVA': ('Geneva',               46.2380,    6.1089),
    'GYE': ('Guayaquil',            -2.1574,  -79.8836),
    'HAV': ('Havana',               22.9892,  -82.4091),
    'HEL': ('Helsinki',             60.3172,   24.9633),
    'IAD': ('Washington Dulles',    38.9531,  -77.4565),
    'IBZ': ('Ibiza',                38.8729,    1.3731),
    'IST': ('Istanbul',             41.2753,   28.7519),
    'JFK': ('New York JFK',         40.6413,  -73.7781),
    'JNB': ('Johannesburg',        -26.1367,   28.2411),
    'KBP': ('Kiev',                 50.3450,   30.8947),
    'LAS': ('Las Vegas',            36.0840, -115.1537),
    'LAX': ('Los Angeles',          33.9425, -118.4081),
    'LCG': ('La Coruna',            43.3021,   -8.3771),
    'LEI': ('Almeria',              36.8439,   -2.3701),
    'LGW': ('London Gatwick',       51.1481,   -0.1903),
    'LHR': ('London Heathrow',      51.4775,   -0.4614),
    'LIM': ('Lima',                -12.0219,  -77.1143),
    'LIN': ('Milan Linate',         45.4454,    9.2767),
    'LIS': ('Lisbon',               38.7813,   -9.1359),
    'LOS': ('Lagos',                 6.5774,    3.3214),
    'LPA': ('Las Palmas',           27.9319,  -15.3866),
    'LYS': ('Lyon',                 45.7256,    5.0811),
    'MAD': ('Madrid',               40.4719,   -3.5626),
    'MAH': ('Menorca',              39.8626,    4.2186),
    'MAN': ('Manchester',           53.3537,   -2.2750),
    'MCO': ('Orlando',              28.4294,  -81.3089),
    'MEX': ('Mexico City',          19.4363,  -99.0721),
    'MIA': ('Miami',                25.7959,  -80.2870),
    'MLN': ('Melilla',              35.2799,   -2.9562),
    'MRS': ('Marseille',            43.4393,    5.2214),
    'MSP': ('Minneapolis',          44.8848,  -93.2223),
    'MUC': ('Munich',               48.3537,   11.7750),
    'MXP': ('Milan Malpensa',       45.6306,    8.7281),
    'NCE': ('Nice',                 43.6584,    7.2159),
    'OPO': ('Porto',                41.2481,   -8.6814),
    'ORD': ('Chicago',              41.9742,  -87.9073),
    'ORY': ('Paris Orly',           48.7233,    2.3794),
    'OTP': ('Bucharest',            44.5711,   26.0850),
    'OVD': ('Asturias',             43.5636,   -6.0346),
    'PMI': ('Palma de Mallorca',    39.5517,    2.7388),
    'PNA': ('Pamplona',             42.7700,   -1.6465),
    'PRG': ('Prague',               50.1008,   14.2600),
    'RAK': ('Marrakech',            31.6069,   -8.0363),
    'SCQ': ('Santiago Compostela',  42.8963,   -8.4151),
    'SDQ': ('Santo Domingo',        18.4297,  -69.6689),
    'SDR': ('Santander',            43.4271,   -3.8200),
    'SFO': ('San Francisco',        37.6213, -122.3790),
    'SOF': ('Sofia',                42.6952,   23.4114),
    'SVO': ('Moscow Sheremetyevo',  55.9726,   37.4147),
    'SVQ': ('Seville',              37.4180,   -5.8931),
    'TFN': ('Tenerife Norte',       28.4827,  -16.3415),
    'TFS': ('Tenerife Sur',         28.0445,  -16.5725),
    'TLS': ('Toulouse',             43.6293,    1.3678),
    'TLV': ('Tel Aviv',             32.0114,   34.8867),
    'TNG': ('Tangier',              35.7269,   -5.9169),
    'TXL': ('Berlin Tegel',         52.5597,   13.2877),
    'UIO': ('Quito',                -0.1292,  -78.3575),
    'VCE': ('Venice',               45.5053,   12.3519),
    'VGO': ('Vigo',                 42.2318,   -8.6277),
    'VIE': ('Vienna',               48.1103,   16.5697),
    'VLC': ('Valencia',             39.4893,   -0.4816),
    'WAW': ('Warsaw',               52.1657,   20.9671),
    'XRY': ('Jerez',                36.7446,   -6.0600),
    'YYZ': ('Toronto',              43.6772,  -79.6306),
    'ZRH': ('Zurich',               47.4647,    8.5492),
}

PROJ = ccrs.PlateCarree()


def _great_circle_pts(lon1, lat1, lon2, lat2, n_pts=80):
    r1 = np.radians([lat1, lon1])
    r2 = np.radians([lat2, lon2])

    def to_xyz(la, lo):
        return np.array([np.cos(la)*np.cos(lo),
                         np.cos(la)*np.sin(lo),
                         np.sin(la)])

    v1 = to_xyz(*r1)
    v2 = to_xyz(*r2)
    omega = np.arccos(np.clip(np.dot(v1, v2), -1, 1))

    if omega < 1e-10:
        return np.array([lon1, lon2]), np.array([lat1, lat2])

    t = np.linspace(0, 1, n_pts)
    pts = (np.sin((1 - t) * omega)[:, None] * v1 +
           np.sin(t * omega)[:, None] * v2) / np.sin(omega)

    lats = np.degrees(np.arcsin(np.clip(pts[:, 2], -1, 1)))
    lons = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return lons, lats


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = os.path.join(base_dir, 'demanda.xlsx')

    # ── Load demand data: top-25 aeropuertos por demanda total ───────────────
    from collections import defaultdict
    TOP_N = 25
    df_full = pd.read_excel(xlsx_path, sheet_name='demanda_completa')
    df_full = df_full.dropna(subset=['Origen', 'Destino'])
    iata_re = re.compile(r'^[A-Z]{3}$')
    df_full = df_full[df_full['Origen'].apply(lambda x: bool(iata_re.match(str(x)))) &
                      df_full['Destino'].apply(lambda x: bool(iata_re.match(str(x))))]
    df_full = df_full[df_full['Promedio de Suma de Demanda'] > 0]

    dem = defaultdict(float)
    for _, r in df_full.iterrows():
        dem[r['Origen']] += r['Promedio de Suma de Demanda']
        dem[r['Destino']] += r['Promedio de Suma de Demanda']
    top_nodes = set(sorted(dem, key=dem.get, reverse=True)[:TOP_N])

    df = df_full[df_full['Origen'].isin(top_nodes) & df_full['Destino'].isin(top_nodes)]
    all_nodes = sorted(top_nodes)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    rows = []
    missing = []
    for iata in all_nodes:
        if iata in AIRPORTS:
            name, lat, lon = AIRPORTS[iata]
            rows.append({'iata': iata, 'name': name, 'lat': lat, 'lon': lon})
        else:
            missing.append(iata)

    if missing:
        print(f"WARNING: No coordinates for: {missing}")

    csv_path = os.path.join(base_dir, 'airports.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved {len(rows)} airports → {csv_path}")

    # ── Plot map ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 13))
    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.Robinson(central_longitude=-10))
    ax.set_global()

    # Background
    ax.add_feature(cfeature.OCEAN.with_scale('110m'),    color='#c8e6f5', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('110m'),     color='#f5f0e8', zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale('110m'),  linewidth=0.4,
                   edgecolor='#888888', zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5,
                   edgecolor='#666666', zorder=2)

    # Arc demand: normalise thickness by demand
    max_demand = df['Promedio de Suma de Demanda'].max()

    for _, row in df.iterrows():
        o, d = row['Origen'], row['Destino']
        if o not in AIRPORTS or d not in AIRPORTS:
            continue
        _, lat1, lon1 = AIRPORTS[o]
        _, lat2, lon2 = AIRPORTS[d]
        lons, lats = _great_circle_pts(lon1, lat1, lon2, lat2)
        demand = row['Promedio de Suma de Demanda']
        lw = 0.3 + 2.0 * (demand / max_demand)
        alpha = 0.25 + 0.55 * (demand / max_demand)
        ax.plot(lons, lats, color='steelblue', linewidth=lw, alpha=alpha,
                transform=PROJ, zorder=3)

    out_dir = os.path.join(base_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    max_demand = df['Promedio de Suma de Demanda'].max()

    def _draw_map(ax, nodes, demand_df, label_offset=1.0, node_ms=6, arc_lw_max=2.0, fontsize=5.5):
        ax.add_feature(cfeature.OCEAN.with_scale('110m'),    color='#c8e6f5', zorder=0)
        ax.add_feature(cfeature.LAND.with_scale('110m'),     color='#f5f0e8', zorder=1)
        ax.add_feature(cfeature.BORDERS.with_scale('110m'),  linewidth=0.4,
                       edgecolor='#888888', zorder=2)
        ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5,
                       edgecolor='#666666', zorder=2)

        for _, row in demand_df.iterrows():
            o, d = row['Origen'], row['Destino']
            if o not in AIRPORTS or d not in AIRPORTS:
                continue
            _, lat1, lon1 = AIRPORTS[o]
            _, lat2, lon2 = AIRPORTS[d]
            lons, lats = _great_circle_pts(lon1, lat1, lon2, lat2)
            demand = row['Promedio de Suma de Demanda']
            lw = 0.3 + arc_lw_max * (demand / max_demand)
            alpha = 0.25 + 0.55 * (demand / max_demand)
            ax.plot(lons, lats, color='steelblue', linewidth=lw, alpha=alpha,
                    transform=PROJ, zorder=3)

        for iata in nodes:
            if iata not in AIRPORTS:
                continue
            _, lat, lon = AIRPORTS[iata]
            ax.plot(lon, lat, 'o', color='#ff7f0e', markersize=node_ms,
                    markeredgecolor='black', markeredgewidth=0.6,
                    transform=PROJ, zorder=5)
            ax.text(lon + label_offset, lat + label_offset, iata, transform=PROJ,
                    fontsize=fontsize, fontweight='bold', color='black', zorder=6,
                    clip_on=True)

    # ── World map ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 13))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=-10))
    ax.set_global()
    _draw_map(ax, all_nodes, df)
    ax.set_title('Red Iberia — pares OD con demanda > 0', fontsize=14,
                 fontweight='bold', pad=10)
    out_path = os.path.join(out_dir, 'red_iberia_demand_map.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved world map → {out_path}")

    # ── Europe map ────────────────────────────────────────────────────────────
    europe_extent = [-25, 45, 28, 72]  # lon_min, lon_max, lat_min, lat_max

    def _in_europe(iata):
        if iata not in AIRPORTS:
            return False
        _, lat, lon = AIRPORTS[iata]
        return (europe_extent[0] <= lon <= europe_extent[1] and
                europe_extent[2] <= lat <= europe_extent[3])

    europe_nodes = [n for n in all_nodes if _in_europe(n)]
    df_eur = df[df['Origen'].apply(_in_europe) & df['Destino'].apply(_in_europe)]

    fig2 = plt.figure(figsize=(16, 12))
    ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.LambertConformal(
        central_longitude=10, central_latitude=50))
    ax2.set_extent(europe_extent, crs=PROJ)
    _draw_map(ax2, europe_nodes, df_eur, label_offset=0.3, node_ms=7,
              arc_lw_max=3.0, fontsize=7.0)
    ax2.set_title('Red Iberia — Europa, pares OD con demanda > 0', fontsize=14,
                  fontweight='bold', pad=10)
    out_path_eur = os.path.join(out_dir, 'red_iberia_demand_map_europa.png')
    fig2.savefig(out_path_eur, dpi=180, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved Europe map → {out_path_eur}")

    print(f"\nTotal nodes: {len(all_nodes)}")
    print(f"Total OD arcs (world): {len(df)}")
    print(f"Europe nodes: {len(europe_nodes)}, Europe arcs: {len(df_eur)}")


if __name__ == '__main__':
    main()
