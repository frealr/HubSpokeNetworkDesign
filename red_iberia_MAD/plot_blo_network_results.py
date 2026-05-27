#!/usr/bin/env python3
"""
Plot network maps and hub connection traffic shares for BLO MAT results.

For each MAT file in hs_prueba_v0_blo, this script generates:
  - A network map with link width scaled by a(i,j)
  - A bar chart with the percentage of connection traffic handled by each hub

Connection traffic for airport i is measured from incident fij flows weighted by
demand(o,d), filtering OD pairs with o != i and d != i.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

from generate_data import build_network, TRANSATLANTIC_AIRPORTS


BASE_DIR = Path(__file__).resolve().parent
MAT_DIR = BASE_DIR / "hs_prueba_v0_blo"
OUT_DIR = BASE_DIR / "plots_blo"
PROJ = ccrs.PlateCarree()
IATA_RE = re.compile(r"^[A-Z]{3}$")


def _great_circle_pts(lon1, lat1, lon2, lat2, n_pts=100):
    r1 = np.radians([lat1, lon1])
    r2 = np.radians([lat2, lon2])

    def to_xyz(la, lo):
        return np.array(
            [
                np.cos(la) * np.cos(lo),
                np.cos(la) * np.sin(lo),
                np.sin(la),
            ]
        )

    v1 = to_xyz(*r1)
    v2 = to_xyz(*r2)
    omega = np.arccos(np.clip(np.dot(v1, v2), -1, 1))

    if omega < 1e-10:
        return np.array([lon1, lon2]), np.array([lat1, lat2])

    t = np.linspace(0, 1, n_pts)
    pts = (
        np.sin((1 - t) * omega)[:, None] * v1
        + np.sin(t * omega)[:, None] * v2
    ) / np.sin(omega)

    lats = np.degrees(np.arcsin(np.clip(pts[:, 2], -1, 1)))
    lons = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return lons, lats


def load_network_context():
    (
        n,
        nodes,
        idx,
        dist,
        link_cost,
        station_cost,
        hub_cost,
        link_capacity_slope,
        station_capacity_slope,
        demand,
        prices,
        load_factor,
        op_link_cost,
        congestion_coef_stations,
        congestion_coef_links,
        travel_time,
        alt_utility,
        a_nom,
        tau,
        eta,
        a_max,
        candidates,
        omega_t,
        omega_p,
    ) = build_network()
    del (
        n,
        idx,
        dist,
        link_cost,
        station_cost,
        hub_cost,
        link_capacity_slope,
        station_capacity_slope,
        load_factor,
        congestion_coef_stations,
        congestion_coef_links,
        travel_time,
        alt_utility,
        eta,
        a_max,
        candidates,
        omega_t,
        omega_p,
    )
    return nodes, demand, prices, op_link_cost, a_nom, tau


def load_airports(nodes):
    airports = pd.read_csv(BASE_DIR / "airports_full.csv")
    airports = airports[airports["iata"].isin(nodes)].copy()
    airports = airports.drop_duplicates(subset=["iata"]).set_index("iata")
    missing = [iata for iata in nodes if iata not in airports.index]
    if missing:
        raise ValueError(f"Missing airport coordinates for: {missing}")
    airports = airports.loc[nodes]
    return airports.to_dict("index")


def compute_map_extent(airports):
    lons = np.array([airport["lon"] for airport in airports.values()], dtype=float)
    lats = np.array([airport["lat"] for airport in airports.values()], dtype=float)
    lon_pad = max(8.0, 0.12 * (lons.max() - lons.min()))
    lat_pad = max(6.0, 0.18 * (lats.max() - lats.min()))
    return [
        float(lons.min() - lon_pad),
        float(lons.max() + lon_pad),
        float(lats.min() - lat_pad),
        float(lats.max() + lat_pad),
    ]


def load_result(mat_path):
    data = sio.loadmat(mat_path)
    fij = np.asarray(data["fij"], dtype=float)
    f = data.get("f")
    if f is None:
        # Fallback for MAT files that only store link-level OD flow.
        # By flow conservation, f(o,d) equals the outbound flow from origin o.
        f = np.sum(fij, axis=1)
    else:
        f = np.asarray(f, dtype=float)
    return {
        "s": np.ravel(data["s"]),
        "sh": np.ravel(data["sh"]),
        "a": np.asarray(data["a"], dtype=float),
        "f": np.asarray(f, dtype=float),
        "fij": fij,
    }


def compute_connection_percentages(fij, demand):
    n = demand.shape[0]
    percentages = np.zeros(n)
    total_traffic = np.zeros(n)
    connection_traffic = np.zeros(n)

    weighted_fij = fij * demand[None, None, :, :]

    for i in range(n):
        total = np.sum(weighted_fij[i, :, :, :])
        mask = np.ones((n, n), dtype=bool)
        mask[i, :] = False
        mask[:, i] = False
        connection = np.sum(weighted_fij[i, :, :, :] * mask[None, :, :])
        total_traffic[i] = total
        connection_traffic[i] = connection
        percentages[i] = 100.0 * connection / total if total > 1e-12 else 0.0

    return percentages, total_traffic, connection_traffic


def add_map_background(ax, map_extent):
    ax.set_extent(map_extent, crs=PROJ)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), color="#c8e6f5", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"), color="#f5f0e8", zorder=1)
    ax.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        linewidth=0.4,
        edgecolor="#888888",
        zorder=2,
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        linewidth=0.5,
        edgecolor="#666666",
        zorder=2,
    )
    ax.gridlines(draw_labels=False, linewidth=0.25, color="white", alpha=0.35)


def plot_network_map(mat_path, nodes, airports, sh, a):
    OUT_DIR.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=-12))
    add_map_background(ax, compute_map_extent(airports))

    A_REF = 6.0  # Fixed reference for absolute arc thickness scaling (global max_a is ~5.65)
    hub_mask = sh > 1e-2
    hub_scale = float(np.max(sh[hub_mask])) if np.any(hub_mask) else 1.0

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            aij = max(float(a[i, j]), float(a[j, i]))
            if aij <= 1e-2:
                continue

            airport_i = airports[nodes[i]]
            airport_j = airports[nodes[j]]
            lons, lats = _great_circle_pts(
                airport_i["lon"],
                airport_i["lat"],
                airport_j["lon"],
                airport_j["lat"],
            )
            rel = min(aij / A_REF, 1.0)
            lw = 0.6 + 5.4 * rel
            alpha = 0.25 + 0.55 * rel
            ax.plot(
                lons,
                lats,
                color="steelblue",
                linewidth=lw,
                alpha=alpha,
                transform=PROJ,
                zorder=3,
            )

    for i, iata in enumerate(nodes):
        airport = airports[iata]
        is_hub = sh[i] > 1e-2
        color = "red" if is_hub else "#24476b"
        size = 40
        if is_hub:
            size = 85 + 180 * sh[i] / hub_scale
        ax.scatter(
            airport["lon"],
            airport["lat"],
            s=size,
            color=color,
            edgecolor="black",
            linewidth=0.7 if is_hub else 0.5,
            transform=PROJ,
            zorder=5,
        )
        ax.text(
            airport["lon"] + 0.8,
            airport["lat"] + 0.5,
            iata,
            fontsize=7.5,
            fontweight="bold" if is_hub else "normal",
            color="darkred" if is_hub else "#16324f",
            transform=PROJ,
            zorder=6,
        )

    title = mat_path.stem
    ax.set_title(f"Network map - {title}", fontsize=12, pad=10)
    out_path = OUT_DIR / f"{mat_path.stem}_map.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_hub_connection_percentages(
    mat_path, nodes, sh, percentages, total_traffic, connection_traffic
):
    OUT_DIR.mkdir(exist_ok=True)
    hub_idx = np.where(sh > 1e-2)[0]
    if len(hub_idx) == 0:
        return None

    rows = []
    for i in hub_idx:
        rows.append(
            (
                nodes[i],
                float(percentages[i]),
                float(total_traffic[i]),
                float(connection_traffic[i]),
            )
        )
    rows.sort(key=lambda item: item[1], reverse=True)

    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, values, color="#c0392b", edgecolor="#6e1f16", linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Connection traffic share (%)")
    ax.set_xlabel("Hub")
    ax.set_title(f"Connection traffic handled by hub - {mat_path.stem}", fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out_path = OUT_DIR / f"{mat_path.stem}_hub_connection_pct.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path, rows


def compute_market_profitability(nodes, demand, prices, op_link_cost, a_nom, tau, f, fij):
    """Rentabilidad por pasajero y mercado para una solución dada."""
    n = len(nodes)
    rows = []
    for o in range(n):
        for d in range(n):
            if o == d or demand[o, d] <= 0 or f[o, d] < 1e-3:
                continue
            revenue_per_pax = prices[o, d]
            cost_per_pax = float(np.sum(
                fij[:, :, o, d] * op_link_cost / (a_nom * tau)
            ))
            margin_per_pax = revenue_per_pax - cost_per_pax
            pax_served = demand[o, d] * f[o, d]
            is_transatl = (nodes[o] in TRANSATLANTIC_AIRPORTS
                           or nodes[d] in TRANSATLANTIC_AIRPORTS)
            rows.append({
                "origin": nodes[o],
                "destination": nodes[d],
                "type": "transatlantic" if is_transatl else "regional",
                "f_iberia": float(f[o, d]),
                "demand_pax_week": float(demand[o, d]),
                "pax_served_week": float(pax_served),
                "price_eur": float(revenue_per_pax),
                "op_cost_per_pax": float(cost_per_pax),
                "margin_per_pax": float(margin_per_pax),
                "total_revenue_week": float(revenue_per_pax * pax_served),
                "total_op_cost_week": float(cost_per_pax * pax_served),
                "total_margin_week": float(margin_per_pax * pax_served),
            })
    return pd.DataFrame(rows).sort_values("total_margin_week", ascending=False).reset_index(drop=True)


def process_mat(mat_path, nodes, airports, demand, prices, op_link_cost, a_nom, tau):
    result = load_result(mat_path)
    map_path = plot_network_map(mat_path, nodes, airports, result["sh"], result["a"])
    percentages, total_traffic, connection_traffic = compute_connection_percentages(
        result["fij"], demand
    )
    pct_result = plot_hub_connection_percentages(
        mat_path,
        nodes,
        result["sh"],
        percentages,
        total_traffic,
        connection_traffic,
    )

    print(f"\n{mat_path.name}")
    print(f"  map: {map_path}")
    if pct_result is None:
        print("  hub connection chart: no hubs found")
    else:
        chart_path, rows = pct_result
        print(f"  hub connection chart: {chart_path}")
        for code, pct, total, conn in rows:
            print(
                f"    {code}: {pct:5.1f}%  "
                f"(connection={conn:10.2f}, total={total:10.2f})"
            )

    df_prof = compute_market_profitability(
        nodes, demand, prices, op_link_cost, a_nom, tau,
        result["f"], result["fij"]
    )
    if len(df_prof) > 0:
        print(f"\n  --- Rentabilidad por mercado (top 10 por margen total) ---")
        print(f"  {'OD':<12} {'Tipo':<14} {'f':>5} {'Pax/sem':>8} "
              f"{'€/pax':>8} {'Cost/pax':>9} {'Margen/pax':>11} {'Margen total':>13}")
        print(f"  {'-'*85}")
        for _, r in df_prof.head(10).iterrows():
            od = f"{r['origin']}-{r['destination']}"
            print(f"  {od:<12} {r['type']:<14} {r['f_iberia']:5.3f} "
                  f"{r['pax_served_week']:8.0f} {r['price_eur']:8.0f} "
                  f"{r['op_cost_per_pax']:9.0f} {r['margin_per_pax']:11.0f} "
                  f"{r['total_margin_week']:13.0f}")
        tot_rev = df_prof["total_revenue_week"].sum()
        tot_cost = df_prof["total_op_cost_week"].sum()
        tot_margin = df_prof["total_margin_week"].sum()
        print(f"\n  TOTAL semana — Ingresos: {tot_rev:,.0f} € | "
              f"Costes op: {tot_cost:,.0f} € | Margen: {tot_margin:,.0f} €")
        avg_margin = tot_margin / df_prof["pax_served_week"].sum() if df_prof["pax_served_week"].sum() > 0 else 0
        print(f"  Margen medio por pasajero: {avg_margin:.0f} €/pax")

        OUT_DIR.mkdir(exist_ok=True)
        csv_path = OUT_DIR / f"{mat_path.stem}_rentabilidad.csv"
        df_prof.to_csv(csv_path, index=False)
        print(f"  CSV guardado: {csv_path}")


def main():
    nodes, demand, prices, op_link_cost, a_nom, tau = load_network_context()
    airports = load_airports(nodes)

    mat_files = sorted(MAT_DIR.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No MAT files found in {MAT_DIR}")

    for mat_path in mat_files:
        process_mat(mat_path, nodes, airports, demand, prices, op_link_cost, a_nom, tau)


if __name__ == "__main__":
    main()
