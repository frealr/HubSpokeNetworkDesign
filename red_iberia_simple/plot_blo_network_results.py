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


BASE_DIR = Path(__file__).resolve().parent
MAT_DIR = BASE_DIR / "hs_prueba_v0_blo"
OUT_DIR = BASE_DIR / "plots_blo"
PROJ = ccrs.PlateCarree()
MAP_EXTENT = [-100, 20, 18, 62]
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


def load_node_order():
    node_map = pd.read_csv(BASE_DIR / "node_mapping.csv")
    return node_map["iata"].tolist()


def load_airports():
    airports = pd.read_csv(BASE_DIR / "airports.csv")
    return airports.set_index("iata").to_dict("index")


def load_demand(nodes):
    df = pd.read_excel(BASE_DIR / "demanda.xlsx", sheet_name="demanda_completa")
    df = df.dropna(subset=["Origen", "Destino"])
    df = df[
        df["Origen"].apply(lambda x: bool(IATA_RE.match(str(x))))
        & df["Destino"].apply(lambda x: bool(IATA_RE.match(str(x))))
    ]
    df = df[df["Promedio de Suma de Demanda"] > 0]
    df = df[df["Origen"].isin(nodes) & df["Destino"].isin(nodes)]

    idx = {iata: i for i, iata in enumerate(nodes)}
    demand = np.zeros((len(nodes), len(nodes)))
    for _, row in df.iterrows():
        demand[idx[row["Origen"]], idx[row["Destino"]]] = float(
            row["Promedio de Suma de Demanda"]
        )
    return demand


def load_result(mat_path):
    data = sio.loadmat(mat_path)
    return {
        "s": np.ravel(data["s"]),
        "sh": np.ravel(data["sh"]),
        "a": np.asarray(data["a"], dtype=float),
        "fij": np.asarray(data["fij"], dtype=float),
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


def add_map_background(ax):
    ax.set_extent(MAP_EXTENT, crs=PROJ)
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
    add_map_background(ax)

    max_a = float(np.max(a)) if np.max(a) > 1e-12 else 1.0
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
            rel = aij / max_a
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


def process_mat(mat_path, nodes, airports, demand):
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
        return

    chart_path, rows = pct_result
    print(f"  hub connection chart: {chart_path}")
    for code, pct, total, conn in rows:
        print(
            f"    {code}: {pct:5.1f}%  "
            f"(connection={conn:10.2f}, total={total:10.2f})"
        )


def main():
    nodes = load_node_order()
    airports = load_airports()
    demand = load_demand(nodes)

    mat_files = sorted(MAT_DIR.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No MAT files found in {MAT_DIR}")

    for mat_path in mat_files:
        process_mat(mat_path, nodes, airports, demand)


if __name__ == "__main__":
    main()
