#!/usr/bin/env python3
"""Exporta un diagnostico por mercado OD para calibrar la logit."""

import os

import numpy as np
import pandas as pd

from generate_data import (
    BASE_DIR,
    assign_yield,
    build_network,
    get_competitor_airline_count,
)


def stable_direct_share(u_direct, u_alt, n_airlines):
    """Cuota directa frente a n_airlines alternativas en forma estable."""
    umax = max(u_direct, u_alt)
    num = np.exp(u_direct - umax)
    den = num + n_airlines * np.exp(u_alt - umax)
    return num / den


def build_airport_potential(df):
    potential_df = df.copy()
    potential_df["market_potential"] = (
        potential_df["price_eur"]
        * potential_df["demand_pax_week"]
        * potential_df["direct_share_bound"]
    )

    rows = []
    airports = sorted(set(potential_df["origin"]).union(potential_df["destination"]))
    for airport in airports:
        origin_mask = potential_df["origin"] == airport
        destination_mask = potential_df["destination"] == airport
        market_mask = origin_mask | destination_mask
        rows.append({
            "airport": airport,
            "potential_total": float(potential_df.loc[market_mask, "market_potential"].sum()),
            "markets_count": int(market_mask.sum()),
            "origin_potential": float(potential_df.loc[origin_mask, "market_potential"].sum()),
            "destination_potential": float(potential_df.loc[destination_mask, "market_potential"].sum()),
        })

    return pd.DataFrame(rows).sort_values("potential_total", ascending=False).reset_index(drop=True)


def load_yield_assigner():
    dy = pd.read_excel(os.path.join(BASE_DIR, "yield.xlsx"), sheet_name="yield")
    y_market = dy.groupby(["Origen", "Destino"])["Yield-PKT"].mean()
    global_mean = dy["Yield-PKT"].mean()
    return assign_yield(y_market, global_mean)


def main():
    n_airlines = get_competitor_airline_count()
    (n, nodes, idx, dist, link_cost, station_cost, hub_cost,
     link_capacity_slope, station_capacity_slope, demand, prices,
     load_factor, op_link_cost, congestion_coef_stations,
     congestion_coef_links, travel_time, alt_utility,
     a_nom, tau, eta, a_max, candidates, omega_t, omega_p) = build_network()

    del (idx, link_cost, station_cost, hub_cost, link_capacity_slope,
         station_capacity_slope, load_factor, op_link_cost,
         congestion_coef_stations, congestion_coef_links, a_nom,
         tau, eta, a_max, candidates, n)

    get_yield = load_yield_assigner()
    rows = []

    for i, origin in enumerate(nodes):
        for j, destination in enumerate(nodes):
            if i == j or demand[i, j] <= 0:
                continue

            market_yield, yield_source = get_yield(origin, destination)
            direct_utility = omega_p * prices[i, j] + omega_t * travel_time[i, j]
            alternative_utility = alt_utility[i, j]

            rows.append({
                "origin": origin,
                "destination": destination,
                "demand_pax_week": float(demand[i, j]),
                "yield_cents_per_km": float(market_yield),
                "yield_eur_per_km": float(market_yield / 100.0),
                "yield_source": yield_source,
                "distance_km": float(dist[i, j]),
                "price_eur": float(prices[i, j]),
                "travel_time_min": float(travel_time[i, j]),
                "direct_utility": float(direct_utility),
                "alternative_utility": float(alternative_utility),
                "utility_gap_direct_minus_alt": float(direct_utility - alternative_utility),
                "direct_share_bound": float(
                    stable_direct_share(direct_utility, alternative_utility, n_airlines)
                ),
            })

    df = pd.DataFrame(rows).sort_values(["origin", "destination"]).reset_index(drop=True)
    airport_potential = build_airport_potential(df)
    out_path = os.path.join(BASE_DIR, "market_diagnostics.csv")
    out_xlsx_path = os.path.join(BASE_DIR, "market_diagnostics.xlsx")
    df.to_csv(out_path, index=False)
    with pd.ExcelWriter(out_xlsx_path) as writer:
        df.to_excel(writer, sheet_name="market_diagnostics", index=False)
        airport_potential.to_excel(writer, sheet_name="airport_potential", index=False)

    print(f"Mercados exportados: {len(df)}")
    print(f"Fichero: {out_path}")
    print(f"Excel: {out_xlsx_path}")
    print("Resumen utility_gap_direct_minus_alt:")
    print(df["utility_gap_direct_minus_alt"].describe().to_string())


if __name__ == "__main__":
    main()
