#!/usr/bin/env python3
"""Exporta el mapeo entre ids internos de nodo e IATA."""

import os

import pandas as pd

from generate_data import BASE_DIR, build_network


def main():
    net_params = build_network()
    nodes = net_params[1]

    df = pd.DataFrame({
        "node_id": [f"i{k}" for k in range(1, len(nodes) + 1)],
        "iata": nodes,
    })

    out_path = os.path.join(BASE_DIR, "node_mapping.csv")
    df.to_csv(out_path, index=False)

    print(f"Nodos exportados: {len(df)}")
    print(f"Fichero: {out_path}")


if __name__ == "__main__":
    main()
