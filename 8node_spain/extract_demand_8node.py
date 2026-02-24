import pandas as pd
import numpy as np

# Configuration
excel_file = 'CO261_Flujos_Viajeros_entre_Aerops_Espanyoles__Anyo.xlsx'
target_airports = ['MAD', 'BCN', 'PMI', 'AGP', 'ALC', 'LPA', 'TFS', 'IBZ']
output_file = 'demand.csv'

try:
    # Read the Excel file without assuming header structure
    df = pd.read_excel(excel_file, header=None)

    n = len(target_airports)
    demand_matrix = np.zeros((n, n))

    # Find row indices (origins are in column 2, index 2)
    row_map = {}
    col_map = {}

    for idx, val in df.iloc[:, 2].items():
        if pd.isna(val):
            continue
        val_str = str(val)
        for code in target_airports:
            if code in val_str:
                row_map[code] = idx
                print(f"Found Row for {code}: {idx} ({val_str})")

    # Find column indices (header in first 10 rows)
    for r in range(10):
        for c, val in df.iloc[r, :].items():
            if pd.isna(val):
                continue
            val_str = str(val)
            for code in target_airports:
                if code in val_str:
                    if code not in col_map:  # Take first occurrence
                        col_map[code] = c
                        print(f"Found Col for {code}: {c} ({val_str}) at Row {r}")

    # Check if we found everything
    missing_rows = set(target_airports) - set(row_map.keys())
    missing_cols = set(target_airports) - set(col_map.keys())

    if missing_rows:
        print(f"Missing rows for: {missing_rows}")
    if missing_cols:
        print(f"Missing cols for: {missing_cols}")

    if not missing_rows and not missing_cols:
        # Extract data
        for i in range(n):
            for j in range(n):
                o = target_airports[i]
                d = target_airports[j]

                r_idx = row_map[o]
                c_idx = col_map[d]

                val = df.iloc[r_idx, c_idx]

                if pd.isna(val):
                    val = 0

                demand_matrix[i, j] = val

        # Save to CSV
        np.savetxt(output_file, demand_matrix, delimiter=',')
        print(f"\nSuccessfully saved {output_file}")
        print("Demand matrix (8 airports: MAD, BCN, PMI, AGP, ALC, LPA, TFS, IBZ):")
        print(demand_matrix)

    else:
        print("Could not generate matrix due to missing mappings.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
