import pandas as pd
import numpy as np

# Configuration
excel_file = '../4node_spain/CO261_Flujos_Viajeros_entre_Aerops_Espanyoles__Anyo.xlsx'
target_airports = ['MAD', 'BCN', 'PMI', 'AGP', 'ALC', 'LPA']
output_file = 'demand.csv'

try:
    # Read the Excel file, assuming header is not in the first row implicitly or it is messy
    # Based on inspection, it seems the data starts around row 4 or 5, but we can search for codes.
    # Let's read the whole thing and search.
    df = pd.read_excel(excel_file, header=None)
    
    # Locate the airport codes in the first column (Origin) and first row (Destination)
    # The output from inspection showed:
    # Row 1 (index 1) has destinations (mostly NaN then names)
    # Col 2 (index 2) has origins? No, let's look at the output again.
    
    # From previous output:
    # 9 NaN NaN MAD : Madrid-Barajas ...
    # So column 2 (index 2) contains the row labels.
    
    # We need to find the row index for each target airport in Column 2
    # And the column index for each target airport in Row 1 (or wherever the header is)
    
    # Let's assume the matrix structure:
    # Rows: Origins
    # Cols: Destinations
    
    n = len(target_airports)
    demand_matrix = np.zeros((n, n))
    
    # Find mappings
    row_map = {}
    col_map = {}
    
    # Find row indices
    for idx, val in df.iloc[:, 2].items():
        if pd.isna(val): continue
        val_str = str(val)
        for code in target_airports:
            if code in val_str:
                row_map[code] = idx
                print(f"Found Row for {code}: {idx} ({val_str})")
                
    # Find column indices. The header seems to be in a row above the data.
    # Based on inspection output, row 1 had 'MAD : Madrid-Barajas' at some columns.
    # Let's search in the first 10 rows for matches.
    
    for r in range(10):
        for c, val in df.iloc[r, :].items():
            if pd.isna(val): continue
            val_str = str(val)
            for code in target_airports:
                if code in val_str:
                    if code not in col_map: # Take first occurrence
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
                
                # Check for NaN
                if pd.isna(val):
                    val = 0
                
                demand_matrix[i, j] = val
        
        # Save to CSV
        np.savetxt(output_file, demand_matrix, delimiter=',')
        print(f"Successfully saved {output_file}")
        print(demand_matrix)
        
    else:
        print("Could not generate matrix due to missing mappings.")

except Exception as e:
    print(f"An error occurred: {e}")
