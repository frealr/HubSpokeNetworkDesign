
import pandas as pd
import numpy as np

files = ['Distancias_España.xlsx', 'YieldPKTnacional.xlsx']
airports = ['MAD', 'BCN', 'PMI', 'AGP', 'ALC', 'LPA', 'TFS', 'IBZ']
n = len(airports)

# Initialize matrices
dist_matrix = np.zeros((n, n))
price_matrix = np.zeros((n, n))

# Read Distances
try:
    df_dist = pd.read_excel('Distancias_España.xlsx')
    # Ensure columns are stripped of whitespace if any
    df_dist.columns = df_dist.columns.str.strip()
    
    # Create a dictionary for quick lookup
    # Assuming 'Origen' and 'Destino' are the airport codes
    # Check if codes are consistent (uppercase etc)
    
    for i, origin in enumerate(airports):
        for j, dest in enumerate(airports):
            if i == j:
                dist_matrix[i, j] = 0
                continue
            
            # Find row
            row = df_dist[((df_dist['Origen'] == origin) & (df_dist['Destino'] == dest)) | 
                          ((df_dist['Origen'] == dest) & (df_dist['Destino'] == origin))]
            
            if not row.empty:
                # Assuming 'Distancia' is in km or similar. 
                # The user script uses 1e4 scaling or something, but let's just get the raw value first.
                # In the original script: distance(1, 2) = 0.75; which seems small. 
                # Maybe it's 1000km units? Or just normalized?
                # The user said "extrae las distancias ... del excel".
                # I will print the raw values and we can adjust scaling later if needed.
                val = row['Distancia'].values[0]
                dist_matrix[i, j] = val
            else:
                print(f"Warning: No distance found for {origin}-{dest}")

except Exception as e:
    print(f"Error reading distances: {e}")

# Read Prices
try:
    df_price = pd.read_excel('YieldPKTnacional.xlsx')
    df_price.columns = df_price.columns.str.strip()
    
    for i, origin in enumerate(airports):
        for j, dest in enumerate(airports):
            if i == j:
                price_matrix[i, j] = 0
                continue
            
            row = df_price[((df_price['Origen'] == origin) & (df_price['Destino'] == dest)) | 
                           ((df_price['Origen'] == dest) & (df_price['Destino'] == origin))]
             
            if not row.empty:
                val = row['YieldPKT'].values[0]
                price_matrix[i, j] = val
            else:
                # Try to find average or something if missing? Or just 0?
                # For now warn
                print(f"Warning: No price found for {origin}-{dest}")

except Exception as e:
    print(f"Error reading prices: {e}")

# Print in MATLAB format
print("\n% Distance Matrix (Raw from Excel)")
print("distance = [")
for i in range(n):
    row_str = ", ".join(f"{x:.4f}" for x in dist_matrix[i])
    print(f"    {row_str};")
print("];")

print("\n% Price Matrix (Raw from Excel - YieldPKT)")
print("prices_raw = [")
for i in range(n):
    row_str = ", ".join(f"{x:.4f}" for x in price_matrix[i])
    print(f"    {row_str};")
print("];")
