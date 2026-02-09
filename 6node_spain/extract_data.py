
import pandas as pd
import numpy as np
import math

airports = ['MAD', 'BCN', 'PMI', 'AGP', 'ALC', 'LPA']
coords = {
    'MAD': (40.4983, -3.5676),
    'BCN': (41.2974, 2.0833),
    'PMI': (39.5517, 2.7388),
    'AGP': (36.6749, -4.4991),
    'ALC': (38.2822, -0.5582),
    'LPA': (27.9319, -15.3866)
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

n = len(airports)
dist_matrix = np.zeros((n, n))
yield_matrix = np.zeros((n, n))

# 1. Distances
# Initialize with Haversine
for i in range(n):
    for j in range(n):
        if i == j: continue
        o, d = airports[i], airports[j]
        dist_matrix[i, j] = haversine(coords[o][0], coords[o][1], coords[d][0], coords[d][1])

# Overwrite with Excel data if available
try:
    df_dist = pd.read_excel('Distancias_España.xlsx')
    df_dist.columns = df_dist.columns.str.strip()
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            o, d = airports[i], airports[j]
            
            # Look for pair
            row = df_dist[((df_dist['Origen'] == o) & (df_dist['Destino'] == d)) | 
                          ((df_dist['Origen'] == d) & (df_dist['Destino'] == o))]
            
            if not row.empty:
                # Take the first one found
                val = row['Distancia'].values[0]
                dist_matrix[i, j] = val
except Exception as e:
    print(f"Error reading distances: {e}")

# 2. Yields
global_yields = []

try:
    df_yield = pd.read_excel('YieldPKTnacional.xlsx')
    df_yield.columns = df_yield.columns.str.strip()
    
    # Calculate average yield for each pair
    for i in range(n):
        for j in range(n):
            if i == j: continue
            o, d = airports[i], airports[j]
            
            rows = df_yield[((df_yield['Origen'] == o) & (df_yield['Destino'] == d)) | 
                            ((df_yield['Origen'] == d) & (df_yield['Destino'] == o))]
            
            if not rows.empty:
                avg_yield = rows['YieldPKT'].mean()
                yield_matrix[i, j] = avg_yield
                global_yields.append(avg_yield)
            else:
                yield_matrix[i, j] = np.nan # Mark as missing

    # Fill missing with global average
    if global_yields:
        avg_global = np.mean(global_yields)
    else:
        avg_global = 15.0 # Default fallback
        
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if np.isnan(yield_matrix[i, j]):
                yield_matrix[i, j] = avg_global
                
    # Calculate Prices
    # Yield is in cents/km. Distance is in km.
    # Price = (Yield * Distance) / 100 (to get Euros)
    price_matrix = (dist_matrix * yield_matrix) / 100.0

except Exception as e:
    print(f"Error reading yields: {e}")
    price_matrix = dist_matrix * 0.15 # Fallback

# Save to CSV
# We use simple CSV format without headers/indices for easy MATLAB reading
pd.DataFrame(dist_matrix).to_csv('distance.csv', index=False, header=False)
pd.DataFrame(price_matrix).to_csv('prices.csv', index=False, header=False)

print("Generated distance.csv and prices.csv")
