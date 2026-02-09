
import pandas as pd

try:
    df = pd.read_excel('CO261_Flujos_Viajeros_entre_Aerops_Espanyoles__Anyo.xlsx', header=None)
    
    print("Row 1 (Destinations):")
    for i, val in enumerate(df.iloc[1, :]):
        print(f"Col {i}: {val}")
        
    print("\nColumn 2 (Origins):")
    for i, val in enumerate(df.iloc[:, 2]):
        print(f"Row {i}: {val}")
        
except Exception as e:
    print(f"Error: {e}")
