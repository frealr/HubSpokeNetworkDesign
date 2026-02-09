
import pandas as pd

try:
    df = pd.read_excel('CO261_Flujos_Viajeros_entre_Aerops_Espanyoles__Anyo.xlsx', header=None)
    
    print("Row 0:")
    print(df.iloc[0, :].tolist())
    
    print("\nRow 1:")
    print(df.iloc[1, :].tolist())
    
    print("\nRow 2:")
    print(df.iloc[2, :].tolist())
    
except Exception as e:
    print(f"Error: {e}")
