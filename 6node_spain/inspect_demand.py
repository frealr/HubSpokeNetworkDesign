
import pandas as pd

try:
    df = pd.read_excel('CO261_Flujos_Viajeros_entre_Aerops_Espanyoles__Anyo.xlsx')
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for airport codes
    print("\nUnique values in first column (assuming Origin):")
    print(df.iloc[:, 0].unique()[:10])
    
except Exception as e:
    print(f"Error: {e}")
