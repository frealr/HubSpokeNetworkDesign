
import pandas as pd

f = 'CO261_Flujos_Viajeros_entre_Aerops_Espanyoles__Anyo.xlsx'
try:
    print(f"\n--- {f} ---")
    df = pd.read_excel(f)
    print(df.head())
    print("Columns:", df.columns.tolist())
except Exception as e:
    print(e)
