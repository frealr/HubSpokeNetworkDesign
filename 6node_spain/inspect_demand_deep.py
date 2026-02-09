
import pandas as pd

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

try:
    df = pd.read_excel('CO261_Flujos_Viajeros_entre_Aerops_Espanyoles__Anyo.xlsx', header=None)
    print(df.iloc[:20, :])
    
except Exception as e:
    print(f"Error: {e}")
