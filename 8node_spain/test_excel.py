import pandas as pd
df = pd.read_excel('output_all.xlsx', sheet_name='a_level')
print("a_level shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(2))

df_s = pd.read_excel('output_all.xlsx', sheet_name='sh_level')
print("\nsh_level shape:", df_s.shape)
print("Columns:", df_s.columns.tolist())
print(df_s.head(2))
