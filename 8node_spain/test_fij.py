import pandas as pd
import numpy as np
def split_and_accumarray(fij_df, iU, jU, oU, dU):
    fij = np.zeros((len(iU), len(jU), len(oU), len(dU)))
    print(f"Shapes: {fij.shape}")
    print(iU)
    print(jU)
    print(oU)
    print(dU)
    return fij

T = pd.read_csv('fij_long.csv')
iU = T['i'].unique(); jU = T['j'].unique(); oU = T['o'].unique(); dU = T['d'].unique()
fij = split_and_accumarray(T, iU, jU, oU, dU)
