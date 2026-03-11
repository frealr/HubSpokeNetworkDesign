import re

with open('generate_data.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace parse_matrix
old_parse = """def parse_matrix(output_csv, name, n):
    \"\"\"Lee una matriz (n x n) desde el CSV de outputs de GAMS.\"\"\"
    m_df = read_gams_csv_robust(output_csv, symbol_name=name)
    if len(m_df) == 0:
        return np.zeros((n, n))
    m_df = m_df.copy()
    m_df.columns = ['i', 'j', 'value']
    try:
        m_df['i_idx'] = m_df['i'].str.extract(r'(\d+)').astype(int) - 1
        m_df['j_idx'] = m_df['j'].str.extract(r'(\d+)').astype(int) - 1
        m = np.zeros((n, n))
        m[m_df['i_idx'].values, m_df['j_idx'].values] = m_df['value'].values
        return m
    except Exception:
        return np.zeros((n, n))"""

new_parse = """def parse_matrix(output_xlsx, name, n):
    \"\"\"Lee una matriz (n x n) desde el XLSX de outputs de GAMS.\"\"\"
    m_df = read_excel_robust(output_xlsx, sheet_name=name)
    if len(m_df) == 0:
        return np.zeros((n, n))
    m = np.zeros((n, n))
    try:
        if m_df.shape[1] > n:
            val_matrix = m_df.iloc[:, 1:].values
        else:
            val_matrix = m_df.values
        r = min(n, val_matrix.shape[0])
        c = min(n, val_matrix.shape[1])
        m[:r, :c] = val_matrix[:r, :c]
        return m
    except Exception as e:
        print(f"Error parsing {name}: {e}")
        return np.zeros((n, n))"""

code = code.replace(old_parse, new_parse)

# cvx_blo read 1
old_read1 = """                if os.path.exists('./output_all.csv'):
                    ctime_vals = read_gams_csv_robust('./output_all.csv', symbol_name='solver_time').values.flatten()
                    if len(ctime_vals) > 0: comp_time += ctime_vals[-1] # Usually last column is value
                    
                    sh_df = read_gams_csv_robust('./output_all.csv', symbol_name='sh_level')
                    sh = np.maximum(sh_df.iloc[:, -1].values.flatten() if len(sh_df)>0 else np.zeros(n), 1e-4)
                    
                    s_df = read_gams_csv_robust('./output_all.csv', symbol_name='s_level')
                    s = np.maximum(s_df.iloc[:, -1].values.flatten() if len(s_df)>0 else np.zeros(n), 1e-4)
                    
                    f = parse_matrix('./output_all.csv', 'f_level', n)
                    a = np.maximum(parse_matrix('./output_all.csv', 'a_level', n), 1e-4)
                    fext = parse_matrix('./output_all.csv', 'fext_level', n)
                else:"""

new_read1 = """                if os.path.exists('./output_all.xlsx'):
                    ctime_vals = read_excel_robust('./output_all.xlsx', sheet_name='solver_time').values.flatten()
                    if len(ctime_vals) > 0: comp_time += ctime_vals[-1]
                    
                    sh_df = read_excel_robust('./output_all.xlsx', sheet_name='sh_level')
                    sh = np.maximum(sh_df.values.flatten() if len(sh_df)>0 else np.zeros(n), 1e-4)
                    
                    s_df = read_excel_robust('./output_all.xlsx', sheet_name='s_level')
                    s = np.maximum(s_df.values.flatten() if len(s_df)>0 else np.zeros(n), 1e-4)
                    
                    f = parse_matrix('./output_all.xlsx', 'f_level', n)
                    a = np.maximum(parse_matrix('./output_all.xlsx', 'a_level', n), 1e-4)
                    fext = parse_matrix('./output_all.xlsx', 'fext_level', n)
                else:"""

code = code.replace(old_read1, new_read1)

# cvx_blo read 2
old_read2 = """                if os.path.exists('./output_all.csv'):
                    ctime_vals = read_gams_csv_robust('./output_all.csv', symbol_name='solver_time').values.flatten()
                    if len(ctime_vals) > 0: comp_time += ctime_vals[-1]
                    
                    sh_df = read_gams_csv_robust('./output_all.csv', symbol_name='sh_level')
                    sh = np.maximum(sh_df.iloc[:, -1].values.flatten() if len(sh_df)>0 else np.zeros(n), 1e-4)
                    
                    s_df = read_gams_csv_robust('./output_all.csv', symbol_name='s_level')
                    s = np.maximum(s_df.iloc[:, -1].values.flatten() if len(s_df)>0 else np.zeros(n), 1e-4)
                    
                    f = parse_matrix('./output_all.csv', 'f_level', n)
                    a = np.maximum(parse_matrix('./output_all.csv', 'a_level', n), 1e-4)
                    fext = parse_matrix('./output_all.csv', 'fext_level', n)"""

new_read2 = """                if os.path.exists('./output_all.xlsx'):
                    ctime_vals = read_excel_robust('./output_all.xlsx', sheet_name='solver_time').values.flatten()
                    if len(ctime_vals) > 0: comp_time += ctime_vals[-1]
                    
                    sh_df = read_excel_robust('./output_all.xlsx', sheet_name='sh_level')
                    sh = np.maximum(sh_df.values.flatten() if len(sh_df)>0 else np.zeros(n), 1e-4)
                    
                    s_df = read_excel_robust('./output_all.xlsx', sheet_name='s_level')
                    s = np.maximum(s_df.values.flatten() if len(s_df)>0 else np.zeros(n), 1e-4)
                    
                    f = parse_matrix('./output_all.xlsx', 'f_level', n)
                    a = np.maximum(parse_matrix('./output_all.xlsx', 'a_level', n), 1e-4)
                    fext = parse_matrix('./output_all.xlsx', 'fext_level', n)"""

code = code.replace(old_read2, new_read2)

# MIP read
old_mip = """    if os.path.exists('./output_all.csv'):
        sh_df = read_gams_csv_robust('./output_all.csv', symbol_name='sh_level')
        sh = sh_df.iloc[:, -1].values.flatten() if len(sh_df)>0 else np.zeros(n)
        
        s_df = read_gams_csv_robust('./output_all.csv', symbol_name='s_level')
        s = s_df.iloc[:, -1].values.flatten() if len(s_df)>0 else np.zeros(n)
        sprim = s.copy()
        deltas = np.zeros(n)
        a = parse_matrix('./output_all.csv', 'a_level', n)
        f = parse_matrix('./output_all.csv', 'f_level', n)
        fext = parse_matrix('./output_all.csv', 'fext_level', n)
        
        mipgap_df = read_gams_csv_robust('./output_all.csv', symbol_name='mip_opt_gap')
        mipgap = mipgap_df.iloc[:, -1].values.flatten() if len(mipgap_df)>0 else [0]
        
        ctime_vals = read_gams_csv_robust('./output_all.csv', symbol_name='solver_time').values.flatten()
        comp_time = ctime_vals[-1] if len(ctime_vals) > 0 else 0
    else:"""

new_mip = """    if os.path.exists('./output_all.xlsx'):
        sh_df = read_excel_robust('./output_all.xlsx', sheet_name='sh_level')
        sh = sh_df.values.flatten() if len(sh_df)>0 else np.zeros(n)
        
        s_df = read_excel_robust('./output_all.xlsx', sheet_name='s_level')
        s = s_df.values.flatten() if len(s_df)>0 else np.zeros(n)
        sprim = s.copy()
        deltas = np.zeros(n)
        a = parse_matrix('./output_all.xlsx', 'a_level', n)
        f = parse_matrix('./output_all.xlsx', 'f_level', n)
        fext = parse_matrix('./output_all.xlsx', 'fext_level', n)
        
        try:
            mipgap_df = read_excel_robust('./output_all.xlsx', sheet_name='mip_opt_gap')
            mipgap = mipgap_df.values.flatten() if len(mipgap_df)>0 else [0]
        except Exception:
            mipgap = [0]
        
        ctime_vals = read_excel_robust('./output_all.xlsx', sheet_name='solver_time').values.flatten()
        comp_time = ctime_vals[-1] if len(ctime_vals) > 0 else 0
    else:"""

code = code.replace(old_mip, new_mip)

with open('generate_data.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Done.")
