import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

CONFIG = {
    'INPUT_FILE': 'Data/investigator_nacc71.csv',
    'OUTPUT_CSV': 'clinical_data.csv',

    'KEEP_COLS': [
        'NACCID', 'NACCVNUM',
        'NACCAGE', 'SEX', 'EDUC', 'RACE', 'HISPANIC', 'MARISTAT', 
        'LIVSIT', 'RESIDENC', 'HANDED',
        'NACCAPOE', 'NACCNE4S', 'NACCFAM', 'NACCMOM', 'NACCDAD',
        'HEIGHT', 'WEIGHT', 'NACCBMI', 'BPSYS', 'BPDIAS', 'HRATE', 
        'VISION', 'HEARING',
        'CVHATT', 'CVAFIB', 'CVANGIO', 'CVBYPASS', 'CVPACE', 
        'CVCHF', 'CVOTHR', 'HYPERTEN', 
        'CBSTROKE', 'CBTIA', 'SEIZURES', 'TBI', 'PARKIN', 'PD',
        'DIABETES', 'HYPERCHO', 'B12DEF', 'THYROID', 
        'INCONTU', 'INCONTF', 'APNEA', 'INSOMN',
        'ALCOHOL', 'TOBAC30', 'TOBAC100', 'PACKSPER', 
        'DEP2YRS', 'ANXIETY', 'ANYMEDS'
    ]
}

def main():
    print("GENERATING PURE CLINICAL DATASET")
    print("="*60)
    
    if not os.path.exists(CONFIG['INPUT_FILE']):
        print(f"File not found: {CONFIG['INPUT_FILE']}")
        return
    
    df = pd.read_csv(CONFIG['INPUT_FILE'], low_memory=False)
    
    if 'NACCID' in df.columns:
        df['NACCID'] = df['NACCID'].astype(str).str.strip()
    if 'NACCVNUM' in df.columns:
        df['NACCVNUM'] = pd.to_numeric(df['NACCVNUM'], errors='coerce')
    
    if 'CDRGLOB' not in df.columns:
        print("CDRGLOB column missing!")
        return

    df['target'] = df['CDRGLOB'].apply(
        lambda x: 0 if x==0 else (1 if x==0.5 else (2 if x==1 else (3 if x>=2 else -1)))
    )
    df = df[df['target'] != -1]
    
    valid_cols = [c for c in CONFIG['KEEP_COLS'] if c in df.columns]
    
    if 'target' not in valid_cols:
        valid_cols.append('target')
    
    df_clean = df[valid_cols].copy()
    
    df_clean = df_clean.sort_values(['NACCID', 'NACCVNUM'])
    
    df_clean['future_target'] = df_clean.groupby('NACCID')['target'].shift(-1)
    
    df_final = df_clean.dropna(subset=['future_target'])
    df_final['future_target'] = df_final['future_target'].astype(int)
    
    print("-" * 60)
    print(f"Dataset saved to: {CONFIG['OUTPUT_CSV']}")
    print(f"Training Pairs: {len(df_final)}")
    print(f"Features:       {len(df_final.columns)}")
    print(f"Unique Patients:{df_final['NACCID'].nunique()}")

    df_final.to_csv(CONFIG['OUTPUT_CSV'], index=False)

if __name__ == "__main__":
    main()