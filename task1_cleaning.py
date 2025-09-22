

import pandas as pd
import numpy as np
import csv
import os

def load_and_clean(path: str, output_path: str):
    # --- Step 1. Load file with delimiter detection ---
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(8192)

    delim = None
    try:
        dialect = csv.Sniffer().sniff(sample)
        delim = dialect.delimiter
        if delim in ['\r', '\n', ' '] or delim is None:
            delim = None
    except Exception:
        delim = None

    df = None
    if delim:
        try:
            df = pd.read_csv(path, sep=delim, engine='c', low_memory=True)
        except Exception:
            df = None

    if df is None or df.shape[1] == 1:
        for sep in ['\t', ';', '|', ',']:
            try:
                df_try = pd.read_csv(path, sep=sep, engine='c', low_memory=True)
                if df_try.shape[1] > 1:
                    df = df_try
                    break
            except Exception:
                pass

    if df is None:
        df = pd.read_csv(path, sep=None, engine='python')

    # --- Step 2. Clean column names ---
    df.columns = df.columns.astype(str).str.strip().str.lower()
    df.columns = df.columns.str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)

    # --- Step 3. Trim whitespace & normalize NA ---
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({'': np.nan, 'nan': np.nan, 'na': np.nan,
                               'n/a': np.nan, 'none': np.nan, 'null': np.nan})

    # --- Step 4. Remove duplicates ---
    df = df.drop_duplicates()

    # --- Step 5. Drop columns with >50% missing ---
    miss_frac = df.isnull().mean()
    df = df.drop(columns=miss_frac[miss_frac > 0.5].index.tolist())

    # --- Step 6. Convert dates ---
    for c in df.columns:
        if any(k in c for k in ['date', 'dob', 'birth', 'joined', 'join', 'timestamp']):
            df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=True)

    # --- Step 7. Standardize gender/country ---
    for c in [col for col in df.columns if 'gender' in col or 'sex' in col]:
        mapping = {
            'm': 'Male', 'male': 'Male', 'man': 'Male',
            'f': 'Female', 'female': 'Female', 'woman': 'Female',
            'other': 'Other', 'non-binary': 'Other', 'nb': 'Other'
        }
        df[c] = df[c].astype(str).str.lower().str.strip().map(mapping).fillna(df[c])

    country_map = {
        'us': 'United States', 'usa':'United States',
        'united states':'United States', 'united states of america':'United States',
        'uk':'United Kingdom', 'gb':'United Kingdom', 'great britain':'United Kingdom',
        'india':'India', 'in':'India'
    }
    for c in [col for col in df.columns if 'country' in col or 'nation' in col]:
        df[c] = df[c].astype(str).str.strip().str.lower().replace(country_map).replace({'nan': np.nan})

    # --- Step 8. Numeric conversion ---
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    for c in df.select_dtypes(include='object').columns:
        sample = df[c].dropna().astype(str).head(50)
        if not sample.empty:
            num_like = sample.str.replace(',','').str.replace('.','',1).str.isnumeric().mean()
            if num_like > 0.6:
                df[c] = pd.to_numeric(df[c].str.replace(',',''), errors='coerce')
                numeric_cols.append(c)
    numeric_cols = sorted(set(numeric_cols))

    # --- Step 9. Imputation ---
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median(skipna=True))

    for c in df.select_dtypes(include='object').columns:
        if df[c].isnull().all():
            df[c] = df[c].fillna('Unknown')
        else:
            try:
                mode = df[c].mode(dropna=True)[0]
                df[c] = df[c].fillna(mode)
            except Exception:
                df[c] = df[c].fillna('Unknown')

    # --- Step 10. Outlier clipping ---
    for c in numeric_cols:
        low, high = df[c].quantile(0.01), df[c].quantile(0.99)
        if pd.notna(low) and pd.notna(high) and low != high:
            df[c] = df[c].clip(lower=low, upper=high)

    # --- Save cleaned dataset ---
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to {output_path}")
    print(f"Final shape: {df.shape}, missing values: {df.isnull().sum().sum()}, duplicates: {df.duplicated().sum()}")
    return df

if __name__ == "__main__":
    input_file = "marketing_campaign.csv"
    output_file = "marketing_campaign_cleaned.csv"
    if os.path.exists(input_file):
        load_and_clean(input_file, output_file)
    else:
        print(f"File {input_file} not found. Place it in the same folder as this script.")
