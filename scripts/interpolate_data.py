
import pandas as pd
import numpy as np
import os

DATA_PATH = r"dataset/EV_Data/EV_Load_Cleaned.csv"
OUTPUT_PATH = r"dataset/EV_Data/EV_Load_Cleaned.csv" # Overwrite

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Exclude 'date' column from interpolation
    if 'date' in df.columns:
        date_col = df['date']
        df_numeric = df.drop(columns=['date'])
    else:
        date_col = None
        df_numeric = df

    print("Interpolating outliers (Zeros)...")
    
    # 1. Replace 0 with NaN
    # We assume strict 0.0 is an outage/error for these specific EV loads in this context
    # (Based on user interaction: 0s were causing massive MSE)
    total_zeros = (df_numeric == 0).sum().sum()
    print(f"  Found {total_zeros} zero values in the dataset.")
    
    df_numeric = df_numeric.replace(0, np.nan)
    
    # 2. Interpolate
    # Limit direction='both' to fill start/end if needed, but mainly 'forward' then 'backward'
    # method='linear' is standard for time series
    df_interp = df_numeric.interpolate(method='linear', limit_direction='both')
    
    # Check if any NaNs remain (e.g. if a whole column is 0)
    remaining_nans = df_interp.isna().sum().sum()
    if remaining_nans > 0:
        print(f"  Warning: {remaining_nans} NaNs remain. Filling with 0 as fallback.")
        df_interp = df_interp.fillna(0)
        
    # Stats
    print("  Interpolation complete.")
    
    # Reassemble
    if date_col is not None:
        df_interp.insert(0, 'date', date_col)
        
    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    df_interp.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
