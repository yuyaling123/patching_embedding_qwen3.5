
import pandas as pd
import numpy as np
import os

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'EV_Data', 'EV_Load_Cleaned_all.csv') # Source with 0s
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'EV_Data', 'EV_Load_Cleaned.csv') # Target for training

def smart_fill(df, station_cols):
    """
    Replaces 0.0 values with values from T-24h (Last Day).
    If T-24h is also 0 or missing, tries T-48h.
    Fallback: Linear Interpolation (for very rare cases where history is also empty).
    """
    df_filled = df.copy()
    
    # Ensure 'date' is datetime index for convenience (optional, depending on logic)
    # Here we just use array indexing since data is hourly sorted
    
    filled_count = 0
    total_zeros = 0
    
    print(f"Processing {len(station_cols)} stations...")
    
    for col in station_cols:
        series = df_filled[col].copy()
        
        # Identify zero indices
        # We assume strict 0.0 indicates outage/missing in this context
        zero_indices = series[series == 0.0].index
        total_zeros += len(zero_indices)
        
        for idx in zero_indices:
            # Try T-24h
            if idx >= 24 and series[idx - 24] > 0.001:
                series[idx] = series[idx - 24]
                filled_count += 1
            # Try T-48h
            elif idx >= 48 and series[idx - 48] > 0.001:
                series[idx] = series[idx - 48]
                filled_count += 1
            # Try T+24h (Look ahead - cheating? No, for gap filling it's valid)
            elif idx + 24 < len(series) and series[idx + 24] > 0.001:
                series[idx] = series[idx + 24]
                filled_count += 1
            else:
                # If history fails, leave as 0 for now, linear interpolate later
                pass
                
        df_filled[col] = series

    print(f"Smart Fill: Replaced {filled_count}/{total_zeros} zero values with historical data.")
    
    # Second Pass: Linear Interpolation for remaining zeros
    # Replace remaining 0s with NaN to allow interpolation
    print("Performing residual Linear Interpolation...")
    df_filled[station_cols] = df_filled[station_cols].replace(0.0, np.nan)
    df_filled[station_cols] = df_filled[station_cols].interpolate(method='linear', limit_direction='both')
    
    # Fill any remaining NaNs (edges) with 0 or mean
    df_filled[station_cols] = df_filled[station_cols].fillna(0.0)
    
    return df_filled

def main():
    print(f"Loading raw data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    
    # Ensure consistency with previous training set size (1 Month approx)
    # The user was using a specific slice. The EV_Load_Cleaned.csv had 720 rows + header.
    # We will take the first 720 rows to match the experiment setup.
    TRAIN_SIZE = 720
    print(f"Slicing first {TRAIN_SIZE} rows for training dataset...")
    df_slice = df.head(TRAIN_SIZE).copy()
    
    # Identify Station columns (excluding covariates)
    # Covariates: date, temp, humidity, precip, windspeed, uvindex, avg_e_price, avg_s_price (8 cols)
    station_cols = [c for c in df_slice.columns if c.startswith('Station_')]
    print(f"Found {len(station_cols)} station columns.")
    
    # Apply Smart Fill
    df_clean = smart_fill(df_slice, station_cols)
    
    # Save
    print(f"Saving cleaned data to {OUTPUT_PATH}...")
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
