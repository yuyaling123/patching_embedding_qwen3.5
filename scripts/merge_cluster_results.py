
import os
import numpy as np
import pandas as pd
import json

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'EV_Data')
OUTPUT_FILE_PRED = os.path.join(RESULTS_DIR, 'All_Stations_Predictions.csv')
OUTPUT_FILE_TRUE = os.path.join(RESULTS_DIR, 'All_Stations_GroundTruth.csv')

def main():
    print("=== Merging Cluster Results ===")
    
    clusters = [0, 1, 2] # Assuming these are the IDs
    
    all_preds_df = pd.DataFrame()
    all_trues_df = pd.DataFrame()
    
    # We need to handle the time index. 
    # Since all clusters use the same time range and split, we can just grab the time index from the first valid cluster we process.
    # However, the model output (B, P, N) is sliding window. Reshaping it to a flat timeline requires careful alignment if strides are involved.
    # But usually 'pred_original.npy' is (Samples, Pred_Len, Features).
    # If we want a simple "Comparison" per sample, we can save it as is.
    # But user likely wants a CSV where rows = timestamps (or sample_ids) and columns = stations.
    # Given 'pred_original.npy' is usually batch outputs concatenated.
    # Let's flatten the first sample's time dimension or just concatenate features if shape matches.
    
    # Actually, Time-LLM's `pred.npy` is shape (N_Samples, Pred_Len, N_Features).
    # Flattening this to a single continuous time series is complex because of overlaps.
    # Usually users want to see the specific forecast for a specific horizon.
    # But to "merge results" into a CSV for analysis usually implies:
    # 1. Column-wise merge: [Pred_Station_A, Pred_Station_B, ...]
    # 2. Row-wise? The rows are samples.
    
    # Let's assume we want to preserve the (N_Samples, Pred_Len) structure but merge features.
    # Or, if we simply want to align "Station 1 Prediction" vs "Station 2 Prediction" for same test samples.
    
    # Strategy:
    # 1. Read Cluster 0. Get shape (S, P, N0). 
    # 2. Read Cluster 1. Verify shape (S, P, N1).
    # 3. Concatenate N dimension -> (S, P, N_Total).
    # 4. Map N indices to Station Names.
    # 5. Flatten to 2D CSV: Rows = Sample_Step, Columns = Station Names.
    
    merged_preds = []
    merged_trues = []
    station_names = []
    
    sample_shape = None
    
    for c_id in clusters:
        print(f"Processing Cluster {c_id}...")
        
        # Paths
        res_path = os.path.join(RESULTS_DIR, f"EV_Cluster_{c_id}")
        pred_path = os.path.join(res_path, 'pred_original.npy')
        true_path = os.path.join(res_path, 'true_original.npy')
        input_csv = os.path.join(DATA_DIR, f"EV_Load_Cluster_{c_id}.csv")
        
        if not os.path.exists(pred_path):
            print(f"  Warning: Results for Cluster {c_id} not found. Skipping.")
            continue
            
        # Load Arrays
        # Shape: (Samples, Pred_Len, Features)
        pred = np.load(pred_path)
        true = np.load(true_path)
        
        if sample_shape is None:
            sample_shape = (pred.shape[0], pred.shape[1])
        else:
            if pred.shape[0] != sample_shape[0] or pred.shape[1] != sample_shape[1]:
                print(f"  Error: Shape mismatch for Cluster {c_id}. Expected {sample_shape}, got {(pred.shape[0], pred.shape[1])}")
                continue
                
        # Load Column Names to identify Stations
        # The model input has Covariates FIRST, then Stations.
        # usually 7 covariates. run_main.py ignores 'date' column if present in csv but data_loader handling might vary.
        # Let's read CSV header.
        df = pd.read_csv(input_csv, nrows=0)
        cols = df.columns.tolist()
        if 'date' in cols:
            cols.remove('date') # Model usually skips date
            
        # The output 'pred' has only the Target Variables?
        # NO. Time-LLM 'M' mode outputs ALL features (enc_in = c_out).
        # But usually we only care about stations. Covariates are cols 0-6.
        # Stations start at index 7.
        
        covariate_count = 7
        cluster_stations = cols[covariate_count:]
        
        # Extract only station columns from pred/true array
        # Check if pred only contains stations or also covariates
        if pred.shape[2] == len(cluster_stations):
            # Model output only contains station predictions
            pred_stations = pred
            true_stations = true
        elif pred.shape[2] == len(cols):
            # Model output contains covariates AND stations
            pred_stations = pred[:, :, covariate_count:]
            true_stations = true[:, :, covariate_count:]
        else:
            print(f"  Warning: Unexpected feature count in pred ({pred.shape[2]}) vs total cols ({len(cols)}) or stations ({len(cluster_stations)})")
            pred_stations = pred[:, :, -len(cluster_stations):]
            true_stations = true[:, :, -len(cluster_stations):]
        
        print(f"  Loaded {len(cluster_stations)} stations.")
        
        merged_preds.append(pred_stations)
        merged_trues.append(true_stations)
        station_names.extend(cluster_stations)

    if not merged_preds:
        print("No results merged.")
        return

    # Concatenate along feature axis (axis 2)
    final_pred_array = np.concatenate(merged_preds, axis=2)
    final_true_array = np.concatenate(merged_trues, axis=2)
    
    print(f"Final Merged Shape: {final_pred_array.shape}")
    print(f"Total Stations: {len(station_names)}")
    
    # SCHEME A: NEXT STEP ANALYSIS
    # User requested to keep only the "first step" of each prediction.
    # Prediction shape is (Samples, Pred_Len, Features).
    # We want Prediction[:, 0, :] -> Shape (Samples, Features).
    
    print("\nApplying Scheme A: Extracting only the first prediction step (Next Step Forecast)...")
    
    # Slice first step
    # final_pred_array: (S, P, F)
    flat_pred = final_pred_array[:, 0, :]
    flat_true = final_true_array[:, 0, :]
    
    # Reconstruct timestamps for these specific points
    # Logic: For sample i, the first prediction is at index: border1 + i + seq_len
    # (i.e., immediately after the input sequence)
    
    S_new = flat_pred.shape[0]
    print(f"  New Sample Count: {S_new}")
    
    single_step_dates = []
    
    # Load raw dates again to be sure
    first_csv = os.path.join(DATA_DIR, f"EV_Load_Cluster_0.csv")
    df_raw = pd.read_csv(first_csv)
    raw_dates = pd.to_datetime(df_raw['date'])
    
    # Recalculate border info just in case interaction scope changed variable mapping
    # Parameters from run_all_clusters.py
    seq_len = 96
    
    n = len(df_raw)
    num_train = int(n * 0.70)
    num_vali = int(n * 0.20)
    num_test = n - num_train - num_vali
    border1 = n - num_test - seq_len # This is where Test data inputs start
    
    # The 'target' for the first step of sample i is at:
    # input_end = border1 + i + seq_len
    # target_index = input_end (since pred starts immediately after seq)
    
    for i in range(S_new):
        target_idx = border1 + i + seq_len
        if target_idx < len(raw_dates):
            single_step_dates.append(raw_dates.iloc[target_idx])
        else:
            # Fallback if index out of bounds (shouldn't happen with correct split)
            single_step_dates.append(pd.NaT)

    # Create DataFrames
    df_pred = pd.DataFrame(flat_pred, columns=station_names)
    df_true = pd.DataFrame(flat_true, columns=station_names)
    
    # Insert Date
    df_pred.insert(0, 'date', single_step_dates)
    df_true.insert(0, 'date', single_step_dates)
    
    # Save
    df_pred.to_csv(OUTPUT_FILE_PRED, index=False)
    df_true.to_csv(OUTPUT_FILE_TRUE, index=False)
    
    print(f"Saved CLEAN Predictions (Scheme A) to: {OUTPUT_FILE_PRED}")
    print(f"Saved CLEAN Ground Truth (Scheme A) to: {OUTPUT_FILE_TRUE}")
    print("Done.")

if __name__ == "__main__":
    main()
