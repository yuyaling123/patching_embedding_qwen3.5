
import json
import pandas as pd
import os
import argparse

# Config
DEFAULT_DATA_PATH = './dataset/EV_Data/EV_Load_Cleaned.csv'
DEFAULT_CLUSTER_PATH = './station_to_cluster.json'
OUTPUT_DIR = './dataset/EV_Data/'

def split_data_by_cluster(data_path, cluster_path):
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    if not os.path.exists(cluster_path):
        print(f"Error: Cluster mapping file not found at {cluster_path}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # [OPTIMIZATION] Keep only last 20 days (assuming hourly data: 20 * 24 = 480)
    # Original length was ~720 (30 days). Reducing to avoid OOM/Freezing.
    # User confirmed 60 days of data, so we remove the truncation to train properly.
    
    print(f"Loading cluster mapping from {cluster_path}...")
    with open(cluster_path, 'r', encoding='utf-8') as f:
        station_to_cluster = json.load(f)
    
    # Invert mapping to cluster -> list of stations
    # If station_to_cluster is {"station_name": cluster_id, ...}
    clusters = {}
    for station, cluster_id in station_to_cluster.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(station)
        
    print(f"Found {len(clusters)} clusters.")
    
    # Identify Covariates (First 8 columns usually: date + 7 covariates)
    # We verify this roughly
    base_cols = df.columns[:8].tolist() # date, temp, humidity, precip, windspeed, uvindex, avg_e_price, avg_s_price
    print(f"Base columns (Preserved): {base_cols}")
    
    generated_files = []
    
    for cluster_id, stations in clusters.items():
        print(f"\nProcessing Cluster {cluster_id} ({len(stations)} stations)...")
        
        # Filter valid stations (that exist in the dataframe)
        valid_stations = [s for s in stations if s in df.columns]
        missing_stations = [s for s in stations if s not in df.columns]
        
        if missing_stations:
            print(f"  Warning: {len(missing_stations)} stations from cluster file not found in CSV. (Skipping them)")
            
        if not valid_stations:
            print(f"  Error: No valid stations found for Cluster {cluster_id}. Skipping.")
            continue
            
        # Select columns
        cols_to_keep = base_cols + valid_stations
        df_cluster = df[cols_to_keep]
        
        # Save
        filename = f"EV_Load_Cluster_{cluster_id}.csv"
        output_path = os.path.join(OUTPUT_DIR, filename)
        df_cluster.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")
        print(f"  Shape: {df_cluster.shape}")
        
        generated_files.append({
            'cluster_id': cluster_id,
            'file': filename,
            'stations_count': len(valid_stations),
            'enc_in': 7 + len(valid_stations) # 7 covariates + N stations
        })
        
    print("\nSummary:")
    for info in generated_files:
        print(f"Cluster {info['cluster_id']}: File={info['file']}, Stations={info['stations_count']}, enc_in={info['enc_in']}")
        
    # Optional: Save a config for the runner script to read
    # import json  <-- Removed this line to fix UnboundLocalError
    with open('cluster_run_config.json', 'w') as f:
        json.dump(generated_files, f, indent=4)
    print("Saved run config to cluster_run_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument('--cluster_path', type=str, default=DEFAULT_CLUSTER_PATH)
    args = parser.parse_args()
    
    split_data_by_cluster(args.data_path, args.cluster_path)
