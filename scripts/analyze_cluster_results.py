
import os
import numpy as np
import pandas as pd
import json

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CLUSTER_FILE = os.path.join(PROJECT_ROOT, 'station_to_cluster.json')
SPLIT_CONFIG_FILE = os.path.join(PROJECT_ROOT, 'cluster_run_config.json')

def load_cluster_config():
    if os.path.exists(SPLIT_CONFIG_FILE):
        with open(SPLIT_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return None

def main():
    print("=== Cluster-Based Training Analysis ===\n")
    
    cluster_configs = load_cluster_config()
    if not cluster_configs:
        print("Warning: cluster_run_config.json not found. Information might be incomplete.")
        cluster_configs = []
        # Fallback: assume 0, 1, 2
        for i in range(3):
            cluster_configs.append({'cluster_id': i, 'stations_count': 1}) # Default weight

    results = []
    total_stations = 0
    weighted_metrics = {'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'mspe': 0}
    
    # Iterate through clusters
    for config in cluster_configs:
        c_id = config['cluster_id']
        n_stations = config.get('stations_count', 1)
        model_id = f"EV_Cluster_{c_id}"
        folder = os.path.join(RESULTS_DIR, model_id)
        metric_path = os.path.join(folder, 'metrics.npy')
        
        if os.path.exists(metric_path):
            # [mae, mse, rmse, mape, mspe]
            metrics = np.load(metric_path)
            mae, mse, rmse, mape, mspe = metrics
            
            # Print Cluster Result
            print(f"Cluster {c_id}:")
            print(f"  Stations: {n_stations}")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAPE: {mape:.4f}")
            print("-" * 30)
            
            results.append({
                'Cluster': c_id,
                'Stations': n_stations,
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'MSPE': mspe
            })
            
            # Accumulate for weighted average (using station count as weight)
            # Or should we use sample count? Usually station count is a good proxy for "global importance"
            # if we consider the task is to predict all stations.
            total_stations += n_stations
            weighted_metrics['mae'] += mae * n_stations
            weighted_metrics['mse'] += mse * n_stations
            weighted_metrics['rmse'] += rmse * n_stations
            weighted_metrics['mape'] += mape * n_stations
            weighted_metrics['mspe'] += mspe * n_stations
            
        else:
            print(f"Cluster {c_id}: No results found at {metric_path}")

    print("\n=== Summary Table ===")
    df = pd.DataFrame(results)
    if not df.empty:
        print(df.to_string(index=False))
        
        print("\n=== Weighted Average (Global Performance) ===")
        if total_stations > 0:
            for k in weighted_metrics:
                weighted_metrics[k] /= total_stations
            
            print(f"Total Stations: {total_stations}")
            print(f"Global MSE:  {weighted_metrics['mse']:.4f}")
            print(f"Global MAE:  {weighted_metrics['mae']:.4f}")
            print(f"Global RMSE: {weighted_metrics['rmse']:.4f}")
            print(f"Global MAPE: {weighted_metrics['mape']:.4f}")
        else:
            print("Weighted metrics could not be calculated.")
    else:
        print("No results available.")

if __name__ == "__main__":
    main()
