import os
import numpy as np
import pandas as pd
import json

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, 'cluster_run_config.json')):
    PROJECT_ROOT = current_dir
else:
    PROJECT_ROOT = os.path.dirname(current_dir)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CLUSTER_FILE = os.path.join(PROJECT_ROOT, 'station_to_cluster.json')
SPLIT_CONFIG_FILE = os.path.join(PROJECT_ROOT, 'cluster_run_config.json')

def load_cluster_config():
    if os.path.exists(SPLIT_CONFIG_FILE):
        with open(SPLIT_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return None

# 动态重新计算 R2
def calculate_r2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot == 0: 
        return 0.0
    return 1 - (ss_res / ss_tot)

def main():
    print("=== Cluster-Based Training Analysis ===\n")
    
    cluster_configs = load_cluster_config()
    if not cluster_configs:
        print("Warning: cluster_run_config.json not found. Information might be incomplete.")
        cluster_configs = []
        for i in range(3):
            cluster_configs.append({'cluster_id': i, 'stations_count': 1})

    results = []
    total_stations = 0
    # 我们现在同时追踪标准(Std)和真实(Real)两套核心指标
    weighted_metrics = {'std_mae': 0, 'std_mse': 0, 'real_mae': 0, 'real_mse': 0, 'real_r2': 0}
    
    for config in cluster_configs:
        c_id = config.get('cluster_id', config.get('cluster', 'Unknown'))
        n_stations = config.get('stations_count', config.get('stations', 1))
        
        target_folder = None
        if os.path.exists(RESULTS_DIR):
            for folder in os.listdir(RESULTS_DIR):
                if f"EV_Cluster_{c_id}" in folder:
                    target_folder = os.path.join(RESULTS_DIR, folder)
                    break
        
        if target_folder:
            metric_path = os.path.join(target_folder, 'metrics.npy')
            pred_orig_path = os.path.join(target_folder, 'pred_original.npy')
            true_orig_path = os.path.join(target_folder, 'true_original.npy')
            
            std_mae, std_mse = 0.0, 0.0
            real_mae, real_mse, real_r2 = 0.0, 0.0, 0.0
            
            # 1. 获取标准化指标 (即使文件旧，标准化指标仍是正确的参考)
            if os.path.exists(metric_path):
                metrics = np.load(metric_path)
                std_mae = metrics[0]
                std_mse = metrics[1]
            
            # 2. 核心修复：直接读取真实文件，动态重算业务指标，无视旧 metrics.npy！
            if os.path.exists(pred_orig_path) and os.path.exists(true_orig_path):
                preds_real = np.load(pred_orig_path)
                trues_real = np.load(true_orig_path)
                
                real_mae = np.mean(np.abs(preds_real - trues_real))
                real_mse = np.mean((preds_real - trues_real) ** 2)
                real_r2 = calculate_r2(preds_real, trues_real)
            else:
                # 兼容旧版本
                if os.path.exists(metric_path) and len(metrics) > 5:
                    real_r2 = metrics[5]

            results.append({
                'Cluster': f"Cluster {c_id}",
                'Stations': n_stations,
                'MSE (Std)': round(std_mse, 4),
                'MAE (Std)': round(std_mae, 4),
                'MSE (Real)': round(real_mse, 4),
                'R2 (Real)': round(real_r2, 4)
            })

            # 累加加权
            total_stations += n_stations
            weighted_metrics['std_mae'] += std_mae * n_stations
            weighted_metrics['std_mse'] += std_mse * n_stations
            weighted_metrics['real_mae'] += real_mae * n_stations
            weighted_metrics['real_mse'] += real_mse * n_stations
            weighted_metrics['real_r2'] += real_r2 * n_stations
        else:
            print(f"Cluster {c_id}: No model output folder found in {RESULTS_DIR}")

    print("\n=== Summary Table ===")
    df = pd.DataFrame(results)
    if not df.empty:
        print(df.to_string(index=False))
        
        print("\n=== Weighted Average (Global Performance) ===")
        if total_stations > 0:
            for k in weighted_metrics:
                weighted_metrics[k] /= total_stations
            
            print(f"Total Stations: {total_stations}")
            print(f"Global MSE (Std)  : {weighted_metrics['std_mse']:.4f}")
            print(f"Global MAE (Std)  : {weighted_metrics['std_mae']:.4f}")
            print(f"Global MSE (Real) : {weighted_metrics['real_mse']:.4f}")
            print(f"Global MAE (Real) : {weighted_metrics['real_mae']:.4f}")
            print(f"Global R2  (Real) : {weighted_metrics['real_r2']:.4f}")
        else:
            print("No valid station weights found.")
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()