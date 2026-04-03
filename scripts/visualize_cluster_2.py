
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'EV_Cluster_1')
DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'EV_Data', 'EV_Load_Cluster_1.csv')
FIGURE_DIR = os.path.join(PROJECT_ROOT, 'figures', 'Cluster_1_Analysis')

if not os.path.exists(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)

def main():
    print("Loading Results...")
    # Load Data
    pred = np.load(os.path.join(RESULTS_DIR, 'pred_original.npy'))
    true = np.load(os.path.join(RESULTS_DIR, 'true_original.npy'))
    
    # Load Columns to map indices to names
    # Note: Model skips 'date', so index 0 = temp
    df = pd.read_csv(DATA_PATH, nrows=1)
    # df columns: date, temp, ...
    # model columns: temp, ...
    cols = df.columns.tolist()
    if 'date' in cols:
        cols.remove('date') # Remove date to align with model output (enc_in=48)
    
    print(f"Shape: Pred={pred.shape}, True={true.shape}")
    print(f"Features: {len(cols)}")
    
    # Calculate MSE per feature (averaging over batch and time)
    # Shape: (B, T, N) -> (N,)
    diff = pred - true
    mse_per_feature = np.mean(diff ** 2, axis=(0, 1))
    mae_per_feature = np.mean(np.abs(diff), axis=(0, 1))
    
    # Separate Stations from Covariates
    # Covariates are usually first 7
    covariate_count = 7
    station_cols = cols[covariate_count:]
    station_indices = list(range(covariate_count, len(cols)))
    
    station_metrics = []
    for i, idx in enumerate(station_indices):
        name = station_cols[i]
        station_metrics.append({
            'name': name,
            'index': idx,
            'mse': mse_per_feature[idx],
            'mae': mae_per_feature[idx]
        })
        
    metric_df = pd.DataFrame(station_metrics)
    metric_df = metric_df.sort_values(by='mse')
    
    print("\n=== Station Performance Ranking (Top 5) ===")
    print(metric_df.head(5).to_string(index=False))
    
    print("\n=== Station Performance Ranking (Bottom 5) ===")
    print(metric_df.tail(5).to_string(index=False))
    
    # Plotting Logic
    best_station = metric_df.iloc[0]
    worst_station = metric_df.iloc[-1]
    
    plot_station(pred, true, best_station, "Best_Performing_Station")
    plot_station(pred, true, worst_station, "Worst_Performing_Station")
    
    plt.close('all')
    print(f"\nFigures saved to {FIGURE_DIR}")

def plot_station(pred, true, station_info, title_prefix):
    idx = int(station_info['index'])
    name = station_info['name']
    mse = station_info['mse']
    
    # Flatten batches to show a continuous sequence (just for visualization feel)
    # However, since pred_len < seq_len in sliding window, simply flattening overlaps.
    # So we plot the first few distinct samples to avoid confusion.
    
    samples_to_plot = 4
    fig, axes = plt.subplots(samples_to_plot, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"{title_prefix}: {name} (MSE={mse:.4f})", fontsize=16)
    
    for b in range(min(samples_to_plot, pred.shape[0])):
        ax = axes[b]
        
        # Plot Prediction vs True (Last pred_len steps)
        p_seq = pred[b, :, idx]
        t_seq = true[b, :, idx]
        
        ax.plot(t_seq, label='Ground Truth', color='black', marker='.', linestyle='--')
        ax.plot(p_seq, label='Prediction', color='blue', linewidth=2)
        
        ax.set_title(f"Sample {b+1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.xlabel("Time Steps (Outlook)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(FIGURE_DIR, f"{title_prefix}_{name}.png"))
    print(f"Saved plot: {title_prefix}_{name}.png")

if __name__ == "__main__":
    main()
