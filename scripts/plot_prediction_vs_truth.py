import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import os

# Configuration Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
TRUE_FILE = os.path.join(RESULTS_DIR, 'All_Stations_GroundTruth.csv')
PRED_FILE = os.path.join(RESULTS_DIR, 'All_Stations_Predictions.csv')
OUTPUT_IMG = os.path.join(RESULTS_DIR, 'All_Stations_Spaghetti_Plot.png')
OUTPUT_IMG_PDF = os.path.join(RESULTS_DIR, 'All_Stations_Spaghetti_Plot.pdf')

def create_scientific_plot():
    print("Loading data...")
    df_true = pd.read_csv(TRUE_FILE)
    df_pred = pd.read_csv(PRED_FILE)

    # Ensure 'date' column is datetime
    df_true['date'] = pd.to_datetime(df_true['date'])
    df_pred['date'] = pd.to_datetime(df_pred['date'])

    # Set date as index
    df_true.set_index('date', inplace=True)
    df_pred.set_index('date', inplace=True)

    stations = df_true.columns.tolist()

    print(f"Plotting all {len(stations)} stations overlaid...")

    # Scientific plotting style setup
    plt.style.use('default')
    sns.set_theme(style="ticks", rc={
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    })

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    color_true = '#1f77b4'  # Scientific Blue
    color_pred = '#d62728'  # Brick Red
    
    # We will plot all 80 lines with very low opacity to form a "trajectory envelope"
    # This prevents the plot from looking like an unreadable mess of 160 lines
    for i, station in enumerate(stations):
        # Only add label for the first one to avoid duplicate legends
        label_true = 'Ground Truth (Individual Stations)' if i == 0 else None
        label_pred = 'Time-LLM Prediction (Individual Stations)' if i == 0 else None
        
        ax.plot(df_true.index, df_true[station], color=color_true, alpha=0.15, linewidth=1.0, label=label_true)
        ax.plot(df_pred.index, df_pred[station], color=color_pred, alpha=0.15, linewidth=1.0, linestyle='-', label=label_pred)

    # Plot the mean line to show the overall trend
    mean_true = df_true.mean(axis=1)
    mean_pred = df_pred.mean(axis=1)
    
    # Thicker solid lines for the mean
    ax.plot(mean_true.index, mean_true, color='#08306b', alpha=1.0, linewidth=3.0, label='Ground Truth (80 Stations Mean)')
    ax.plot(mean_pred.index, mean_pred, color='#7f0000', alpha=1.0, linewidth=3.0, linestyle='--', marker='o', markersize=5, label='Prediction (80 Stations Mean)')

    ax.set_title('Global EV Load Forecast Overview (80 Stations Overlaid Trajectories)', loc='left', fontweight='bold')
    ax.set_ylabel('Load (kW)')
    ax.set_xlabel('Time')
    
    # Customizing the legend to only show the unique representative lines
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', frameon=True, fancybox=False, edgecolor='black')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=15)
    sns.despine(ax=ax)

    plt.tight_layout()

    print("Saving plots...")
    plt.savefig(OUTPUT_IMG, bbox_inches='tight', dpi=300)
    plt.savefig(OUTPUT_IMG_PDF, bbox_inches='tight')
    
    print(f"High-quality scientific plots saved to:\n- {OUTPUT_IMG}\n- {OUTPUT_IMG_PDF}")

if __name__ == "__main__":
    create_scientific_plot()
