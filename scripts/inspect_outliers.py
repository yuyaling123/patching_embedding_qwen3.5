
import pandas as pd
import os

DATA_PATH = r"dataset/EV_Data/EV_Load_Cluster_2.csv"

def main():
    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
        return

    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Stations identified from previous step
    stations_of_interest = ['Station_98', 'Station_178', 'Station_160', 'Station_125']
    
    print(f"\n{'Station':<15} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Zeros (%)':<10}")
    print("-" * 80)
    
    for station in stations_of_interest:
        if station not in df.columns:
            print(f"{station:<15} | Not Found")
            continue
            
        series = df[station]
        mean_val = series.mean()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()
        zero_pct = (series == 0).mean() * 100
        
        print(f"{station:<15} | {mean_val:<10.2f} | {std_val:<10.2f} | {min_val:<10.2f} | {max_val:<10.2f} | {zero_pct:<10.1f}%")

if __name__ == "__main__":
    main()
