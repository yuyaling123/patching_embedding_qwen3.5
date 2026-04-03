
import json
import os

CLUSTER_FILE = "station_to_cluster.json"
# Outliers identified from Cluster 2 analysis (Bottom 5 MSE)
BAD_STATIONS = [
    'Station_160', # MSE ~87, Zeros ~12.7%
    'Station_125', # MSE ~64, Zeros ~19.8%
    'Station_126', # MSE ~55
    'Station_112', # MSE ~54
    'Station_129'  # MSE ~48
]

def main():
    if not os.path.exists(CLUSTER_FILE):
        print(f"Error: {CLUSTER_FILE} not found.")
        return

    with open(CLUSTER_FILE, 'r') as f:
        mapping = json.load(f)
        
    initial_count = len(mapping)
    print(f"Initial Station Count: {initial_count}")
    
    removed_count = 0
    for station in BAD_STATIONS:
        if station in mapping:
            del mapping[station]
            removed_count += 1
            print(f"  Removed {station} (Cluster {2})") # We know they are cluster 2
            
    with open(CLUSTER_FILE, 'w') as f:
        json.dump(mapping, f, indent=4)
        
    print(f"Removed {removed_count} stations.")
    print(f"Final Station Count: {len(mapping)}")
    print("Done. Please re-run Run_Cluster_Training.bat to apply these changes.")

if __name__ == "__main__":
    main()
