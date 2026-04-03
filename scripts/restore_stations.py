
import json
import os

CLUSTER_FILE = "station_to_cluster.json"
# Stations to restore to Cluster 2
RESTORE_STATIONS = [
    'Station_160',
    'Station_125',
    'Station_126',
    'Station_112',
    'Station_129'
]
TARGET_CLUSTER = 2

def main():
    if not os.path.exists(CLUSTER_FILE):
        print(f"Error: {CLUSTER_FILE} not found.")
        return

    with open(CLUSTER_FILE, 'r') as f:
        mapping = json.load(f)
        
    initial_count = len(mapping)
    print(f"Initial Station Count: {initial_count}")
    
    restored_count = 0
    for station in RESTORE_STATIONS:
        if station not in mapping:
            mapping[station] = TARGET_CLUSTER
            restored_count += 1
            print(f"  Restored {station} -> Cluster {TARGET_CLUSTER}")
        else:
            print(f"  {station} already exists.")
            
    with open(CLUSTER_FILE, 'w') as f:
        json.dump(mapping, f, indent=4)
        
    print(f"Restored {restored_count} stations.")
    print(f"Final Station Count: {len(mapping)}")

if __name__ == "__main__":
    main()
