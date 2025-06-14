import os
import numpy as np
import pandas as pd
import base64
import zlib
from tqdm import tqdm
from enum import Enum
import argparse


class ENV(Enum):
    STUDENTCLUSTER = 1
    LOCAL = 2

RUN_ON = ENV.LOCAL  # or RUNON.STUDENTCLUSTER, depending on your environment
data_dir = os.getcwd() if RUN_ON == ENV.LOCAL else '/work/scratch/timrieder/cil_monocular_depth'
current_dir = os.getcwd()
parser = argparse.ArgumentParser(description="Create prediction CSV.")
parser.add_argument('-o', '--output_folder', type=str, default='output', help='Output folder name')
args, _ = parser.parse_known_args()
output_folder = args.output_folder

# Path definitions
# data_root = os.getcwd()
predictions_dir = os.path.join(current_dir, output_folder, 'predictions')
test_list_file = os.path.join(data_dir, 'test_list.txt')
output_csv = os.path.join(current_dir, output_folder, 'predictions.csv')

def compress_depth_values(depth_values):
    # Convert depth values to bytes
    depth_bytes = ','.join(f"{x:.2f}" for x in depth_values).encode('utf-8')
    # Compress using zlib
    compressed = zlib.compress(depth_bytes, level=9)  # level 9 is maximum compression
    # Encode as base64 for safe CSV storage
    return base64.b64encode(compressed).decode('utf-8')

def process_depth_maps():
    # Read file list
    with open(test_list_file, 'r') as f:
        file_pairs = [line.strip().split() for line in f]
    
    # Initialize lists to store data
    ids = []
    depths_list = []
    
    # Process each depth map
    for rgb_path, depth_path in tqdm(file_pairs, desc="Processing depth maps"):
        # Get file ID (without extension)
        file_id = os.path.splitext(os.path.basename(depth_path))[0]
        
        # Load depth map
        depth = np.load(os.path.join(predictions_dir, depth_path))
        # Flatten the depth map and round to two decimal points
        flattened_depth = np.round(depth.flatten(), 2)
        
        # Compress the depth values
        compressed_depths = compress_depth_values(flattened_depth)
        ids.append(file_id)
        depths_list.append(compressed_depths)

    # Create DataFrame
    df = pd.DataFrame({
        'id': ids,
        'Depths': depths_list,
    })
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to: {output_csv}")
    print(f"Shape of the CSV: {df.shape}")
    

if __name__ == "__main__":
    process_depth_maps() 