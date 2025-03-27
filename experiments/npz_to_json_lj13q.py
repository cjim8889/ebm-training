import numpy as np
import json
import os
from pathlib import Path

def npz_to_json(npz_file, json_file):
    """
    Convert NPZ trajectory data to JSON format.
    
    Args:
        npz_file (str): Path to the NPZ file.
        json_file (str): Path to save the JSON file.
    """
    print(f"Loading data from {npz_file}...")
    # Load the NPZ file
    data = np.load(npz_file)
    
    # Extract the keys and data
    positions = data['positions'][:, :256, :].tolist() # Shape: (128, 256, 39)
    times = data['times'].tolist() if 'times' in data else list(range(data['positions'].shape[0]))
    
    # Create a dictionary to store the data
    json_data = {
        'positions': positions,
        'times': times,
        'metadata': {
            'n_steps': len(positions),
            'n_samples': len(positions[0]) if positions else 0,
            'dimensions': len(positions[0][0]) if positions and positions[0] else 0,
            'original_file': os.path.basename(npz_file)
        }
    }
    
    # Save to JSON file
    print(f"Converting data to JSON and saving to {json_file}...")
    with open(json_file, 'w') as f:
        json.dump(json_data, f)
    
    file_size_mb = Path(json_file).stat().st_size / (1024 * 1024)
    print(f"Conversion complete! JSON file size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    npz_file = "data/lj13q_samples_128_steps_trajectory.npz"
    json_file = "data/lj13q_trajectory.json"
    
    npz_to_json(npz_file, json_file)