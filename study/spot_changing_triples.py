#!/usr/bin/env python3
"""
Identify triples changing linking signs (vorticity flips) and visualize them.
Use N=200 dataset.
"""

import numpy as np
import pandas as pd
import sys
import argparse
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from compute_topology import compute_linking_matrix
from find_stable_core import compute_vorticity_tensor, find_diffs

def load_rods_raw(filepath, n_rods, frame_idx):
    """Load rods from raw endpoints.csv (lines are rods)."""
    # File format: One line per rod. 
    # Frame 0: Rod 0, Rod 1, ... Rod N-1
    # Frame 1: Rod 0, ...
    
    data = np.loadtxt(filepath, delimiter=',')
    # Shape: (total_lines, 6)
    
    total_lines = data.shape[0]
    num_frames = total_lines // n_rods
    
    # Extract the slice for the requested frame (or last frame if -1)
    if frame_idx == -1:
        frame_idx = num_frames - 1
        
    start_idx = frame_idx * n_rods
    end_idx = start_idx + n_rods
    
    if start_idx >= total_lines:
        raise ValueError(f"Frame {frame_idx} out of bounds (total frames: {num_frames})")
        
    rods = data[start_idx:end_idx, :]
    # rods shape: (N, 6)
    
    return rods

def load_rods_formatted(filepath, frame_idx):
    """Load rods from formatted csv."""
    df = pd.read_csv(filepath, comment='#')
    frame_data = df[df['frame'] == frame_idx]
    
    rods = []
    for _, row in frame_data.iterrows():
        rod = np.array([row['x1'], row['y1'], row['z1'], row['x2'], row['y2'], row['z2']])
        rods.append(rod)
    return np.array(rods)

def write_polyscope_script(output_file, rods0, rods_final, changing_triples, ar):
    """
    Write a python script that uses polyscope to visualize the changing triples.
    """
    
    # We will pick top 3 changed triples to visualize
    # or just all of them if few.
    
    # Collect all unique rod indices involved in changes
    involved_rods = set()
    for (i, j, k) in changing_triples:
        involved_rods.add(i)
        involved_rods.add(j)
        involved_rods.add(k)
        
    involved_rods_list = sorted(list(involved_rods))
    
    # Prepare data for the script
    # We need to serialize rods and the triples
    
    rods0_list = rods0.tolist()
    rods_final_list = rods_final.tolist()
    triples_list = list(changing_triples)
    
    radius = 1.0 / (2.0 * ar)
    
    script_content = f"""
import polyscope as ps
import numpy as np

# Data
rods0 = np.array({rods0_list})
rods_final = np.array({rods_final_list})
triples = {triples_list}
radius = {radius}

def visualize():
    ps.init()
    ps.set_up_dir("z_up")
    
    # Register all rods at t=0
    # Structure: [N, 6] -> need edges and nodes
    
    def register_rods(name_prefix, rod_data, color=(0.5, 0.5, 0.5)):
        nodes = []
        edges = []
        for i, rod in enumerate(rod_data):
            p1 = rod[:3]
            p2 = rod[3:]
            base_idx = len(nodes)
            nodes.append(p1)
            nodes.append(p2)
            edges.append([base_idx, base_idx+1])
            
        nodes = np.array(nodes)
        edges = np.array(edges)
        
        ps_net = ps.register_curve_network(name_prefix, nodes, edges)
        ps_net.set_radius(radius, relative=False)
        ps_net.set_color(color)
        return ps_net

    # Register full state at t=0 (ghost)
    register_rods("all_rods_t0", rods0, color=(0.8, 0.8, 0.8)).set_transparency(0.2)
    register_rods("all_rods_final", rods_final, color=(0.8, 0.8, 0.8)).set_transparency(0.2)
    
    # Visualize specific triples
    # Only show the first 5 unique triples to avoid clutter, or handle user input
    
    unique_triples = triples[:5]
    print(f"Visualizing {{len(unique_triples)}} triples out of {{len(triples)}} total changes.")
    
    for idx, (i, j, k) in enumerate(unique_triples):
        suffix = f"_triple{{idx}}_{{i}}_{{j}}_{{k}}"
        
        # t=0
        nodes0 = np.array([
            rods0[i][:3], rods0[i][3:],
            rods0[j][:3], rods0[j][3:],
            rods0[k][:3], rods0[k][3:]
        ])
        edges = np.array([[0, 1], [2, 3], [4, 5]])
        
        net0 = ps.register_curve_network(f"t0{{suffix}}", nodes0, edges)
        net0.set_radius(radius * 1.5, relative=False) # thicker
        net0.set_color((0.2, 0.6, 1.0)) # Blueish
        
        # t=final
        nodesF = np.array([
            rods_final[i][:3], rods_final[i][3:],
            rods_final[j][:3], rods_final[j][3:],
            rods_final[k][:3], rods_final[k][3:]
        ])
        
        netF = ps.register_curve_network(f"tf{{suffix}}", nodesF, edges)
        netF.set_radius(radius * 1.5, relative=False)
        netF.set_color((1.0, 0.4, 0.2)) # Orangeish
        
        # Add labels maybe?
        
    ps.show()

if __name__ == "__main__":
    visualize()
"""
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    print(f"Visualization script written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Spot changing linking numbers')
    parser.add_argument('input_dir', type=str, help='Input run directory')
    parser.add_argument('--ar', type=float, default=100.0, help='Aspect ratio (default 100)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    raw_csv = input_path / "endpoints.csv"
    formatted_csv = input_path / "endpoints_formatted.csv"
    
    import re
    # Extract N from path
    match = re.search(r'N(\d+)', str(input_path))
    if match:
        n_rods = int(match.group(1))
    else:
        # fallback/guess
        n_rods = 200 # default
        print(f"Could not infer N from path, assuming N={n_rods}")

    if formatted_csv.exists():
        print("Using formatted csv")
        rods0 = load_rods_formatted(formatted_csv, 0)
        df = pd.read_csv(formatted_csv, comment='#')
        last_frame = df['frame'].max()
        rods_final = load_rods_formatted(formatted_csv, last_frame)
    elif raw_csv.exists():
        # Check if it has a header
        with open(raw_csv, 'r') as f:
            first_line = f.readline()
        
        if "frame" in first_line or "rod" in first_line:
            print("Detected headers in endpoints.csv, treating as formatted.")
            rods0 = load_rods_formatted(raw_csv, 0)
            df = pd.read_csv(raw_csv, comment='#')
            last_frame = df['frame'].max()
            rods_final = load_rods_formatted(raw_csv, last_frame)
        else:
            print(f"Using raw csv with N={n_rods} (no headers detected)")
            rods0 = load_rods_raw(raw_csv, n_rods, 0)
            rods_final = load_rods_raw(raw_csv, n_rods, -1)
    else:
        print("No data found")
        return

    print(f"Loaded {len(rods0)} rods.")
    
    # Compute X
    print("Computing X0...")
    X0 = compute_linking_matrix(rods0)
    print("Computing X_final...")
    X_final = compute_linking_matrix(rods_final)
    
    # Compute v
    print("Computing v0...")
    v0 = compute_vorticity_tensor(X0)
    print("Computing v_final...")
    v_final = compute_vorticity_tensor(X_final)
    
    # Diff
    changed = find_diffs(v0, v_final)
    print(f"Found {len(changed)} changed triples.")
    
    if len(changed) > 0:
        print("First 10 changed triples:")
        sorted_changed = sorted(list(changed))
        for (i, j, k) in sorted_changed[:10]:
            val0 = v0.get((i, j, k), 0)
            valF = v_final.get((i, j, k), 0)
            print(f"  ({i}, {j}, {k}): {val0} -> {valF}")
            
        # Export visualization
        output_viz = "viz_triples.py"
        write_polyscope_script(output_viz, rods0, rods_final, sorted_changed, args.ar)
        print("You can run 'python viz_triples.py' locally (install polyscope first) to see the changes.")
    else:
        print("No changes found.")

if __name__ == '__main__':
    main()
