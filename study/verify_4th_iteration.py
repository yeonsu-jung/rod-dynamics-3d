#!/usr/bin/env python3
"""
Convert raw endpoints.csv to endpoints_formatted.csv with proper headers.
Also verify that initial states are consistent across friction values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).parent))
from compute_topology import compute_linking_matrix, compute_total_chirality

def parse_metadata_from_path(run_dir):
    """Extract N, AR, mu from directory name."""
    dirname = run_dir.name
    n_match = re.search(r'N(\d+)', dirname)
    ar_match = re.search(r'AR(\d+)', dirname)
    mu_match = re.search(r'mu([\d.]+)', dirname)
    
    n = int(n_match.group(1)) if n_match else None
    ar = int(ar_match.group(1)) if ar_match else None
    mu = float(mu_match.group(1)) if mu_match else None
    
    return n, ar, mu

def convert_endpoints(run_dir, ar_value):
    """Convert endpoints.csv to endpoints_formatted.csv."""
    endpoints_file = run_dir / "endpoints.csv"
    formatted_file = run_dir / "endpoints_formatted.csv"
    
    if not endpoints_file.exists():
        return False
    
    if formatted_file.exists():
        return True  # Already converted
    
    # Read raw endpoints (no header)
    data = np.loadtxt(endpoints_file, delimiter=',')
    
    # Determine rod radius from AR
    rod_length = 1.0
    rod_radius = rod_length / ar_value / 2
    
    # Reshape to (num_frames, num_rods, 6)
    num_rods = data.shape[1] // 6
    num_frames = data.shape[0]
    
    # Create formatted dataframe
    rows = []
    for frame_idx in range(num_frames):
        for rod_idx in range(num_rods):
            start_col = rod_idx * 6
            row = {
                'frame': frame_idx,
                'id': rod_idx,
                'x1': data[frame_idx, start_col + 0],
                'y1': data[frame_idx, start_col + 1],
                'z1': data[frame_idx, start_col + 2],
                'x2': data[frame_idx, start_col + 3],
                'y2': data[frame_idx, start_col + 4],
                'z2': data[frame_idx, start_col + 5]
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Write with metadata header
    with open(formatted_file, 'w') as f:
        f.write(f"#rod_radius={rod_radius}\n")
        f.write(f"#rod_length={rod_length}\n")
        df.to_csv(f, index=False)
    
    return True

def verify_initial_consistency(base_dir, n_value, ar_value):
    """Verify that initial chirality is consistent across friction values."""
    print(f"\nVerifying N={n_value}, AR={ar_value}")
    print("="*60)
    
    results = []
    
    for run_dir in sorted(base_dir.glob(f"*_N{n_value}_mu*_AR{ar_value}_*")):
        n, ar, mu = parse_metadata_from_path(run_dir)
        
        formatted_file = run_dir / "endpoints_formatted.csv"
        if not formatted_file.exists():
            continue
        
        # Load frame 0
        df = pd.read_csv(formatted_file, comment='#')
        frame0 = df[df['frame'] == 0]
        
        # Extract rod endpoints
        rods = frame0[['x1', 'y1', 'z1', 'x2', 'y2', 'z2']].values
        
        # Compute chirality
        X = compute_linking_matrix(rods)
        C = compute_total_chirality(X)
        
        results.append({'mu': mu, 'C_initial': C, 'n_rods': len(rods)})
        print(f"  μ={mu:.4f}: C_initial={C}, N_rods={len(rods)}")
    
    if len(results) > 1:
        c_values = [r['C_initial'] for r in results]
        if len(set(c_values)) == 1:
            print(f"\n✅ PASS: All friction values have same initial chirality: {c_values[0]}")
            return True
        else:
            print(f"\n❌ FAIL: Initial chirality varies: {c_values}")
            return False
    
    return None

def main():
    base_dir = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs_mujoco/relaxation_3rd_multithreading_4th_iterated_runs")
    
    # Process N200 AR300 as a test case
    n_test = 200
    ar_test = 300
    
    print(f"Converting endpoints.csv to endpoints_formatted.csv for N={n_test}")
    print("="*60)
    
    converted_count = 0
    for run_dir in sorted((base_dir / f"N{n_test}").glob(f"*_N{n_test}_*_AR{ar_test}_*")):
        n, ar, mu = parse_metadata_from_path(run_dir)
        success = convert_endpoints(run_dir, ar)
        if success:
            converted_count += 1
            print(f"  Converted: {run_dir.name}")
    
    print(f"\nConverted {converted_count} runs")
    
    # Verify consistency
    verify_initial_consistency(base_dir / f"N{n_test}", n_test, ar_test)

if __name__ == '__main__':
    main()
