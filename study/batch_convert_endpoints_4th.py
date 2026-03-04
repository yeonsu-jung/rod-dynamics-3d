#!/usr/bin/env python3
"""
Batch convert all endpoints.csv to endpoints_formatted.csv in 4th iteration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from multiprocessing import Pool, cpu_count

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

def convert_single_run(run_dir):
    """Convert endpoints.csv to endpoints_formatted.csv for a single run."""
    try:
        endpoints_file = run_dir / "endpoints.csv"
        formatted_file = run_dir / "endpoints_formatted.csv"
        
        if not endpoints_file.exists():
            return f"SKIP: {run_dir.name} (no endpoints.csv)"
        
        if formatted_file.exists():
            return f"EXISTS: {run_dir.name}"
        
        # Get AR from directory name
        n, ar, mu = parse_metadata_from_path(run_dir)
        if ar is None:
            return f"ERROR: {run_dir.name} (cannot parse AR)"
        
        # Read raw endpoints (no header)
        data = np.loadtxt(endpoints_file, delimiter=',')
        
        # Determine rod radius from AR
        rod_length = 1.0
        rod_radius = rod_length / ar / 2
        
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
        
        return f"CONVERTED: {run_dir.name}"
    
    except Exception as e:
        return f"ERROR: {run_dir.name} - {str(e)}"

def main():
    base_dir = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs_mujoco/relaxation_3rd_multithreading_4th_iterated_runs")
    
    # Find all run directories
    run_dirs = []
    for n_dir in sorted(base_dir.glob("N*")):
        if not n_dir.is_dir():
            continue
        for run_dir in n_dir.glob("*_RUN_*"):
            if run_dir.is_dir():
                run_dirs.append(run_dir)
    
    print(f"Found {len(run_dirs)} run directories")
    print(f"Using {cpu_count()} CPU cores for parallel processing")
    print("="*60)
    
    # Process in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(convert_single_run, run_dirs)
    
    # Summarize results
    converted = sum(1 for r in results if r.startswith("CONVERTED"))
    existed = sum(1 for r in results if r.startswith("EXISTS"))
    skipped = sum(1 for r in results if r.startswith("SKIP"))
    errors = sum(1 for r in results if r.startswith("ERROR"))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total runs: {len(run_dirs)}")
    print(f"Converted: {converted}")
    print(f"Already existed: {existed}")
    print(f"Skipped (no endpoints.csv): {skipped}")
    print(f"Errors: {errors}")
    
    if errors > 0:
        print("\nErrors:")
        for r in results:
            if r.startswith("ERROR"):
                print(f"  {r}")
    
    print("="*60)
    print("Done!")

if __name__ == '__main__':
    main()
