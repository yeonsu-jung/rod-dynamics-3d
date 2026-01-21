#!/usr/bin/env python3
"""
Analyze stable core size as a function of N (number of rods).
Processes multiple directories with different N values and computes stable core metrics.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from compute_topology import load_rods_from_csv, compute_linking_matrix
from find_stable_core import compute_vorticity_tensor, find_diffs, find_stable_core


def compute_segment_distance(rod_a, rod_b):
    """
    Compute minimum distance between two line segments.
    rod_a, rod_b: [x1, y1, z1, x2, y2, z2]
    """
    p1 = rod_a[:3]
    d1 = rod_a[3:] - p1
    p2 = rod_b[:3]
    d2 = rod_b[3:] - p2
    
    SMALL_NUM = 0.00000001
    
    u = d1
    v = d2
    w = p1 - p2
    
    a = np.dot(u,u)
    b = np.dot(u,v)
    c = np.dot(v,v)
    d = np.dot(u,w)
    e = np.dot(v,w)
    D = a*c - b*b
    
    sc, sN, sD = 0.0, 0.0, D
    tc, tN, tD = 0.0, 0.0, D
    
    if D < SMALL_NUM:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
            
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a
            
    sc = 0.0 if abs(sN) < SMALL_NUM else sN / sD
    tc = 0.0 if abs(tN) < SMALL_NUM else tN / tD
    
    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)


def compute_avg_stable_core_distance(rods, core_indices):
    """
    Compute average pairwise distance between rods in the stable core.
    """
    if len(core_indices) < 2:
        return 0.0
        
    distances = []
    core_list = list(core_indices)
    
    # Sample if core is too large
    if len(core_list) > 300:
        # Sample 10000 pairs
        for _ in range(10000):
            idx1 = np.random.choice(core_list)
            idx2 = np.random.choice(core_list)
            if idx1 != idx2:
                dist = compute_segment_distance(rods[idx1], rods[idx2])
                distances.append(dist)
    else:
        # Calculate for all pairs
        for i in range(len(core_list)):
            idx1 = core_list[i]
            for j in range(i + 1, len(core_list)):
                idx2 = core_list[j]
                dist = compute_segment_distance(rods[idx1], rods[idx2])
                distances.append(dist)
            
    return np.mean(distances) if distances else 0.0


def analyze_endpoints_file(filepath, frame_initial=0, frame_final=None):
    """
    Analyze a single endpoints file and compute stable core metrics.
    
    Args:
        filepath: Path to endpoints_formatted.csv
        frame_initial: Initial frame to analyze (default: 0)
        frame_final: Final frame to analyze (default: last frame)
    
    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing {filepath}...")
    
    try:
        # Load initial frame
        rods0, _ = load_rods_from_csv(filepath, frame_initial)
        
        # Determine final frame if not specified
        if frame_final is None:
            df = pd.read_csv(filepath, comment='#')
            frame_final = df['frame'].max()
        
        # Load final frame
        rods_final, _ = load_rods_from_csv(filepath, frame_final)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    N = len(rods0)
    
    # Compute Linking Matrices & Vorticities
    X0 = compute_linking_matrix(rods0)
    X_final = compute_linking_matrix(rods_final)
    
    v0 = compute_vorticity_tensor(X0)
    v_final = compute_vorticity_tensor(X_final)
    
    # Find Differences
    changed = find_diffs(v0, v_final)
    n_changes = len(changed)
    
    # Find Stable Core
    core = find_stable_core(N, changed)
    core_size = len(core)
    
    # Compute avg distance within stable core in FINAL frame
    avg_dist = compute_avg_stable_core_distance(rods_final, core)
    
    # Compute avg distance for ENTIRE packing in FINAL frame
    all_indices = list(range(N))
    all_avg_dist = compute_avg_stable_core_distance(rods_final, all_indices)
    
    print(f"  N={N}, Stable core size: {core_size} ({100*core_size/N:.1f}%), Core Avg Dist: {avg_dist:.4f}, All Avg Dist: {all_avg_dist:.4f}")
    
    return {
        'N': N,
        'core_size': core_size,
        'core_fraction': core_size / N,
        'n_changes': n_changes,
        'avg_dist': avg_dist,
        'all_avg_dist': all_avg_dist,
        'frame_initial': frame_initial,
        'frame_final': frame_final
    }


def parse_directory_params(dirname):
    """
    Extract N and AR from directory name.
    Expected format: ...N{value}_AR{value}... or similar
    """
    params = {}
    
    # Try to extract N
    import re
    n_match = re.search(r'[Nn](\d+)', dirname)
    if n_match:
        params['N'] = int(n_match.group(1))
    
    # Try to extract AR
    ar_match = re.search(r'[Aa][Rr](\d+)', dirname)
    if ar_match:
        params['AR'] = int(ar_match.group(1))
    
    # Try to extract mu (friction) - handle both 'Friction' and 'mu' patterns
    mu_match = re.search(r'[Ff]riction([\d.]+)', dirname)
    if not mu_match:
        mu_match = re.search(r'[Mm]u([\d.]+)', dirname)
    if mu_match:
        params['mu'] = float(mu_match.group(1))
    
    return params


def main():
    parser = argparse.ArgumentParser(description='Analyze stable core size vs N')
    parser.add_argument('input_dirs', nargs='+', help='Input directories or CSV files to analyze')
    parser.add_argument('--output', '-o', default='stable_core_vs_n.csv', 
                        help='Output CSV file (default: stable_core_vs_n.csv)')
    parser.add_argument('--frame-initial', type=int, default=0, 
                        help='Initial frame to analyze (default: 0)')
    parser.add_argument('--frame-final', type=int, default=None, 
                        help='Final frame to analyze (default: last frame)')
    parser.add_argument('--n-value', type=int, default=None,
                        help='Explicitly specify N value (overrides auto-detection)')
    
    args = parser.parse_args()
    
    results = []
    
    for input_path in args.input_dirs:
        if os.path.isdir(input_path):
            # Look for endpoints.csv in directory
            csv_path = os.path.join(input_path, 'endpoints.csv')
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found, skipping")
                continue
        else:
            csv_path = input_path
        
        # Analyze the file
        result = analyze_endpoints_file(csv_path, args.frame_initial, args.frame_final)
        
        if result:
            # Try to extract additional parameters from directory name
            dirname = os.path.basename(os.path.dirname(csv_path))
            params = parse_directory_params(dirname)
            
            # Override N if explicitly provided
            if args.n_value is not None:
                params['N'] = args.n_value
            
            result.update(params)
            
            results.append(result)
    
    if not results:
        print("No results to save")
        return
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    
    # Sort by N if available
    if 'N' in df.columns:
        df = df.sort_values('N')
    
    df.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")
    
    # Print summary
    print("\nResults Summary:")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
