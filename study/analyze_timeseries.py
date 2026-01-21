#!/usr/bin/env python3
"""
Analyze topological and geometric metrics over time.
Computes delta C (chirality change), average pair distance, and stable core size as functions of time.
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from compute_topology import (
    load_rods_from_csv,
    compute_linking_matrix,
    compute_total_chirality
)
from find_stable_core import compute_vorticity_tensor, find_diffs, find_stable_core


def compute_segment_distance(rod_a, rod_b):
    """Compute minimum distance between two line segments."""
    p1 = rod_a[:3]
    d1 = rod_a[3:] - p1
    p2 = rod_b[:3]
    d2 = rod_b[3:] - p2
    
    SMALL_NUM = 0.00000001
    u = d1; v = d2; w = p1 - p2
    a = np.dot(u,u); b = np.dot(u,v); c = np.dot(v,v)
    d = np.dot(u,w); e = np.dot(v,w)
    D = a*c - b*b
    
    sc, sN, sD = 0.0, 0.0, D
    tc, tN, tD = 0.0, 0.0, D
    
    if D < SMALL_NUM:
        sN = 0.0; sD = 1.0; tN = e; tD = c
    else:
        sN = (b*e - c*d); tN = (a*e - b*d)
        if sN < 0.0: sN = 0.0; tN = e; tD = c
        elif sN > sD: sN = sD; tN = e + b; tD = c
            
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0: sN = 0.0
        elif -d > a: sN = sD
        else: sN = -d; sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0: sN = 0.0
        elif (-d + b) > a: sN = sD
        else: sN = (-d + b); sD = a
            
    sc = 0.0 if abs(sN) < SMALL_NUM else sN / sD
    tc = 0.0 if abs(tN) < SMALL_NUM else tN / tD
    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)


def compute_avg_pair_distance(rods, max_samples=5000):
    """
    Compute average pairwise distance between all rods.
    Uses sampling for large N to keep computation tractable.
    """
    N = len(rods)
    
    if N < 2:
        return 0.0
    
    # For large N, sample pairs
    if N > 100:
        distances = []
        for _ in range(min(max_samples, N * (N-1) // 2)):
            idx1 = np.random.randint(0, N)
            idx2 = np.random.randint(0, N)
            if idx1 != idx2:
                dist = compute_segment_distance(rods[idx1], rods[idx2])
                distances.append(dist)
        return np.mean(distances)
    else:
        # Calculate all pairs for small N
        distances = []
        for i in range(N):
            for j in range(i + 1, N):
                dist = compute_segment_distance(rods[i], rods[j])
                distances.append(dist)
        return np.mean(distances)


def analyze_timeseries(filepath, frames=None, stride=None):
    """
    Analyze metrics over time from an endpoints CSV file.
    
    Args:
        filepath: Path to endpoints_formatted.csv
        frames: List of specific frames to analyze (optional)
        stride: Analyze every N-th frame (optional)
    
    Returns:
        DataFrame with time series data
    """
    print(f"Analyzing {filepath}...")
    
    # Load the CSV to determine available frames
    df = pd.read_csv(filepath, comment='#')
    available_frames = sorted(df['frame'].unique())
    
    # Determine which frames to analyze
    if frames is not None:
        analyze_frames = [f for f in frames if f in available_frames]
    elif stride is not None:
        analyze_frames = available_frames[::stride]
    else:
        # Default: analyze 20 evenly spaced frames
        n_samples = min(20, len(available_frames))
        indices = np.linspace(0, len(available_frames)-1, n_samples, dtype=int)
        analyze_frames = [available_frames[i] for i in indices]
    
    print(f"  Analyzing {len(analyze_frames)} frames out of {len(available_frames)} total")
    
    # Load initial frame for reference
    rods0, _ = load_rods_from_csv(filepath, analyze_frames[0])
    X0 = compute_linking_matrix(rods0)
    C0 = compute_total_chirality(X0)
    v0 = compute_vorticity_tensor(X0)
    
    N = len(rods0)
    
    results = []
    
    for frame in analyze_frames:
        print(f"  Processing frame {frame}...")
        
        try:
            rods, _ = load_rods_from_csv(filepath, frame)
            
            # Compute topology
            X = compute_linking_matrix(rods)
            C = compute_total_chirality(X)
            v = compute_vorticity_tensor(X)
            
            # Compute delta C from initial frame
            delta_C = abs(C - C0)
            
            # Compute stable core size
            changed = find_diffs(v0, v)
            core = find_stable_core(N, changed)
            core_size = len(core)
            
            # Compute average pair distance
            avg_dist = compute_avg_pair_distance(rods)
            
            results.append({
                'frame': frame,
                'total_chirality': C,
                'delta_C': delta_C,
                'core_size': core_size,
                'core_fraction': core_size / N,
                'avg_pair_distance': avg_dist,
                'n_changed_triples': len(changed)
            })
            
        except Exception as e:
            print(f"  Error processing frame {frame}: {e}")
            continue
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze topological and geometric metrics over time'
    )
    parser.add_argument('input_file', help='Input endpoints CSV file')
    parser.add_argument('--output', '-o', default='timeseries_analysis.csv',
                        help='Output CSV file (default: timeseries_analysis.csv)')
    parser.add_argument('--frames', nargs='+', type=int,
                        help='Specific frames to analyze')
    parser.add_argument('--stride', type=int,
                        help='Analyze every N-th frame')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    
    args = parser.parse_args()
    
    # Analyze the time series
    df = analyze_timeseries(args.input_file, frames=args.frames, stride=args.stride)
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")
    
    # Print summary
    print("\nTime Series Summary:")
    print(df.to_string(index=False))
    
    # Generate plots if requested
    if args.plot:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Delta C vs time
        axes[0, 0].plot(df['frame'], df['delta_C'], 'o-', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Frame', fontsize=12)
        axes[0, 0].set_ylabel(r'$|\Delta C|$', fontsize=12)
        axes[0, 0].set_title('Chirality Change vs Time', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Average pair distance vs time
        axes[0, 1].plot(df['frame'], df['avg_pair_distance'], 's-', 
                        linewidth=2, markersize=6, color='green')
        axes[0, 1].set_xlabel('Frame', fontsize=12)
        axes[0, 1].set_ylabel('Avg Pair Distance', fontsize=12)
        axes[0, 1].set_title('Average Pair Distance vs Time', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Stable core size vs time
        axes[1, 0].plot(df['frame'], df['core_size'], '^-', 
                        linewidth=2, markersize=6, color='red')
        axes[1, 0].set_xlabel('Frame', fontsize=12)
        axes[1, 0].set_ylabel('Stable Core Size', fontsize=12)
        axes[1, 0].set_title('Stable Core Size vs Time', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Core fraction vs time
        axes[1, 1].plot(df['frame'], df['core_fraction'], 'd-', 
                        linewidth=2, markersize=6, color='purple')
        axes[1, 1].set_xlabel('Frame', fontsize=12)
        axes[1, 1].set_ylabel('Stable Core Fraction', fontsize=12)
        axes[1, 1].set_title('Stable Core Fraction vs Time', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_plot = args.output.replace('.csv', '.png')
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_plot}")


if __name__ == '__main__':
    main()
