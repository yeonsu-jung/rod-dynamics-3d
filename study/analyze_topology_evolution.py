#!/usr/bin/env python3
"""
Analyze topological invariants over time for a rod packing trajectory.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from compute_topology import (
    load_rods_from_csv,
    compute_linking_matrix,
    compute_total_chirality,
    compute_per_rod_chirality
)


def analyze_frame(filepath, frame):
    """Analyze a single frame and return key invariants."""
    rods, _ = load_rods_from_csv(filepath, frame)
    X = compute_linking_matrix(rods)
    C = compute_total_chirality(X)
    c_i = compute_per_rod_chirality(X)
    
    # Compute additional metrics
    num_positive_links = np.sum(X > 0)
    num_negative_links = np.sum(X < 0)
    
    return {
        'frame': frame,
        'total_chirality': C,
        'per_rod_mean': np.mean(c_i),
        'per_rod_std': np.std(c_i),
        'per_rod_min': np.min(c_i),
        'per_rod_max': np.max(c_i),
        'per_rod_range': np.max(c_i) - np.min(c_i),
        'num_positive_links': num_positive_links,
        'num_negative_links': num_negative_links,
        'link_imbalance': num_positive_links - num_negative_links
    }


def compare_frames(filepath, frames):
    """Compare topological invariants across multiple frames."""
    results = []
    
    for frame in frames:
        print(f"Analyzing frame {frame}...")
        result = analyze_frame(filepath, frame)
        results.append(result)
    
    return pd.DataFrame(results)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_topology_evolution.py <input_csv> [frame1 frame2 ...]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Determine which frames to analyze
    if len(sys.argv) > 2:
        frames = [int(f) for f in sys.argv[2:]]
    else:
        # Default: analyze first, middle, and last frames
        df = pd.read_csv(input_file, comment='#')
        max_frame = df['frame'].max()
        frames = [0, max_frame // 2, max_frame]
    
    print(f"Analyzing frames: {frames}")
    print()
    
    # Analyze frames
    results = compare_frames(input_file, frames)
    
    # Print comparison table
    print("\n" + "="*80)
    print("TOPOLOGICAL INVARIANTS COMPARISON")
    print("="*80)
    print(results.to_string(index=False))
    print("="*80)
    
    # Compute changes
    print("\nCHANGES OVER TIME:")
    print("-"*80)
    
    if len(frames) >= 2:
        for i in range(1, len(frames)):
            prev = results.iloc[i-1]
            curr = results.iloc[i]
            
            print(f"\nFrame {int(prev['frame'])} → Frame {int(curr['frame'])}:")
            print(f"  ΔC (total chirality):     {curr['total_chirality'] - prev['total_chirality']:+.0f}")
            print(f"  Δ(per-rod range):         {curr['per_rod_range'] - prev['per_rod_range']:+.0f}")
            print(f"  Δ(link imbalance):        {curr['link_imbalance'] - prev['link_imbalance']:+.0f}")
    
    # Check if topology is preserved
    print("\n" + "="*80)
    print("TOPOLOGICAL CONSTRAINT ANALYSIS:")
    print("="*80)
    
    C_values = results['total_chirality'].values
    if len(np.unique(C_values)) == 1:
        print(f"✓ Total chirality C is CONSTANT: {C_values[0]}")
        print("  → Topology is PRESERVED (isotopy class unchanged)")
    else:
        print(f"✗ Total chirality C CHANGES: {C_values}")
        print("  → Topology is NOT preserved (configuration crossed topological barriers)")
        print("  → This suggests rods passed through each other or boundary effects")
    
    # Scalar measure of topological constraint
    print("\nSCALAR MEASURES OF TOPOLOGICAL CONSTRAINT:")
    print("-"*80)
    print(f"Total chirality C:           {results['total_chirality'].values}")
    print(f"  Variance:                  {np.var(C_values):.2f}")
    print(f"  Max change:                {np.max(np.abs(np.diff(C_values))) if len(C_values) > 1 else 0}")
    
    per_rod_range = results['per_rod_range'].values
    print(f"\nPer-rod chirality range:     {per_rod_range}")
    print(f"  Variance:                  {np.var(per_rod_range):.2f}")
    print(f"  Max change:                {np.max(np.abs(np.diff(per_rod_range))) if len(per_rod_range) > 1 else 0:.0f}")
    
    print("="*80)
    
    # Save results
    output_file = "topology_evolution.csv"
    results.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")


if __name__ == '__main__':
    main()
