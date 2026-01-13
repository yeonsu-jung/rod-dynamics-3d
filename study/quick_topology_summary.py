#!/usr/bin/env python3
"""
Quick summary script for topological invariants.
Computes only the essential invariants without the full per-pair matrix.
"""

import numpy as np
import pandas as pd
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_topology_summary.py <input_csv> [frame_number]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print(f"Loading rod data from {input_file}...")
    rods, df = load_rods_from_csv(input_file, frame)
    N = len(rods)
    
    print(f"Number of rods: {N}")
    print(f"\nComputing linking matrix ({N}×{N})...")
    X = compute_linking_matrix(rods)
    
    print("Computing spectrum...")
    eigenvalues = np.linalg.eigvals(X)
    
    print("Computing total chirality...")
    C = compute_total_chirality(X)
    
    print("Computing per-rod chirality...")
    c_i = compute_per_rod_chirality(X)
    
    # Print summary
    print("\n" + "="*70)
    print("TOPOLOGICAL INVARIANTS SUMMARY")
    print("="*70)
    print(f"Number of rods: {N}")
    print(f"\nLinking Matrix Statistics:")
    print(f"  Non-zero entries: {np.count_nonzero(X)}/{N*N}")
    print(f"  Positive links: {np.sum(X > 0)}")
    print(f"  Negative links: {np.sum(X < 0)}")
    
    print(f"\nTotal Chirality:")
    print(f"  C = {C}")
    
    print(f"\nPer-Rod Chirality c_i:")
    print(f"  Mean:   {np.mean(c_i):.2f}")
    print(f"  Std:    {np.std(c_i):.2f}")
    print(f"  Median: {np.median(c_i):.2f}")
    print(f"  Range:  [{np.min(c_i)}, {np.max(c_i)}]")
    
    print(f"\nSpectrum:")
    print(f"  Eigenvalues (first 10):")
    for i, eig in enumerate(eigenvalues[:10]):
        if abs(eig.real) < 1e-10:
            print(f"    λ_{i:3d} = {eig.imag:+.6f}i")
        else:
            print(f"    λ_{i:3d} = {eig.real:+.6f} {eig.imag:+.6f}i")
    if len(eigenvalues) > 10:
        print(f"    ... ({len(eigenvalues) - 10} more)")
    
    print(f"\n  Max |Re(λ)|: {np.max(np.abs(eigenvalues.real)):.2e}")
    print(f"  Max |Im(λ)|: {np.max(np.abs(eigenvalues.imag)):.2e}")
    
    print("="*70)
    
    # Save compact summary
    output_file = f"topology_summary_frame{frame if frame is not None else 0}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Topological Invariants Summary\n")
        f.write(f"{'='*70}\n")
        f.write(f"Number of rods: {N}\n")
        f.write(f"Total chirality C: {C}\n")
        f.write(f"\nPer-rod chirality statistics:\n")
        f.write(f"  Mean: {np.mean(c_i):.2f}\n")
        f.write(f"  Std:  {np.std(c_i):.2f}\n")
        f.write(f"  Range: [{np.min(c_i)}, {np.max(c_i)}]\n")
        f.write(f"\nPer-rod chirality values:\n")
        for i, val in enumerate(c_i):
            f.write(f"  c_{i:3d} = {val:6d}\n")
    
    print(f"\nSaved summary to {output_file}")


if __name__ == '__main__':
    main()
