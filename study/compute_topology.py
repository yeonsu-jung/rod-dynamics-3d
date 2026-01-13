#!/usr/bin/env python3
"""
Compute topological invariants for rod packing configurations.

This module computes:
1. Linking matrix X (pairwise linking numbers)
2. Spectrum of X (eigenvalues and characteristic polynomial)
3. Triple-vorticity summaries:
   - Total chirality C
   - Per-rod chirality c_i
   - Per-pair chirality c_ij

Based on the mathematical framework in rod_packing_topology.md
"""

import numpy as np
import pandas as pd
import json
from typing import Tuple, Dict, Optional
from pathlib import Path
import argparse


def compute_linking_number(rod_a: np.ndarray, rod_b: np.ndarray) -> int:
    """
    Compute the linking number between two oriented rods.
    
    Args:
        rod_a: Array of shape (6,) containing [x1, y1, z1, x2, y2, z2] for rod A
        rod_b: Array of shape (6,) containing [x1, y1, z1, x2, y2, z2] for rod B
    
    Returns:
        Linking number in {-1, 0, +1}
        
    The linking number is computed using:
    lk(L_A, L_B) = sign(det([A_ω - A_α, B_α - A_ω, B_ω - B_α]))
    
    where A_α, A_ω are the start and end points of rod A,
    and B_α, B_ω are the start and end points of rod B.
    """
    # Extract endpoints
    A_alpha = rod_a[:3]  # Start point of rod A
    A_omega = rod_a[3:]  # End point of rod A
    B_alpha = rod_b[:3]  # Start point of rod B
    B_omega = rod_b[3:]  # End point of rod B
    
    # Construct the matrix columns
    col1 = A_omega - A_alpha
    col2 = B_alpha - A_omega
    col3 = B_omega - B_alpha
    
    # Compute determinant
    matrix = np.column_stack([col1, col2, col3])
    det = np.linalg.det(matrix)
    
    # Return sign
    if abs(det) < 1e-10:  # Threshold for numerical zero
        return 0
    return int(np.sign(det))


def compute_linking_matrix(rods: np.ndarray) -> np.ndarray:
    """
    Compute the linking matrix X for a configuration of rods.
    
    Args:
        rods: Array of shape (N, 6) where each row is [x1, y1, z1, x2, y2, z2]
    
    Returns:
        Linking matrix X of shape (N, N) with entries in {-1, 0, +1}
        X is antisymmetric: X[i,j] = -X[j,i], X[i,i] = 0
    """
    N = len(rods)
    X = np.zeros((N, N), dtype=int)
    
    for i in range(N):
        for j in range(i + 1, N):
            lk = compute_linking_number(rods[i], rods[j])
            X[i, j] = lk
            X[j, i] = -lk  # Antisymmetry
    
    return X


def compute_spectrum(X: np.ndarray) -> Dict:
    """
    Compute the spectrum of the linking matrix.
    
    Args:
        X: Linking matrix (N, N)
    
    Returns:
        Dictionary containing:
        - eigenvalues: List of eigenvalues (complex)
        - eigenvalues_real: Real parts
        - eigenvalues_imag: Imaginary parts
        - char_poly_coeffs: Coefficients of characteristic polynomial
    """
    eigenvalues = np.linalg.eigvals(X)
    
    # Compute characteristic polynomial coefficients
    # Note: numpy.poly returns coefficients in descending order
    char_poly = np.poly(X)
    
    return {
        'eigenvalues': eigenvalues.tolist(),
        'eigenvalues_real': eigenvalues.real.tolist(),
        'eigenvalues_imag': eigenvalues.imag.tolist(),
        'char_poly_coeffs': char_poly.tolist()
    }


def compute_vorticity(X: np.ndarray, i: int, j: int, k: int) -> int:
    """
    Compute the vorticity for a triple of rods.
    
    Args:
        X: Linking matrix
        i, j, k: Indices of three rods
    
    Returns:
        v_ijk = x_ij * x_jk * x_ki ∈ {-1, +1}
    """
    return X[i, j] * X[j, k] * X[k, i]


def compute_total_chirality(X: np.ndarray) -> int:
    """
    Compute total chirality C = Σ_{i<j<k} v_ijk
    
    Args:
        X: Linking matrix (N, N)
    
    Returns:
        Total chirality C
    """
    N = len(X)
    C = 0
    
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                C += compute_vorticity(X, i, j, k)
    
    return C


def compute_per_rod_chirality(X: np.ndarray) -> np.ndarray:
    """
    Compute per-rod chirality c_i = Σ_{j<k, j,k≠i} v_ijk
    
    Args:
        X: Linking matrix (N, N)
    
    Returns:
        Array of shape (N,) containing c_i for each rod i
    """
    N = len(X)
    c = np.zeros(N, dtype=int)
    
    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            for k in range(j + 1, N):
                if k == i:
                    continue
                c[i] += compute_vorticity(X, i, j, k)
    
    return c


def compute_per_pair_chirality(X: np.ndarray) -> np.ndarray:
    """
    Compute per-pair chirality c_ij = Σ_{k≠i,j} v_ijk
    
    Args:
        X: Linking matrix (N, N)
    
    Returns:
        Array of shape (N, N) containing c_ij for each pair (i,j)
    """
    N = len(X)
    c_ij = np.zeros((N, N), dtype=int)
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for k in range(N):
                if k == i or k == j:
                    continue
                c_ij[i, j] += compute_vorticity(X, i, j, k)
    
    return c_ij


def load_rods_from_csv(filepath: str, frame: Optional[int] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load rod data from CSV file.
    
    Args:
        filepath: Path to CSV file
        frame: Frame number to load (if None, loads first frame)
    
    Returns:
        Tuple of (rods array, full dataframe)
        rods: Array of shape (N, 6) with columns [x1, y1, z1, x2, y2, z2]
    """
    # Read CSV, skipping comment lines
    df = pd.read_csv(filepath, comment='#')
    
    # Filter by frame if specified
    if frame is not None:
        df = df[df['frame'] == frame]
    else:
        # Use first frame
        frame = df['frame'].iloc[0]
        df = df[df['frame'] == frame]
    
    # Extract rod endpoints
    rods = df[['x1', 'y1', 'z1', 'x2', 'y2', 'z2']].values
    
    return rods, df


def compute_all_invariants(rods: np.ndarray) -> Dict:
    """
    Compute all topological invariants for a rod configuration.
    
    Args:
        rods: Array of shape (N, 6) containing rod endpoints
    
    Returns:
        Dictionary containing all computed invariants
    """
    N = len(rods)
    
    print(f"Computing linking matrix for {N} rods...")
    X = compute_linking_matrix(rods)
    
    print("Computing spectrum...")
    spectrum = compute_spectrum(X)
    
    print("Computing total chirality...")
    C = compute_total_chirality(X)
    
    print("Computing per-rod chirality...")
    c_i = compute_per_rod_chirality(X)
    
    print("Computing per-pair chirality...")
    c_ij = compute_per_pair_chirality(X)
    
    # Compute statistics
    results = {
        'num_rods': N,
        'linking_matrix': X.tolist(),
        'spectrum': spectrum,
        'total_chirality': int(C),
        'per_rod_chirality': {
            'values': c_i.tolist(),
            'mean': float(np.mean(c_i)),
            'std': float(np.std(c_i)),
            'min': int(np.min(c_i)),
            'max': int(np.max(c_i))
        },
        'per_pair_chirality': {
            'matrix': c_ij.tolist(),
            'mean': float(np.mean(c_ij[c_ij != 0])) if np.any(c_ij != 0) else 0.0,
            'std': float(np.std(c_ij[c_ij != 0])) if np.any(c_ij != 0) else 0.0,
            'min': int(np.min(c_ij)),
            'max': int(np.max(c_ij))
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute topological invariants for rod packings'
    )
    parser.add_argument('input', type=str, help='Input CSV file with rod data')
    parser.add_argument('--frame', type=int, default=None, 
                       help='Frame number to analyze (default: first frame)')
    parser.add_argument('--output', type=str, default='topology_results.json',
                       help='Output JSON file (default: topology_results.json)')
    parser.add_argument('--save-matrix', action='store_true',
                       help='Save linking matrix as CSV')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading rod data from {args.input}...")
    rods, df = load_rods_from_csv(args.input, args.frame)
    
    # Compute invariants
    results = compute_all_invariants(rods)
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Optionally save linking matrix
    if args.save_matrix:
        matrix_file = Path(args.output).stem + '_linking_matrix.csv'
        np.savetxt(matrix_file, results['linking_matrix'], fmt='%d', delimiter=',')
        print(f"Saved linking matrix to {matrix_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("TOPOLOGICAL INVARIANTS SUMMARY")
    print("="*60)
    print(f"Number of rods: {results['num_rods']}")
    print(f"Total chirality C: {results['total_chirality']}")
    print(f"\nPer-rod chirality c_i:")
    print(f"  Mean: {results['per_rod_chirality']['mean']:.2f}")
    print(f"  Std:  {results['per_rod_chirality']['std']:.2f}")
    print(f"  Range: [{results['per_rod_chirality']['min']}, {results['per_rod_chirality']['max']}]")
    print(f"\nPer-pair chirality c_ij:")
    print(f"  Mean: {results['per_pair_chirality']['mean']:.2f}")
    print(f"  Std:  {results['per_pair_chirality']['std']:.2f}")
    print(f"  Range: [{results['per_pair_chirality']['min']}, {results['per_pair_chirality']['max']}]")
    print(f"\nSpectrum (first 10 eigenvalues):")
    eigs = np.array(results['spectrum']['eigenvalues'])
    for i, eig in enumerate(eigs[:10]):
        print(f"  λ_{i}: {eig}")
    if len(eigs) > 10:
        print(f"  ... ({len(eigs) - 10} more)")
    print("="*60)


if __name__ == '__main__':
    main()
