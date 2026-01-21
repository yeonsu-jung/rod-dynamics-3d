#!/usr/bin/env python3
"""
Compare topology changes between 2nd and 3rd iteration runs.
Specifically for N200 AR300 runs.
"""

import sys
sys.path.insert(0, '/n/home01/yjung/Github/rod-dynamics-3d/study')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from compute_topology import load_rods_from_csv, compute_linking_matrix, compute_total_chirality

def analyze_run(endpoints_file):
    """Analyze initial and final chirality for a single run."""
    # Load first and last frames
    df = pd.read_csv(endpoints_file, comment='#')
    frames = sorted(df['frame'].unique())
    
    if len(frames) < 2:
        return None
    
    first_frame = frames[0]
    last_frame = frames[-1]
    
    # Compute chirality for first frame
    rods_initial, _ = load_rods_from_csv(endpoints_file, first_frame)
    X_initial = compute_linking_matrix(rods_initial)
    C_initial = compute_total_chirality(X_initial)
    
    # Compute chirality for last frame
    rods_final, _ = load_rods_from_csv(endpoints_file, last_frame)
    X_final = compute_linking_matrix(rods_final)
    C_final = compute_total_chirality(X_final)
    
    delta_C = C_final - C_initial
    
    return {
        'C_initial': C_initial,
        'C_final': C_final,
        'delta_C': delta_C,
        'abs_delta_C': abs(delta_C),
        'first_frame': first_frame,
        'last_frame': last_frame,
        'num_frames': len(frames)
    }

def main():
    dir_2nd = Path('/n/home01/yjung/Github/rod-dynamics-3d/study/topology_study_run_comparison/n200_ar300_2nd_run')
    dir_3rd = Path('/n/home01/yjung/Github/rod-dynamics-3d/study/topology_study_run_comparison/n200_ar300_3rd_run')
    
    # Friction values
    mu_values = [0.05, 0.1, 0.15, 0.2, 0.4, 1.0]
    
    results_2nd = []
    results_3rd = []
    
    print("Analyzing 2nd iteration runs...")
    for mu in mu_values:
        file_2nd = dir_2nd / f"endpoints_formatted_n200_ar300_mu{mu}.csv"
        if file_2nd.exists():
            print(f"  μ={mu}...")
            result = analyze_run(file_2nd)
            if result:
                result['mu'] = mu
                results_2nd.append(result)
    
    print("\nAnalyzing 3rd iteration runs...")
    for mu in mu_values:
        file_3rd = dir_3rd / f"endpoints_formatted_n200_ar300_mu{mu}.csv"
        if file_3rd.exists():
            print(f"  μ={mu}...")
            result = analyze_run(file_3rd)
            if result:
                result['mu'] = mu
                results_3rd.append(result)
    
    # Convert to DataFrames
    df_2nd = pd.DataFrame(results_2nd)
    df_3rd = pd.DataFrame(results_3rd)
    
    # Print comparison
    print("\n" + "="*80)
    print("TOPOLOGY CHANGE COMPARISON: N200 AR300")
    print("="*80)
    print("\n2nd Iteration:")
    print(df_2nd[['mu', 'C_initial', 'C_final', 'delta_C', 'abs_delta_C']].to_string(index=False))
    print("\n3rd Iteration:")
    print(df_3rd[['mu', 'C_initial', 'C_final', 'delta_C', 'abs_delta_C']].to_string(index=False))
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: |ΔC| vs μ
    ax1.plot(df_2nd['mu'], df_2nd['abs_delta_C'], 'o-', label='2nd Iteration', linewidth=2, markersize=8)
    ax1.plot(df_3rd['mu'], df_3rd['abs_delta_C'], 's-', label='3rd Iteration', linewidth=2, markersize=8)
    ax1.set_xlabel('Friction Coefficient (μ)', fontsize=12)
    ax1.set_ylabel('|ΔC| = |C_final - C_initial|', fontsize=12)
    ax1.set_title('Topology Change vs Friction\n(N=200, AR=300)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: ΔC (signed) vs μ
    ax2.plot(df_2nd['mu'], df_2nd['delta_C'], 'o-', label='2nd Iteration', linewidth=2, markersize=8)
    ax2.plot(df_3rd['mu'], df_3rd['delta_C'], 's-', label='3rd Iteration', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Friction Coefficient (μ)', fontsize=12)
    ax2.set_ylabel('ΔC = C_final - C_initial', fontsize=12)
    ax2.set_title('Signed Topology Change vs Friction\n(N=200, AR=300)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    output_file = '/n/home01/yjung/Github/rod-dynamics-3d/study/topology_study_run_comparison/comparison_n200_ar300.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved plot: {output_file}")
    
    # Save CSV
    csv_file = '/n/home01/yjung/Github/rod-dynamics-3d/study/topology_study_run_comparison/comparison_n200_ar300.csv'
    comparison = pd.DataFrame({
        'mu': df_2nd['mu'],
        '2nd_C_initial': df_2nd['C_initial'],
        '2nd_C_final': df_2nd['C_final'],
        '2nd_delta_C': df_2nd['delta_C'],
        '2nd_abs_delta_C': df_2nd['abs_delta_C'],
        '3rd_C_initial': df_3rd['C_initial'],
        '3rd_C_final': df_3rd['C_final'],
        '3rd_delta_C': df_3rd['delta_C'],
        '3rd_abs_delta_C': df_3rd['abs_delta_C'],
    })
    comparison.to_csv(csv_file, index=False)
    print(f"Saved data: {csv_file}")
    
    print("="*80)

if __name__ == '__main__':
    main()
