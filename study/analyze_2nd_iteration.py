#!/usr/bin/env python3
"""
Analyze 2nd iteration N200 and N500 datasets for stable core and chirality changes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from compute_topology import load_rods_from_csv, compute_linking_matrix, compute_total_chirality
from find_stable_core import compute_vorticity_tensor, find_diffs, find_stable_core

def analyze_run(endpoints_file, mu, ar, n):
    """Analyze a single run for chirality and stable core."""
    print(f"  Analyzing N={n}, AR={ar}, μ={mu}...")
    
    try:
        rods0, _ = load_rods_from_csv(endpoints_file, 0)
        rods_final, df = load_rods_from_csv(endpoints_file, None)  # Get last frame
        max_frame = df['frame'].max()
        rods_final, _ = load_rods_from_csv(endpoints_file, max_frame)
    except Exception as e:
        print(f"    Error: {e}")
        return None
    
    N_rods = len(rods0)
    
    # Compute chirality
    X0 = compute_linking_matrix(rods0)
    X_final = compute_linking_matrix(rods_final)
    
    C0 = compute_total_chirality(X0)
    C_final = compute_total_chirality(X_final)
    delta_C = C_final - C0
    
    # Compute stable core
    v0 = compute_vorticity_tensor(X0)
    v_final = compute_vorticity_tensor(X_final)
    changed = find_diffs(v0, v_final)
    core = find_stable_core(N_rods, changed)
    
    print(f"    C_initial={C0}, C_final={C_final}, ΔC={delta_C}")
    print(f"    Stable core: {len(core)}/{N_rods} ({len(core)/N_rods*100:.1f}%)")
    
    return {
        'n': n,
        'ar': ar,
        'mu': mu,
        'C_initial': C0,
        'C_final': C_final,
        'delta_C': delta_C,
        'abs_delta_C': abs(delta_C),
        'core_size': len(core),
        'core_fraction': len(core) / N_rods,
        'n_changes': len(changed),
        'max_frame': max_frame
    }

def main():
    # Define datasets
    base_n200 = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs_mujoco/relaxation_3rd_multithreading_2nd_iterated_runs/mujoco_entangled_N200_20260103-1624/N200/199,97,131")
    base_n500 = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs_mujoco/relaxation_3rd_multithreading_2nd_iterated_runs/mujoco_entangled_N500_20260103-1859/N500/20,909,910")
    
    mu_values = [0.05, 0.1, 0.15, 0.2, 0.4, 1.0]
    ar_target = 300  # Focus on AR=300
    
    results_n200 = []
    results_n500 = []
    
    print("="*60)
    print("Analyzing N=200, AR=300")
    print("="*60)
    for mu in mu_values:
        # Pattern: directory name contains the parameters
        pattern = f"*_N200_mu{mu:.4f}_AR{ar_target}_*"
        dirs = list(base_n200.glob(pattern))
        
        if not dirs:
            print(f"  No directory found for μ={mu}")
            continue
        
        endpoints_file = dirs[0] / "endpoints_formatted.csv"
        if not endpoints_file.exists():
            print(f"  No endpoints file in {dirs[0].name}")
            continue
            
        result = analyze_run(endpoints_file, mu, ar_target, 200)
        if result:
            results_n200.append(result)
    
    print("\n" + "="*60)
    print("Analyzing N=500, AR=300")
    print("="*60)
    for mu in mu_values:
        pattern = f"*_N500_mu{mu:.4f}_AR{ar_target}_*"
        dirs = list(base_n500.glob(pattern))
        
        if not dirs:
            print(f"  No directory found for μ={mu}")
            continue
        
        endpoints_file = dirs[0] / "endpoints_formatted.csv"
        if not endpoints_file.exists():
            print(f"  No endpoints file in {dirs[0].name}")
            continue
            
        result = analyze_run(endpoints_file, mu, ar_target, 500)
        if result:
            results_n500.append(result)
    
    # Save results
    df_n200 = pd.DataFrame(results_n200)
    df_n500 = pd.DataFrame(results_n500)
    
    output_dir = Path("study/2nd_iteration_analysis")
    output_dir.mkdir(exist_ok=True)
    
    df_n200.to_csv(output_dir / "n200_ar300_analysis.csv", index=False)
    df_n500.to_csv(output_dir / "n500_ar300_analysis.csv", index=False)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print("\nN=200, AR=300:")
    print(df_n200[['mu', 'C_initial', 'C_final', 'abs_delta_C', 'core_size', 'core_fraction']].to_string(index=False))
    
    print("\nN=500, AR=300:")
    print(df_n500[['mu', 'C_initial', 'C_final', 'abs_delta_C', 'core_size', 'core_fraction']].to_string(index=False))
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: |ΔC| vs μ for both N
    ax1.plot(df_n200['mu'], df_n200['abs_delta_C'], 'o-', label='N=200', linewidth=2, markersize=8)
    ax1.plot(df_n500['mu'], df_n500['abs_delta_C'], 's-', label='N=500', linewidth=2, markersize=8)
    ax1.set_xlabel('Friction Coefficient (μ)', fontsize=12)
    ax1.set_ylabel('|ΔC|', fontsize=12)
    ax1.set_title('Topological Change vs Friction (AR=300)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Core size vs μ
    ax2.plot(df_n200['mu'], df_n200['core_size'], 'o-', label='N=200', linewidth=2, markersize=8)
    ax2.plot(df_n500['mu'], df_n500['core_size'], 's-', label='N=500', linewidth=2, markersize=8)
    ax2.set_xlabel('Friction Coefficient (μ)', fontsize=12)
    ax2.set_ylabel('Stable Core Size (# rods)', fontsize=12)
    ax2.set_title('Stable Core Size vs Friction (AR=300)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # Plot 3: Core fraction vs μ
    ax3.plot(df_n200['mu'], df_n200['core_fraction']*100, 'o-', label='N=200', linewidth=2, markersize=8)
    ax3.plot(df_n500['mu'], df_n500['core_fraction']*100, 's-', label='N=500', linewidth=2, markersize=8)
    ax3.set_xlabel('Friction Coefficient (μ)', fontsize=12)
    ax3.set_ylabel('Stable Core Fraction (%)', fontsize=12)
    ax3.set_title('Stable Core Fraction vs Friction (AR=300)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Plot 4: Number of changed triples vs μ
    ax4.plot(df_n200['mu'], df_n200['n_changes'], 'o-', label='N=200', linewidth=2, markersize=8)
    ax4.plot(df_n500['mu'], df_n500['n_changes'], 's-', label='N=500', linewidth=2, markersize=8)
    ax4.set_xlabel('Friction Coefficient (μ)', fontsize=12)
    ax4.set_ylabel('Number of Changed Triples', fontsize=12)
    ax4.set_title('Topological Rearrangements vs Friction (AR=300)', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / "2nd_iteration_analysis_ar300.png", dpi=150)
    print(f"\nPlot saved: {output_dir / '2nd_iteration_analysis_ar300.png'}")
    
    print(f"Data saved: {output_dir / 'n200_ar300_analysis.csv'}")
    print(f"Data saved: {output_dir / 'n500_ar300_analysis.csv'}")

if __name__ == '__main__':
    main()
