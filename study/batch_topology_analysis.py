#!/usr/bin/env python3
"""
Batch analyze topological evolution across multiple friction coefficients.
Creates a plot of |ΔC| vs μ.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from compute_topology import (
    load_rods_from_csv,
    compute_linking_matrix,
    compute_total_chirality
)


def extract_mu_from_filename(filename):
    """Extract friction coefficient from filename."""
    match = re.search(r'mu([0-9.]+?)(?:\.csv|_|$)', filename)
    if match:
        return float(match.group(1))
    return None


def analyze_dataset(filepath, frames_to_analyze=None):
    """Analyze a single dataset and return key metrics."""
    # Get mu from filename
    mu = extract_mu_from_filename(str(filepath))
    
    # Determine frames
    df = pd.read_csv(filepath, comment='#')
    max_frame = df['frame'].max()
    
    if frames_to_analyze is None:
        # Analyze first, middle, and last frames
        frames = [0, max_frame // 2, max_frame]
    else:
        frames = frames_to_analyze
    
    print(f"\nAnalyzing μ={mu}: {Path(filepath).name}")
    print(f"  Total frames: {max_frame + 1}")
    print(f"  Analyzing frames: {frames}")
    
    # Compute chirality for each frame
    C_values = []
    for frame in frames:
        rods, _ = load_rods_from_csv(filepath, frame)
        X = compute_linking_matrix(rods)
        C = compute_total_chirality(X)
        C_values.append(C)
        print(f"    Frame {frame}: C = {C}")
    
    # Compute metrics
    C_initial = C_values[0]
    C_final = C_values[-1]
    delta_C = C_final - C_initial
    abs_delta_C = abs(delta_C)
    percent_change = 100 * delta_C / C_initial if C_initial != 0 else float('inf')
    
    return {
        'mu': mu,
        'filename': Path(filepath).name,
        'total_frames': max_frame + 1,
        'C_initial': C_initial,
        'C_final': C_final,
        'delta_C': delta_C,
        'abs_delta_C': abs_delta_C,
        'percent_change': percent_change,
        'frames_analyzed': frames,
        'C_values': C_values
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_topology_analysis.py <file1> <file2> ...")
        sys.exit(1)
    
    files = sys.argv[1:]
    
    print("="*80)
    print("BATCH TOPOLOGICAL ANALYSIS")
    print("="*80)
    
    # Analyze all datasets
    results = []
    for filepath in files:
        try:
            result = analyze_dataset(filepath)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
    
    # Sort by mu
    results = sorted(results, key=lambda x: x['mu'])
    
    # Create summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    df_results = pd.DataFrame([{
        'mu': r['mu'],
        'C_initial': r['C_initial'],
        'C_final': r['C_final'],
        'ΔC': r['delta_C'],
        '|ΔC|': r['abs_delta_C'],
        '% change': r['percent_change']
    } for r in results])
    
    print(df_results.to_string(index=False))
    print("="*80)
    
    # Save results
    df_results.to_csv('topology_vs_friction.csv', index=False)
    print("\nSaved results to topology_vs_friction.csv")
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    mu_values = [r['mu'] for r in results]
    abs_delta_C = [r['abs_delta_C'] for r in results]
    delta_C = [r['delta_C'] for r in results]
    
    # Plot 1: |ΔC| vs μ
    ax1.plot(mu_values, abs_delta_C, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Friction coefficient μ', fontsize=12)
    ax1.set_ylabel('|ΔC| (absolute change in chirality)', fontsize=12)
    ax1.set_title('Topological Change vs Friction', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Add annotations
    for mu, abs_dc in zip(mu_values, abs_delta_C):
        ax1.annotate(f'{abs_dc:.0f}', 
                    xy=(mu, abs_dc), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=9)
    
    # Plot 2: ΔC vs μ (with sign)
    colors = ['red' if dc < 0 else 'blue' for dc in delta_C]
    ax2.scatter(mu_values, delta_C, s=100, c=colors, alpha=0.6, edgecolors='black')
    ax2.plot(mu_values, delta_C, '--', alpha=0.3, color='gray')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Friction coefficient μ', fontsize=12)
    ax2.set_ylabel('ΔC (signed change in chirality)', fontsize=12)
    ax2.set_title('Signed Topological Change vs Friction', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('topology_vs_friction.png', dpi=300, bbox_inches='tight')
    print("Saved plot to topology_vs_friction.png")
    
    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Find extremes
    max_change_idx = np.argmax(abs_delta_C)
    min_change_idx = np.argmin(abs_delta_C)
    
    print(f"\nMost topological change:")
    print(f"  μ = {results[max_change_idx]['mu']}: |ΔC| = {results[max_change_idx]['abs_delta_C']:.0f}")
    
    print(f"\nLeast topological change:")
    print(f"  μ = {results[min_change_idx]['mu']}: |ΔC| = {results[min_change_idx]['abs_delta_C']:.0f}")
    
    print(f"\nRatio (max/min): {abs_delta_C[max_change_idx] / abs_delta_C[min_change_idx]:.1f}×")
    
    # Check for power law
    if len(mu_values) >= 3:
        # Fit power law: |ΔC| ~ μ^α
        log_mu = np.log(mu_values)
        log_abs_delta_C = np.log(abs_delta_C)
        coeffs = np.polyfit(log_mu, log_abs_delta_C, 1)
        alpha = coeffs[0]
        
        print(f"\nPower law fit: |ΔC| ∝ μ^{alpha:.2f}")
        if alpha < 0:
            print(f"  → Lower friction allows MORE topological change")
        else:
            print(f"  → Higher friction allows MORE topological change")
    
    print("="*80)


if __name__ == '__main__':
    main()
