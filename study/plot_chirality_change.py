#!/usr/bin/env python3
"""
Plot chirality change (final - initial) vs AR for different friction values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys


def parse_metadata(dirname):
    """Extract N, AR, mu from directory name."""
    n_match = re.search(r'N(\d+)', dirname)
    ar_match = re.search(r'AR(\d+)', dirname)
    mu_match = re.search(r'mu([\d.]+)', dirname)
    
    n = int(n_match.group(1)) if n_match else None
    ar = int(ar_match.group(1)) if ar_match else None
    mu = float(mu_match.group(1)) if mu_match else None
    
    return n, ar, mu


def load_topology_results(topology_dir):
    """Load all topology CSV results and extract chirality changes."""
    topology_dir = Path(topology_dir)
    
    results = []
    
    for csv_file in topology_dir.glob("*_topology.csv"):
        # Parse metadata from filename
        run_name = csv_file.stem.replace('_topology', '')
        n, ar, mu = parse_metadata(run_name)
        
        if n is None or ar is None or mu is None:
            continue
        
        # Load topology data
        df = pd.read_csv(csv_file)
        
        if len(df) < 2:
            continue
        
        # Get initial and final chirality
        initial_chirality = df.iloc[0]['total_chirality']
        final_chirality = df.iloc[-1]['total_chirality']
        chirality_change = final_chirality - initial_chirality
        
        results.append({
            'n': n,
            'ar': ar,
            'mu': mu,
            'initial_chirality': initial_chirality,
            'final_chirality': final_chirality,
            'chirality_change': chirality_change,
            'abs_chirality_change': abs(chirality_change)
        })
    
    return pd.DataFrame(results)


def plot_chirality_change_vs_ar(df, output_file='chirality_change_vs_ar.png'):
    """Plot chirality change vs AR, overlaid by friction."""
    
    # Get unique friction values
    mu_values = sorted(df['mu'].unique())
    
    plt.figure(figsize=(10, 6))
    
    # Color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(mu_values)))
    
    for i, mu in enumerate(mu_values):
        df_mu = df[df['mu'] == mu].sort_values('ar')
        
        plt.plot(df_mu['ar'], df_mu['chirality_change'], 
                marker='o', label=f'μ={mu:.2f}', color=colors[i], 
                linewidth=2, markersize=6, alpha=0.7)
    
    plt.xlabel('Aspect Ratio (AR)', fontsize=12)
    plt.ylabel('Chirality Change (ΔC = C_final - C_initial)', fontsize=12)
    plt.title('Topological Evolution: Chirality Change vs Aspect Ratio', fontsize=14)
    plt.legend(title='Friction', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def plot_abs_chirality_change_vs_ar(df, output_file='abs_chirality_change_vs_ar.png'):
    """Plot absolute chirality change vs AR, overlaid by friction."""
    
    # Get unique friction values
    mu_values = sorted(df['mu'].unique())
    
    plt.figure(figsize=(10, 6))
    
    # Color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(mu_values)))
    
    for i, mu in enumerate(mu_values):
        df_mu = df[df['mu'] == mu].sort_values('ar')
        
        plt.plot(df_mu['ar'], df_mu['abs_chirality_change'], 
                marker='o', label=f'μ={mu:.2f}', color=colors[i], 
                linewidth=2, markersize=6, alpha=0.7)
    
    plt.xlabel('Aspect Ratio (AR)', fontsize=12)
    plt.ylabel('|Chirality Change| (|ΔC|)', fontsize=12)
    plt.title('Topological Evolution: Absolute Chirality Change vs Aspect Ratio', fontsize=14)
    plt.legend(title='Friction', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_chirality_change.py <topology_analysis_dir> [output_dir]")
        sys.exit(1)
    
    topology_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else topology_dir
    
    print(f"Loading topology results from: {topology_dir}")
    df = load_topology_results(topology_dir)
    
    if len(df) == 0:
        print("No topology results found!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} topology results")
    print(f"\nN values: {sorted(df['n'].unique())}")
    print(f"AR values: {sorted(df['ar'].unique())}")
    print(f"μ values: {sorted(df['mu'].unique())}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("CHIRALITY CHANGE STATISTICS")
    print("="*60)
    print(f"Mean |ΔC|: {df['abs_chirality_change'].mean():.2f}")
    print(f"Max |ΔC|: {df['abs_chirality_change'].max():.2f}")
    print(f"Runs with ΔC = 0: {len(df[df['chirality_change'] == 0])}/{len(df)}")
    print(f"Runs with |ΔC| > 0: {len(df[df['abs_chirality_change'] > 0])}/{len(df)}")
    
    # Generate plots
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_chirality_change_vs_ar(df, output_dir / 'chirality_change_vs_ar.png')
    plot_abs_chirality_change_vs_ar(df, output_dir / 'abs_chirality_change_vs_ar.png')
    
    # Save summary CSV
    summary_file = output_dir / 'chirality_change_summary.csv'
    df.to_csv(summary_file, index=False)
    print(f"Saved summary: {summary_file}")
    
    print("\nPlotting complete!")


if __name__ == '__main__':
    main()
