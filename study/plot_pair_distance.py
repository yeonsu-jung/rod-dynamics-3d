#!/usr/bin/env python3
"""
Plot average pairwise distance (stable core and total packing) vs AR.
Overlays different friction coefficients (mu).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
import os

def plot_pair_distance(csv_files, output_dir='study/stable_core_analysis', n_value=None):
    """
    Generate overlaid plots of pairwise distance vs AR for different mu values.
    
    Args:
        csv_files: List of combined CSV files
        output_dir: Directory to save plots
        n_value: N value for title
    """
    # Load and combine data
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            
    if not dfs:
        print("No data found")
        return

    data = pd.concat(dfs, ignore_index=True)
    
    # Check columns
    required_cols = ['AR', 'avg_dist', 'all_avg_dist']
    if not all(col in data.columns for col in required_cols):
        print(f"Error: Missing columns. Data columns: {data.columns}")
        return
        
    # Check mu
    if 'mu' in data.columns:
        data = data.sort_values(['mu', 'AR'])
        has_mu = True
        mu_values = sorted(data['mu'].unique())
        print(f"Found {len(mu_values)} mu values: {mu_values}")
    else:
        data = data.sort_values('AR')
        has_mu = False
        mu_values = [None]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    title_suffix = f" (N={n_value})" if n_value else ""
    
    # Colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(mu_values)))
    
    # Loop over mu
    for idx, mu in enumerate(mu_values):
        if has_mu:
            mu_data = data[data['mu'] == mu]
            label = f'μ={mu}'
            color = colors[idx]
        else:
            mu_data = data
            label = 'All data'
            color = 'blue'
            
        # Group metrics
        grouped = mu_data.groupby('AR').agg({
            'avg_dist': ['mean', 'std'],
            'all_avg_dist': ['mean', 'std']
        })
        
        ar_values = grouped.index.values
        
        # Plot 1: Stable Core Distance
        ax = axes[0]
        y_mean = grouped[('avg_dist', 'mean')].values
        y_std = grouped[('avg_dist', 'std')].values
        ax.errorbar(ar_values, y_mean, yerr=y_std, fmt='o-', label=label, color=color, capsize=4, alpha=0.8)
        
        # Plot 2: All Packing Distance
        ax = axes[1]
        y_mean = grouped[('all_avg_dist', 'mean')].values
        y_std = grouped[('all_avg_dist', 'std')].values
        ax.errorbar(ar_values, y_mean, yerr=y_std, fmt='s-', label=label, color=color, capsize=4, alpha=0.8)

    # Configure axes
    for i, metric_name in enumerate(['Stable Core', 'Entire Packing']):
        ax = axes[i]
        ax.set_xlabel('Aspect Ratio (AR)', fontsize=12)
        ax.set_ylabel('Avg Pairwise Distance', fontsize=12)
        ax.set_title(f'{metric_name} Avg Distance vs AR{title_suffix}', fontsize=14)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')

    plt.tight_layout()
    
    # Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"pair_distance_vs_ar_N{n_value}.png" if n_value else "pair_distance_vs_ar.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot pairwise distance metrics vs AR')
    parser.add_argument('csv_files', nargs='+', help='CSV files')
    parser.add_argument('--output-dir', default='study/stable_core_analysis', help='Output directory')
    parser.add_argument('--n-value', type=int, help='N value')
    
    args = parser.parse_args()
    
    # Expand globs
    csv_files = []
    for p in args.csv_files:
        csv_files.extend(glob.glob(p))
        
    if not csv_files:
        print("No files found")
        return
        
    plot_pair_distance(csv_files, args.output_dir, args.n_value)

if __name__ == "__main__":
    main()
