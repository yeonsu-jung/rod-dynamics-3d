#!/usr/bin/env python3
"""
plot_results.py

Plots results from the existing_configs_N200 box sweep.
Generates:
1. Individual summary plots for each run.
2. Overlaid comparisons for Entanglement, Relative Displacement, and Overlap vs Time.
"""

import os
import glob
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path

def parse_folder_info(folder_name):
    """
    Extracts AR (and potentially other info) from the folder name.
    Expected format example: ...-N0200-AR0100-Scale1...
    """
    info = {'name': folder_name}
    
    # regex for AR
    match_ar = re.search(r'AR(\d+)', folder_name)
    if match_ar:
        info['AR'] = float(match_ar.group(1))
    else:
        info['AR'] = 0.0 # Fallback
        
    # regex for fSigma (if present in name)
    match_f = re.search(r'fSig([\d\.e\-\+]+)', folder_name)
    if match_f:
        info['fSigma'] = float(match_f.group(1))
    else:
        info['fSigma'] = 0.0

    return info

def plot_individual_run(df, info, output_dir):
    """
    Creates a dashboard of plots for a single simulation run.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"Run: {info['name']}\nAR={info['AR']}, fSig={info['fSigma']}", fontsize=14)
    
    # 1. Entanglement Sum
    axes[0, 0].plot(df['frame'], df['ent_sum'])
    axes[0, 0].set_title('Entanglement Sum')
    axes[0, 0].set_ylabel('Sum Abs Linking Number')
    
    # 2. Entanglement Pairs
    axes[0, 1].plot(df['frame'], df['ent_pairs'])
    axes[0, 1].set_title('Entanglement Pairs')
    
    # 3. Relative Displacement Sq
    axes[1, 0].plot(df['frame'], df['reldisp_sq'])
    axes[1, 0].set_title('Relative Displacement Squared')
    axes[1, 0].set_ylabel('MSRD')
    
    # 4. Max Overlap
    axes[1, 1].plot(df['frame'], df['max_overlap'], color='orange')
    axes[1, 1].set_title('Max Pairwise Overlap')
    
    # 5. Gyration Radius Sq
    axes[2, 0].plot(df['frame'], df['gyration_sq'], color='green')
    axes[2, 0].set_title('Squared Radius of Gyration')
    
    # 6. Kinetic Energy
    axes[2, 1].plot(df['frame'], df['KE'], color='red')
    axes[2, 1].set_title('Total Kinetic Energy')
    axes[2, 1].set_yscale('log')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = output_dir / f"summary_{info['name']}.png"
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved individual plot: {save_path}")

def plot_overlaid(dfs, metric, metric_name, output_dir, log_x=False, log_y=False, suffix=""):
    """
    Plots a specific metric overlaid for all runs, colored by Aspect Ratio.
    """
    plt.figure(figsize=(10, 6))
    
    # Sort keys by AR for cleaner legend
    sorted_runs = sorted(dfs.items(), key=lambda x: x[1]['info']['AR'])
    
    for name, data in sorted_runs:
        df = data['df']
        info = data['info']
        ar = int(info['AR'])
        label = f"AR={ar}"
        
        plt.plot(df['frame'], df[metric], label=label, alpha=0.7, linewidth=1.5)
        
    plt.title(f"{metric_name} vs Time (Overlaid)")
    plt.xlabel("Frame")
    plt.ylabel(metric_name)
    
    if log_x:
        plt.xscale('log')
    if log_y:
        plt.yscale('log')
        
    plt.legend(title="Aspect Ratio")
    plt.grid(True, linestyle='--', alpha=0.5, which="both")
    
    save_path = output_dir / f"overlaid_{metric}{suffix}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved overlaid plot: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot results from experiment runs.")
    parser.add_argument('--results-dir', type=str, 
                        default="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/existing_configs_N200",
                        help='Directory containing run subfolders.')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Find all output.csv files
    # output.csv is usually at the top level of the run folder
    run_folders = [f for f in results_dir.iterdir() if f.is_dir() and (f / "output.csv").exists()]
    
    if not run_folders:
        print("No run folders with 'output.csv' found.")
        return
        
    print(f"Found {len(run_folders)} runs to plot.")
    
    all_data = {}
    
    for folder in run_folders:
        csv_path = folder / "output.csv"
        try:
            df = pd.read_csv(csv_path)
            
            # Basic validation
            required_cols = ['frame', 'ent_sum', 'reldisp_sq']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {folder.name}: Missing expected columns in output.csv")
                continue
                
            info = parse_folder_info(folder.name)
            
            # Plot individual
            plot_individual_run(df, info, plots_dir)
            
            all_data[folder.name] = {'df': df, 'info': info}
            
        except Exception as e:
            print(f"Error processing {folder.name}: {e}")
            
    # Overlaid Plots
    if all_data:
        print("Generating overlaid comparisons...")
        
        # Standard Linear
        plot_overlaid(all_data, 'ent_sum', 'Entanglement Sum', plots_dir)
        plot_overlaid(all_data, 'reldisp_sq', 'Relative Displacement Sq', plots_dir)
        plot_overlaid(all_data, 'max_overlap', 'Max Overlap', plots_dir, log_y=True) # Usually log-y makes sense here
        plot_overlaid(all_data, 'gyration_sq', 'Gyration Radius Sq', plots_dir)
        
        # Semilogy (Log Y)
        plot_overlaid(all_data, 'ent_sum', 'Entanglement Sum', plots_dir, log_y=True, suffix="_semilogy")
        plot_overlaid(all_data, 'reldisp_sq', 'Relative Displacement Sq', plots_dir, log_y=True, suffix="_semilogy")
        plot_overlaid(all_data, 'gyration_sq', 'Gyration Radius Sq', plots_dir, log_y=True, suffix="_semilogy")
        
        # LogLog (Log X, Log Y)
        plot_overlaid(all_data, 'ent_sum', 'Entanglement Sum', plots_dir, log_x=True, log_y=True, suffix="_loglog")
        plot_overlaid(all_data, 'reldisp_sq', 'Relative Displacement Sq', plots_dir, log_x=True, log_y=True, suffix="_loglog")
        plot_overlaid(all_data, 'max_overlap', 'Max Overlap', plots_dir, log_x=True, log_y=True, suffix="_loglog")
        plot_overlaid(all_data, 'gyration_sq', 'Gyration Radius Sq', plots_dir, log_x=True, log_y=True, suffix="_loglog")

if __name__ == "__main__":
    main()
