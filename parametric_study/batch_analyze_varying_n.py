#!/usr/bin/env python3
"""
Batch analysis script for varying-n runs: analyze all subfolders and aggregate results.

Usage:
    python batch_analyze_varying_n.py [--base-dir DIR] [--make-plots] [--outdir DIR]
    
Example:
    python batch_analyze_varying_n.py --base-dir /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/varying-n --make-plots --outdir analysis_varying_n
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def extract_params_from_dirname(dirname):
    """Extract aspect ratio, friction, and noise from directory name."""
    # Format: 20251121-015656_RUN_aspect_LD10_mu0.00_noise1.0e-01_varying-n
    params = {}
    
    parts = dirname.split('_')
    for part in parts:
        if part.startswith('LD'):
            try:
                params['aspect_ratio'] = float(part[2:])
            except:
                pass
        elif part.startswith('mu'):
            try:
                params['friction'] = float(part[2:])
            except:
                pass
        elif part.startswith('noise'):
            try:
                # Next part should be the value
                idx = parts.index(part)
                if idx + 1 < len(parts):
                    params['noise'] = float(parts[idx + 1])
            except:
                pass
    
    return params

def analyze_single_run(run_dir):
    """Analyze a single run and return statistics."""
    run_dir = Path(run_dir)
    
    result = {
        'run_dir': str(run_dir),
        'run_name': run_dir.name,
        'success': False
    }
    
    # Extract parameters from directory name
    params = extract_params_from_dirname(run_dir.name)
    result.update(params)
    
    # Load profile.csv for KE data
    profile_csv = run_dir / "profile.csv"
    if not profile_csv.exists():
        print(f"  [SKIP] No profile.csv in {run_dir.name}")
        return result
    
    try:
        df_profile = pd.read_csv(profile_csv)
        ke = df_profile['KE'].to_numpy()
        frames = df_profile['frame'].to_numpy()
        
        # Store trajectory data
        result['frames'] = frames
        result['ke'] = ke
        
        # Full statistics
        result['ke_mean'] = float(np.mean(ke))
        result['ke_std'] = float(np.std(ke))
        result['ke_min'] = float(np.min(ke))
        result['ke_max'] = float(np.max(ke))
        result['ke_final'] = float(ke[-1])
        
        # Latter half statistics (equilibrated)
        half_idx = len(ke) // 2
        ke_late = ke[half_idx:]
        result['ke_late_mean'] = float(np.mean(ke_late))
        result['ke_late_std'] = float(np.std(ke_late))
        result['ke_late_cv'] = float(np.std(ke_late) / np.mean(ke_late)) if np.mean(ke_late) > 0 else np.nan
        
        # Growth rate
        start_idx = int(0.3 * len(ke))
        end_idx = int(0.9 * len(ke))
        ke_fit = ke[start_idx:end_idx]
        frames_fit = frames[start_idx:end_idx]
        
        valid = ke_fit > 0
        if np.sum(valid) > 10:
            log_ke = np.log(ke_fit[valid])
            t_fit = frames_fit[valid]
            coeffs = np.polyfit(t_fit, log_ke, 1)
            result['growth_rate'] = float(coeffs[0])
        else:
            result['growth_rate'] = np.nan
        
        # Load com.csv if available
        com_csv = run_dir / "com.csv"
        if com_csv.exists():
            df_com = pd.read_csv(com_csv)
            
            # Handle different column name formats
            if 'com_x' in df_com.columns:
                com_x = df_com['com_x'].to_numpy()
                com_y = df_com['com_y'].to_numpy()
                com_z = df_com['com_z'].to_numpy()
            elif 'x' in df_com.columns:
                com_x = df_com['x'].to_numpy()
                com_y = df_com['y'].to_numpy()
                com_z = df_com['z'].to_numpy()
            else:
                print(f"  [WARN] Unexpected COM columns in {run_dir.name}: {df_com.columns.tolist()}")
                com_x = com_y = com_z = None
            
            if com_x is not None:
                # Displacement
                dx = com_x[-1] - com_x[0]
                dy = com_y[-1] - com_y[0]
                dz = com_z[-1] - com_z[0]
                total_displacement = np.sqrt(dx**2 + dy**2 + dz**2)
                
                result['com_displacement'] = float(total_displacement)
                result['com_dx'] = float(dx)
                result['com_dy'] = float(dy)
                result['com_dz'] = float(dz)
                
                # Drift rate
                com_frames = df_com['frame'].to_numpy()
                drift_rate = total_displacement / (com_frames[-1] - com_frames[0]) if len(com_frames) > 1 else 0
                result['com_drift_rate'] = float(drift_rate)
                
                # Store trajectory
                result['com_frames'] = com_frames
                result['com_x'] = com_x
                result['com_y'] = com_y
                result['com_z'] = com_z
        
        result['success'] = True
        print(f"  [OK] {run_dir.name}: KE={result['ke_late_mean']:.4e}, growth={result['growth_rate']:.2e}")
        
    except Exception as e:
        print(f"  [ERROR] {run_dir.name}: {e}")
        return result
    
    return result

def make_individual_plots(result, run_dir):
    """Create plots for a single run."""
    run_dir = Path(run_dir)
    figs_dir = run_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    
    # KE time trace
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result['frames'], result['ke'], 'b-', linewidth=1, alpha=0.7)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Kinetic Energy', fontsize=12)
    ax.set_title(f"KE Dynamics\n{run_dir.name}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(figs_dir / "ke_dynamics.png", dpi=150)
    plt.close()
    
    # COM trajectory if available
    if 'com_x' in result:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # XY projection
        axes[0, 0].plot(result['com_x'], result['com_y'], 'b-', linewidth=0.5, alpha=0.6)
        axes[0, 0].plot(result['com_x'][0], result['com_y'][0], 'go', markersize=8, label='Start')
        axes[0, 0].plot(result['com_x'][-1], result['com_y'][-1], 'ro', markersize=8, label='End')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].set_title('COM XY Projection')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axis('equal')
        
        # XZ projection
        axes[0, 1].plot(result['com_x'], result['com_z'], 'b-', linewidth=0.5, alpha=0.6)
        axes[0, 1].plot(result['com_x'][0], result['com_z'][0], 'go', markersize=8, label='Start')
        axes[0, 1].plot(result['com_x'][-1], result['com_z'][-1], 'ro', markersize=8, label='End')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Z')
        axes[0, 1].set_title('COM XZ Projection')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axis('equal')
        
        # YZ projection
        axes[1, 0].plot(result['com_y'], result['com_z'], 'b-', linewidth=0.5, alpha=0.6)
        axes[1, 0].plot(result['com_y'][0], result['com_z'][0], 'go', markersize=8, label='Start')
        axes[1, 0].plot(result['com_y'][-1], result['com_z'][-1], 'ro', markersize=8, label='End')
        axes[1, 0].set_xlabel('Y')
        axes[1, 0].set_ylabel('Z')
        axes[1, 0].set_title('COM YZ Projection')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        # Time evolution
        axes[1, 1].plot(result['com_frames'], result['com_x'], 'r-', linewidth=1, alpha=0.7, label='X')
        axes[1, 1].plot(result['com_frames'], result['com_y'], 'g-', linewidth=1, alpha=0.7, label='Y')
        axes[1, 1].plot(result['com_frames'], result['com_z'], 'b-', linewidth=1, alpha=0.7, label='Z')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Position')
        axes[1, 1].set_title('COM Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f"COM Trajectory\n{run_dir.name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(figs_dir / "com_trajectory.png", dpi=150)
        plt.close()

def make_aggregate_plots(results, outdir):
    """Create aggregate plots across all runs."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating aggregate plots...")
    
    # Filter successful runs
    valid_results = [r for r in results if r['success']]
    
    if not valid_results:
        print("  No valid results to plot!")
        return
    
    # Group by aspect ratio
    aspect_ratios = sorted(set(r['aspect_ratio'] for r in valid_results if 'aspect_ratio' in r))
    colors_ar = plt.cm.viridis(np.linspace(0, 1, len(aspect_ratios)))
    
    # Plot 1: KE overlay by aspect ratio
    fig, ax = plt.subplots(figsize=(12, 7))
    for ar, color in zip(aspect_ratios, colors_ar):
        runs = [r for r in valid_results if r.get('aspect_ratio') == ar]
        for r in runs:
            label = f'L/D={ar:.0f}' if r == runs[0] else None
            ax.semilogy(r['frames'], r['ke'], color=color, alpha=0.3, linewidth=0.8, label=label)
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Kinetic Energy', fontsize=12)
    ax.set_title('KE Dynamics by Aspect Ratio', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'ke_overlay_by_aspect_ratio.png', dpi=150)
    plt.close()
    print(f"  Saved: {outdir / 'ke_overlay_by_aspect_ratio.png'}")
    
    # Plot 2: Aggregate KE statistics by aspect ratio (mean ± std)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: KE by aspect ratio
    ax = axes[0, 0]
    for ar, color in zip(aspect_ratios, colors_ar):
        runs = [r for r in valid_results if r.get('aspect_ratio') == ar]
        if not runs:
            continue
        
        min_len = min(len(r['ke']) for r in runs)
        frames = runs[0]['frames'][:min_len]
        ke_stack = np.array([r['ke'][:min_len] for r in runs])
        ke_mean = np.mean(ke_stack, axis=0)
        ke_std = np.std(ke_stack, axis=0)
        
        ax.semilogy(frames, ke_mean, color=color, linewidth=2, label=f'L/D={ar:.0f}')
        ax.fill_between(frames, ke_mean - ke_std, ke_mean + ke_std, color=color, alpha=0.2)
    
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Kinetic Energy', fontsize=11)
    ax.set_title('KE by Aspect Ratio (mean ± std)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Top-right: KE by friction
    ax = axes[0, 1]
    frictions = sorted(set(r['friction'] for r in valid_results if 'friction' in r))
    colors_f = plt.cm.plasma(np.linspace(0, 1, len(frictions)))
    
    for friction, color in zip(frictions, colors_f):
        runs = [r for r in valid_results if r.get('friction') == friction]
        if not runs:
            continue
        
        min_len = min(len(r['ke']) for r in runs)
        frames = runs[0]['frames'][:min_len]
        ke_stack = np.array([r['ke'][:min_len] for r in runs])
        ke_mean = np.mean(ke_stack, axis=0)
        ke_std = np.std(ke_stack, axis=0)
        
        ax.semilogy(frames, ke_mean, color=color, linewidth=2, label=f'μ={friction:.2f}')
        ax.fill_between(frames, ke_mean - ke_std, ke_mean + ke_std, color=color, alpha=0.2)
    
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Kinetic Energy', fontsize=11)
    ax.set_title('KE by Friction (mean ± std)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: KE by noise
    ax = axes[1, 0]
    noises = sorted(set(r['noise'] for r in valid_results if 'noise' in r))
    colors_n = plt.cm.coolwarm(np.linspace(0, 1, len(noises)))
    
    for noise, color in zip(noises, colors_n):
        runs = [r for r in valid_results if r.get('noise') == noise]
        if not runs:
            continue
        
        min_len = min(len(r['ke']) for r in runs)
        frames = runs[0]['frames'][:min_len]
        ke_stack = np.array([r['ke'][:min_len] for r in runs])
        ke_mean = np.mean(ke_stack, axis=0)
        ke_std = np.std(ke_stack, axis=0)
        
        ax.semilogy(frames, ke_mean, color=color, linewidth=2, label=f'σ={noise:.1e}')
        ax.fill_between(frames, ke_mean - ke_std, ke_mean + ke_std, color=color, alpha=0.2)
    
    ax.set_xlabel('Frame', fontsize=11)
    ax.set_ylabel('Kinetic Energy', fontsize=11)
    ax.set_title('KE by Noise (mean ± std)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Summary statistics bar chart
    ax = axes[1, 1]
    summary_data = []
    labels = []
    for ar in aspect_ratios:
        runs = [r for r in valid_results if r.get('aspect_ratio') == ar]
        if runs:
            ke_means = [r['ke_late_mean'] for r in runs]
            summary_data.append(np.mean(ke_means))
            labels.append(f'L/D={ar:.0f}')
    
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, summary_data, color=colors_ar[:len(labels)])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean KE (latter half)', fontsize=11)
    ax.set_title('Average Equilibrated KE', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(outdir / 'aggregate_ke_statistics.png', dpi=150)
    plt.close()
    print(f"  Saved: {outdir / 'aggregate_ke_statistics.png'}")
    
    # Plot 3: COM displacement summary
    com_results = [r for r in valid_results if 'com_displacement' in r]
    if com_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # By aspect ratio
        ax = axes[0]
        summary_data = []
        error_data = []
        labels = []
        for ar in aspect_ratios:
            runs = [r for r in com_results if r.get('aspect_ratio') == ar]
            if runs:
                displacements = [r['com_displacement'] for r in runs]
                summary_data.append(np.mean(displacements))
                error_data.append(np.std(displacements))
                labels.append(f'L/D={ar:.0f}')
        
        x_pos = np.arange(len(labels))
        ax.bar(x_pos, summary_data, yerr=error_data, color=colors_ar[:len(labels)], 
               capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('COM Displacement', fontsize=11)
        ax.set_title('COM Displacement by Aspect Ratio', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # By friction
        ax = axes[1]
        summary_data = []
        error_data = []
        labels = []
        for friction in frictions:
            runs = [r for r in com_results if r.get('friction') == friction]
            if runs:
                displacements = [r['com_displacement'] for r in runs]
                summary_data.append(np.mean(displacements))
                error_data.append(np.std(displacements))
                labels.append(f'μ={friction:.2f}')
        
        x_pos = np.arange(len(labels))
        ax.bar(x_pos, summary_data, yerr=error_data, color=colors_f[:len(labels)], 
               capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('COM Displacement', fontsize=11)
        ax.set_title('COM Displacement by Friction', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(outdir / 'com_displacement_summary.png', dpi=150)
        plt.close()
        print(f"  Saved: {outdir / 'com_displacement_summary.png'}")

def save_summary_table(results, outdir):
    """Save summary statistics to CSV."""
    outdir = Path(outdir)
    
    # Extract relevant fields for summary
    summary_rows = []
    for r in results:
        if not r['success']:
            continue
        
        row = {
            'run_name': r['run_name'],
            'aspect_ratio': r.get('aspect_ratio', np.nan),
            'friction': r.get('friction', np.nan),
            'noise': r.get('noise', np.nan),
            'ke_mean': r.get('ke_mean', np.nan),
            'ke_std': r.get('ke_std', np.nan),
            'ke_late_mean': r.get('ke_late_mean', np.nan),
            'ke_late_std': r.get('ke_late_std', np.nan),
            'ke_late_cv': r.get('ke_late_cv', np.nan),
            'growth_rate': r.get('growth_rate', np.nan),
            'com_displacement': r.get('com_displacement', np.nan),
            'com_drift_rate': r.get('com_drift_rate', np.nan),
        }
        summary_rows.append(row)
    
    df = pd.DataFrame(summary_rows)
    df = df.sort_values(['aspect_ratio', 'friction', 'noise'])
    
    csv_file = outdir / 'summary_table.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nSaved summary table: {csv_file}")
    print(f"  {len(df)} successful runs")

def main():
    parser = argparse.ArgumentParser(description="Batch analyze varying-n runs")
    parser.add_argument('--base-dir', type=str,
                       default='/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/varying-n',
                       help='Base directory containing run folders')
    parser.add_argument('--make-plots', action='store_true',
                       help='Generate individual plots for each run')
    parser.add_argument('--outdir', type=str, default='analysis_varying_n',
                       help='Output directory for aggregate results')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"ERROR: Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    # Find all run directories
    run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and '_RUN_' in d.name])
    
    print("=" * 80)
    print(f"BATCH ANALYSIS: varying-n runs")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Found {len(run_dirs)} run directories")
    print(f"Output directory: {args.outdir}")
    print("=" * 80)
    
    # Analyze all runs
    results = []
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"\n[{i}/{len(run_dirs)}] Analyzing {run_dir.name}...")
        result = analyze_single_run(run_dir)
        results.append(result)
        
        # Make individual plots if requested
        if args.make_plots and result['success']:
            try:
                make_individual_plots(result, run_dir)
                print(f"  [PLOT] Created individual plots in {run_dir}/figs/")
            except Exception as e:
                print(f"  [WARN] Failed to create plots: {e}")
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Save summary table
    save_summary_table(results, outdir)
    
    # Create aggregate plots
    make_aggregate_plots(results, outdir)
    
    print("\n" + "=" * 80)
    print("BATCH ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {outdir}")
    print(f"  - summary_table.csv")
    print(f"  - ke_overlay_by_aspect_ratio.png")
    print(f"  - aggregate_ke_statistics.png")
    print(f"  - com_displacement_summary.png")
    if args.make_plots:
        print(f"\nIndividual plots saved to each run's figs/ subdirectory")
    print("=" * 80)

if __name__ == "__main__":
    main()
