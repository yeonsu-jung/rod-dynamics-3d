#!/usr/bin/env python3
"""
post_analyze_soft_contact.py

Analyze results from soft contact parametric sweep: Effects of friction coefficient (mu) and noise amplitude on KE evolution.

Scans run directories, extracts KE data from profile.csv, computes statistics, generates comprehensive plots.

Usage:
    python post_analyze_soft_contact.py --job-name soft_contact_sweep --make-plots --outdir analysis_soft_contact
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import argparse

def _as_float(v):
    """Convert value to float, handling various types."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return None
    return None

def find_friction_heuristic(obj):
    """Extract friction coefficient from scene dict.
    Tries common locations/keys, then falls back to DFS search.
    """
    if not isinstance(obj, (dict, list)):
        return float('nan')

    # 1) Prefer explicit path: scene -> bodies[0] -> friction
    try:
        scene = obj.get('scene', {}) if isinstance(obj, dict) else {}
        bodies = scene.get('bodies', []) if isinstance(scene, dict) else []
        if isinstance(bodies, list) and bodies:
            b0 = bodies[0]
            if isinstance(b0, dict):
                for k in ('friction', 'friction_s', 'friction_d'):
                    if k in b0:
                        v = _as_float(b0[k])
                        if v is not None:
                            return v
    except Exception:
        pass

    # 2) Try physics -> soft_contact -> mu
    try:
        physics = obj.get('physics', {}) if isinstance(obj, dict) else {}
        soft_contact = physics.get('soft_contact', {}) if isinstance(physics, dict) else {}
        if isinstance(soft_contact, dict) and 'mu' in soft_contact:
            v = _as_float(soft_contact['mu'])
            if v is not None:
                return v
    except Exception:
        pass

    # 3) Generic DFS for likely keys
    def dfs(o):
        if isinstance(o, dict):
            for k, val in o.items():
                if isinstance(val, (int, float, str)) and (k.lower() in ('friction', 'mu') or 'fric' in k.lower()):
                    fv = _as_float(val)
                    if fv is not None:
                        return fv
            for val in o.values():
                r = dfs(val)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for it in o:
                r = dfs(it)
                if r is not None:
                    return r
        return None

    r = dfs(obj)
    return r if r is not None else float('nan')

def find_noise_heuristic(obj):
    """Extract noise amplitude from scene dict.
    Tries common locations/keys, then falls back to DFS search.
    """
    if not isinstance(obj, (dict, list)):
        return float('nan')

    # 1) Prefer explicit path: scene -> randomForce -> fSigma
    try:
        scene = obj.get('scene', {}) if isinstance(obj, dict) else {}
        rf = scene.get('randomForce', {}) if isinstance(scene, dict) else {}
        if isinstance(rf, dict):
            for k in ('fSigma', 'tauMag', 'sigma'):
                if k in rf:
                    v = _as_float(rf[k])
                    if v is not None:
                        return v
    except Exception:
        pass

    # 2) Generic DFS for likely keys
    def dfs(o):
        if isinstance(o, dict):
            for k, val in o.items():
                if isinstance(val, (int, float, str)) and (k.lower() in ('fsigma', 'sigma', 'taumag', 'noise')):
                    fv = _as_float(val)
                    if fv is not None:
                        return fv
            for val in o.values():
                r = dfs(val)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for it in o:
                r = dfs(it)
                if r is not None:
                    return r
        return None

    r = dfs(obj)
    return r if r is not None else float('nan')

def analyze_run(run_dir):
    """Analyze a single run directory, extracting parameters and KE data."""
    csv_path = run_dir / "profile.csv"
    scene_path = run_dir / "scene.json"
    
    if not csv_path.exists() or not scene_path.exists():
        return None

    # Load scene to extract params
    try:
        with open(scene_path, 'r') as f:
            scene = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {scene_path}: {e}")
        return None

    friction = find_friction_heuristic(scene)
    noise = find_noise_heuristic(scene)

    if np.isnan(friction) or np.isnan(noise):
        print(f"Warning: Could not extract params from {scene_path}")
        # Try to infer from directory name: *_muX_noiseY_*
        try:
            name = run_dir.name
            if '_mu' in name and '_noise' in name:
                import re
                m1 = re.search(r"_mu([0-9\.eE+-]+)", name)
                m2 = re.search(r"_noise([0-9\.eE+-]+)", name)
                if m1 and np.isnan(friction):
                    friction = float(m1.group(1))
                if m2 and np.isnan(noise):
                    noise = float(m2.group(1))
        except Exception:
            pass
        
        if np.isnan(friction) or np.isnan(noise):
            return None

    # Load KE data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Warning: Could not load {csv_path}: {e}")
        return None
    
    if 'frame' not in df.columns or 'KE' not in df.columns:
        print(f"Warning: Missing required columns in {csv_path}")
        return None
    
    frames = df['frame'].values.astype(float)
    ke = df['KE'].values.astype(float)
    
    # Extract other columns if available
    result = {
        'frames': frames,
        'ke': ke,
        'friction': friction,
        'noise': noise,
        'run_dir': run_dir,
    }
    
    # Add contact count if available
    if 'n_contacts' in df.columns:
        result['n_contacts'] = df['n_contacts'].values.astype(float)
    
    return result

def compute_ke_statistics(ke, frames):
    """Compute statistics on KE data."""
    stats = {}
    
    # Mean and std of latter half
    n = len(ke)
    latter_half = ke[n//2:]
    stats['mean_ke'] = np.mean(latter_half)
    stats['std_ke'] = np.std(latter_half)
    stats['median_ke'] = np.median(latter_half)
    
    # Growth rate from linear fit in log space (middle 60%)
    try:
        start_idx = int(0.3 * n)
        end_idx = int(0.9 * n)
        t_fit = frames[start_idx:end_idx]
        ke_fit = ke[start_idx:end_idx]
        
        valid = ke_fit > 0
        if np.sum(valid) > 10:
            log_ke = np.log(ke_fit[valid])
            t_fit = t_fit[valid]
            coeffs = np.polyfit(t_fit, log_ke, 1)
            stats['growth_rate'] = coeffs[0]
        else:
            stats['growth_rate'] = np.nan
    except Exception:
        stats['growth_rate'] = np.nan
    
    return stats

def make_summary_table(data, stats, outdir):
    """Create a summary table of all runs."""
    rows = []
    for key, d in data.items():
        friction, noise = key
        s = stats.get(key, {})
        rows.append({
            'friction': friction,
            'noise': noise,
            'mean_ke': s.get('mean_ke', np.nan),
            'std_ke': s.get('std_ke', np.nan),
            'growth_rate': s.get('growth_rate', np.nan),
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['friction', 'noise'])
    
    csv_path = outdir / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary table to {csv_path}")
    
    return df

def make_heatmaps(data, stats, outdir):
    """Create heatmaps of KE statistics vs friction and noise."""
    frictions = sorted({d['friction'] for d in data.values()})
    noises = sorted({d['noise'] for d in data.values()})
    
    # Mean KE heatmap
    mean_ke_matrix = np.full((len(noises), len(frictions)), np.nan)
    for i, noise in enumerate(noises):
        for j, friction in enumerate(frictions):
            key = (friction, noise)
            if key in stats:
                mean_ke_matrix[i, j] = stats[key].get('mean_ke', np.nan)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.log10(mean_ke_matrix), aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(len(frictions)))
    ax.set_xticklabels([f"{f:.2f}" for f in frictions])
    ax.set_xlabel('Friction Coefficient μ', fontsize=12)
    ax.set_yticks(range(len(noises)))
    ax.set_yticklabels([f"{n:.0e}" for n in noises])
    ax.set_ylabel('Noise Amplitude σ', fontsize=12)
    ax.set_title('Mean Kinetic Energy (log10) - Soft Contact', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(Mean KE)', fontsize=12)
    plt.tight_layout()
    plt.savefig(outdir / 'heatmap_mean_ke.png', dpi=150)
    plt.close()
    print(f"Saved heatmap: {outdir / 'heatmap_mean_ke.png'}")
    
    # Growth rate heatmap
    growth_rate_matrix = np.full((len(noises), len(frictions)), np.nan)
    for i, noise in enumerate(noises):
        for j, friction in enumerate(frictions):
            key = (friction, noise)
            if key in stats:
                growth_rate_matrix[i, j] = stats[key].get('growth_rate', np.nan)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(growth_rate_matrix, aspect='auto', cmap='RdYlGn_r', origin='lower')
    ax.set_xticks(range(len(frictions)))
    ax.set_xticklabels([f"{f:.2f}" for f in frictions])
    ax.set_xlabel('Friction Coefficient μ', fontsize=12)
    ax.set_yticks(range(len(noises)))
    ax.set_yticklabels([f"{n:.0e}" for n in noises])
    ax.set_ylabel('Noise Amplitude σ', fontsize=12)
    ax.set_title('KE Growth Rate - Soft Contact', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Growth Rate (1/frame)', fontsize=12)
    plt.tight_layout()
    plt.savefig(outdir / 'heatmap_growth_rate.png', dpi=150)
    plt.close()
    print(f"Saved heatmap: {outdir / 'heatmap_growth_rate.png'}")

def plot_ke_by_friction(data, outdir):
    """Plot KE traces grouped by friction coefficient."""
    frictions = sorted({d['friction'] for d in data.values()})
    noises = sorted({d['noise'] for d in data.values()})
    
    n_frictions = len(frictions)
    ncols = 3
    nrows = (n_frictions + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows), sharex=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(noises)))
    
    for i, friction in enumerate(frictions):
        ax = axes[i]
        for j, noise in enumerate(noises):
            key = (friction, noise)
            if key in data:
                frames = data[key]['frames']
                ke = data[key]['ke']
                ax.semilogy(frames, ke, label=f'σ={noise:.0e}', color=colors[j], alpha=0.7)
        
        ax.set_title(f'μ = {friction:.2f}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Frame', fontsize=10)
        ax.set_ylabel('Kinetic Energy (J)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(len(frictions), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('KE Evolution: Soft Contact Model (by Friction)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outdir / 'ke_traces_by_friction.png', dpi=150)
    plt.close()
    print(f"Saved plot: {outdir / 'ke_traces_by_friction.png'}")

def plot_ke_by_noise(data, outdir):
    """Plot KE traces grouped by noise amplitude."""
    frictions = sorted({d['friction'] for d in data.values()})
    noises = sorted({d['noise'] for d in data.values()})
    
    n_noises = len(noises)
    ncols = 3
    nrows = (n_noises + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows), sharex=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(frictions)))
    
    for i, noise in enumerate(noises):
        ax = axes[i]
        for j, friction in enumerate(frictions):
            key = (friction, noise)
            if key in data:
                frames = data[key]['frames']
                ke = data[key]['ke']
                ax.semilogy(frames, ke, label=f'μ={friction:.2f}', color=colors[j], alpha=0.7)
        
        ax.set_title(f'σ = {noise:.0e}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Frame', fontsize=10)
        ax.set_ylabel('Kinetic Energy (J)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(len(noises), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('KE Evolution: Soft Contact Model (by Noise)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outdir / 'ke_traces_by_noise.png', dpi=150)
    plt.close()
    print(f"Saved plot: {outdir / 'ke_traces_by_noise.png'}")

def plot_growth_rate_analysis(stats, outdir):
    """Plot growth rate vs noise for each friction level."""
    frictions = sorted({k[0] for k in stats.keys()})
    noises = sorted({k[1] for k in stats.keys()})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(frictions)))
    
    for i, friction in enumerate(frictions):
        noise_vals = []
        growth_vals = []
        for noise in noises:
            key = (friction, noise)
            if key in stats and not np.isnan(stats[key].get('growth_rate', np.nan)):
                noise_vals.append(noise)
                growth_vals.append(stats[key]['growth_rate'])
        
        if noise_vals:
            ax.semilogx(noise_vals, growth_vals, 'o-', label=f'μ={friction:.2f}', 
                       color=colors[i], markersize=8, linewidth=2)
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Noise Amplitude σ', fontsize=12)
    ax.set_ylabel('KE Growth Rate (1/frame)', fontsize=12)
    ax.set_title('Energy Growth Rate vs Noise Amplitude', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'growth_rate_vs_noise.png', dpi=150)
    plt.close()
    print(f"Saved plot: {outdir / 'growth_rate_vs_noise.png'}")

def main():
    parser = argparse.ArgumentParser(description="Analyze soft contact parametric sweep results.")
    parser.add_argument('--job-name', type=str, required=True, help='Job name used in submission.')
    parser.add_argument('--runs-root', type=str, 
                       default='/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs',
                       help='Root directory for runs.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for plots and analysis.')
    parser.add_argument('--make-plots', action='store_true', help='Generate plots.')
    args = parser.parse_args()

    runs_root = Path(args.runs_root) / args.job_name
    if not runs_root.exists():
        raise SystemExit(f"Runs root not found: {runs_root}")

    print("=" * 80)
    print("SOFT CONTACT PARAMETRIC SWEEP ANALYSIS")
    print("=" * 80)
    print(f"Job name: {args.job_name}")
    print(f"Runs root: {runs_root}")
    print()

    # Find run directories
    run_dirs_file = runs_root / "run_dirs.txt"
    if run_dirs_file.exists():
        print(f"Loading run directories from {run_dirs_file}")
        with open(run_dirs_file, 'r') as f:
            run_dirs = [Path(line.strip()) for line in f if line.strip()]
    else:
        print("Scanning for run directories...")
        run_dirs = [d for d in runs_root.iterdir() if d.is_dir() and '_RUN_soft_' in d.name]

    print(f"Found {len(run_dirs)} run directories")
    print()

    # Analyze all runs
    data = {}
    stats = {}
    
    print("Analyzing runs...")
    for i, run_dir in enumerate(run_dirs, 1):
        result = analyze_run(run_dir)
        if result is None:
            print(f"  [{i}/{len(run_dirs)}] Skipped: {run_dir.name}")
            continue
        
        key = (result['friction'], result['noise'])
        data[key] = result
        stats[key] = compute_ke_statistics(result['ke'], result['frames'])
        
        print(f"  [{i}/{len(run_dirs)}] μ={result['friction']:.2f}, σ={result['noise']:.1e}: "
              f"mean_KE={stats[key]['mean_ke']:.2e}, growth={stats[key]['growth_rate']:.4e}")

    if not data:
        print("\nNo data found. Exiting.")
        return

    print()
    print(f"Successfully analyzed {len(data)} runs")
    print("=" * 80)
    print()

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Create summary table
    print("Creating summary table...")
    summary_df = make_summary_table(data, stats, outdir)

    # Generate plots if requested
    if args.make_plots:
        print()
        print("Generating plots...")
        make_heatmaps(data, stats, outdir)
        plot_ke_by_friction(data, outdir)
        plot_ke_by_noise(data, outdir)
        plot_growth_rate_analysis(stats, outdir)
        
        print()
        print("=" * 80)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {outdir}")
        print("=" * 80)
    else:
        print()
        print("Skipping plots (use --make-plots to generate)")

if __name__ == "__main__":
    main()
