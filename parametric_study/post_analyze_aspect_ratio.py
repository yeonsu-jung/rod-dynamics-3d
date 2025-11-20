#!/usr/bin/env python3
"""
post_analyze_aspect_ratio.py

Aggregate and analyze results from aspect ratio parametric sweep.

Usage:
  python3 post_analyze_aspect_ratio.py --job-name aspect_ratio_test --make-plots --outdir analysis_aspect_ratio
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

def find_aspect_ratio_heuristic(obj):
    """Extract aspect ratio from scene JSON by computing L/D."""
    if isinstance(obj, dict):
        if "scene" in obj and "bodies" in obj["scene"]:
            bodies = obj["scene"]["bodies"]
            if isinstance(bodies, list) and len(bodies) > 0:
                body = bodies[0]
                if "length" in body and "diameter" in body:
                    length = float(body["length"])
                    diameter = float(body["diameter"])
                    if diameter > 0:
                        return length / diameter
        # DFS search
        for v in obj.values():
            result = find_aspect_ratio_heuristic(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_aspect_ratio_heuristic(item)
            if result is not None:
                return result
    return None

def find_friction_heuristic(obj):
    """Extract friction coefficient from scene JSON."""
    if isinstance(obj, dict):
        if "scene" in obj and "bodies" in obj["scene"]:
            bodies = obj["scene"]["bodies"]
            if isinstance(bodies, list) and len(bodies) > 0:
                body = bodies[0]
                if "friction" in body:
                    return float(body["friction"])
        if "physics" in obj and "soft_contact" in obj["physics"]:
            sc = obj["physics"]["soft_contact"]
            if "mu" in sc:
                return float(sc["mu"])
        # DFS search
        for v in obj.values():
            result = find_friction_heuristic(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_friction_heuristic(item)
            if result is not None:
                return result
    return None

def find_noise_heuristic(obj):
    """Extract noise amplitude from scene JSON."""
    if isinstance(obj, dict):
        if "scene" in obj and "randomForce" in obj["scene"]:
            rf = obj["scene"]["randomForce"]
            if "fSigma" in rf:
                return float(rf["fSigma"])
            if "tauMag" in rf:
                return float(rf["tauMag"])
        # DFS search
        for v in obj.values():
            result = find_noise_heuristic(v)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_noise_heuristic(item)
            if result is not None:
                return result
    return None

def analyze_run(run_dir):
    """Extract parameters and KE data from a single run directory."""
    run_dir = Path(run_dir)
    
    # Load scene.json for parameters
    scene_json = run_dir / "scene.json"
    if not scene_json.exists():
        print(f"  [WARN] No scene.json in {run_dir.name}")
        return None
    
    with open(scene_json, 'r') as f:
        scene = json.load(f)
    
    aspect_ratio = find_aspect_ratio_heuristic(scene)
    friction = find_friction_heuristic(scene)
    noise = find_noise_heuristic(scene)
    
    if aspect_ratio is None:
        print(f"  [WARN] Could not extract aspect ratio from {run_dir.name}")
        return None
    
    # Load profile.csv for KE data
    profile_csv = run_dir / "profile.csv"
    if not profile_csv.exists():
        print(f"  [WARN] No profile.csv in {run_dir.name}")
        return None
    
    try:
        df = pd.read_csv(profile_csv)
        if 'KE' not in df.columns or 'frame' not in df.columns:
            print(f"  [WARN] Missing KE or frame columns in {run_dir.name}")
            return None
        
        frames = df['frame'].to_numpy()
        ke = df['KE'].to_numpy()
        
        return {
            'run_dir': str(run_dir),
            'aspect_ratio': aspect_ratio,
            'friction': friction if friction is not None else np.nan,
            'noise': noise if noise is not None else np.nan,
            'frames': frames,
            'ke': ke,
        }
    except Exception as e:
        print(f"  [ERROR] Failed to load {profile_csv}: {e}")
        return None

def compute_ke_statistics(ke, frames):
    """Compute KE statistics: mean, std, growth rate."""
    stats = {}
    
    # Mean and std of latter half
    half_idx = len(ke) // 2
    stats['mean_ke'] = np.mean(ke[half_idx:])
    stats['std_ke'] = np.std(ke[half_idx:])
    stats['median_ke'] = np.median(ke[half_idx:])
    
    # Growth rate: fit log(KE) vs frame in middle 60%
    start_idx = int(0.3 * len(ke))
    end_idx = int(0.9 * len(ke))
    ke_fit = ke[start_idx:end_idx]
    frames_fit = frames[start_idx:end_idx]
    
    valid = ke_fit > 0
    if np.sum(valid) > 10:
        try:
            log_ke = np.log(ke_fit[valid])
            t_fit = frames_fit[valid]
            coeffs = np.polyfit(t_fit, log_ke, 1)
            stats['growth_rate'] = coeffs[0]
        except:
            stats['growth_rate'] = np.nan
    else:
        stats['growth_rate'] = np.nan
    
    return stats

def make_summary_table(results, outdir):
    """Create summary table with all statistics."""
    rows = []
    for res in results:
        stats = compute_ke_statistics(res['ke'], res['frames'])
        rows.append({
            'aspect_ratio': res['aspect_ratio'],
            'friction': res['friction'],
            'noise': res['noise'],
            'mean_ke': stats['mean_ke'],
            'std_ke': stats['std_ke'],
            'median_ke': stats['median_ke'],
            'growth_rate': stats['growth_rate'],
            'run_dir': res['run_dir'],
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['aspect_ratio', 'friction', 'noise'])
    
    outfile = Path(outdir) / "summary_table.csv"
    df.to_csv(outfile, index=False)
    print(f"  Saved summary table: {outfile}")
    
    return df

def plot_ke_by_aspect_ratio(results, outdir):
    """Plot KE time traces grouped by aspect ratio."""
    plt.figure(figsize=(12, 8))
    
    # Group by aspect ratio
    aspect_ratios = sorted(set(r['aspect_ratio'] for r in results))
    colors = plt.cm.viridis(np.linspace(0, 1, len(aspect_ratios)))
    
    for ar, color in zip(aspect_ratios, colors):
        runs = [r for r in results if r['aspect_ratio'] == ar]
        for r in runs:
            label = f"L/D={ar:.0f}" if r == runs[0] else None
            plt.semilogy(r['frames'], r['ke'], color=color, alpha=0.7, label=label)
    
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Kinetic Energy (J)', fontsize=12)
    plt.title('KE Time Traces by Aspect Ratio', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    outfile = Path(outdir) / "ke_traces_by_aspect_ratio.png"
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved plot: {outfile}")

def plot_ke_statistics_vs_aspect_ratio(summary_df, outdir):
    """Plot mean KE and growth rate vs aspect ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by friction if multiple values
    frictions = sorted(summary_df['friction'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(frictions)))
    
    for friction, color in zip(frictions, colors):
        df_fric = summary_df[summary_df['friction'] == friction]
        df_fric = df_fric.sort_values('aspect_ratio')
        
        # Mean KE plot
        axes[0].plot(df_fric['aspect_ratio'], df_fric['mean_ke'], 
                     marker='o', color=color, label=f'μ={friction:.2f}')
        
        # Growth rate plot
        axes[1].plot(df_fric['aspect_ratio'], df_fric['growth_rate'], 
                     marker='s', color=color, label=f'μ={friction:.2f}')
    
    axes[0].set_xlabel('Aspect Ratio (L/D)', fontsize=12)
    axes[0].set_ylabel('Mean KE (J)', fontsize=12)
    axes[0].set_title('Mean Kinetic Energy vs Aspect Ratio', fontsize=14)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel('Aspect Ratio (L/D)', fontsize=12)
    axes[1].set_ylabel('Growth Rate (1/frame)', fontsize=12)
    axes[1].set_title('Energy Growth Rate vs Aspect Ratio', fontsize=14)
    axes[1].set_xscale('log')
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    outfile = Path(outdir) / "statistics_vs_aspect_ratio.png"
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved plot: {outfile}")

def plot_contact_count_vs_aspect_ratio(results, outdir):
    """Plot average contact count vs aspect ratio."""
    contact_data = []
    
    for res in results:
        run_dir = Path(res['run_dir'])
        profile_csv = run_dir / "profile.csv"
        
        try:
            df = pd.read_csv(profile_csv)
            if 'n_contacts' in df.columns:
                mean_contacts = df['n_contacts'].iloc[len(df)//2:].mean()
                contact_data.append({
                    'aspect_ratio': res['aspect_ratio'],
                    'friction': res['friction'],
                    'mean_contacts': mean_contacts
                })
        except:
            pass
    
    if not contact_data:
        print("  [WARN] No contact count data found, skipping contact plot")
        return
    
    df_contacts = pd.DataFrame(contact_data)
    
    plt.figure(figsize=(10, 6))
    
    frictions = sorted(df_contacts['friction'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(frictions)))
    
    for friction, color in zip(frictions, colors):
        df_fric = df_contacts[df_contacts['friction'] == friction]
        df_fric = df_fric.sort_values('aspect_ratio')
        
        plt.plot(df_fric['aspect_ratio'], df_fric['mean_contacts'], 
                marker='o', color=color, label=f'μ={friction:.2f}')
    
    plt.xlabel('Aspect Ratio (L/D)', fontsize=12)
    plt.ylabel('Mean Contact Count', fontsize=12)
    plt.title('Average Contact Count vs Aspect Ratio', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    outfile = Path(outdir) / "contacts_vs_aspect_ratio.png"
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved plot: {outfile}")

def main():
    parser = argparse.ArgumentParser(description="Analyze aspect ratio parametric sweep results.")
    parser.add_argument('--job-name', type=str, required=True, help='Job name used in submission.')
    parser.add_argument('--make-plots', action='store_true', help='Generate plots.')
    parser.add_argument('--outdir', type=str, default='analysis_aspect_ratio', help='Output directory for analysis.')
    args = parser.parse_args()
    
    # Find run directories
    runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs") / args.job_name
    if not runs_root.exists():
        print(f"ERROR: Runs directory not found: {runs_root}")
        sys.exit(1)
    
    # Try to load run_dirs.txt first
    run_dirs_file = runs_root / "run_dirs.txt"
    if run_dirs_file.exists():
        print(f"Loading run directories from: {run_dirs_file}")
        with open(run_dirs_file, 'r') as f:
            run_dirs = [Path(line.strip()) for line in f if line.strip()]
    else:
        print(f"Scanning for run directories in: {runs_root}")
        run_dirs = sorted([d for d in runs_root.iterdir() if d.is_dir() and '_RUN_aspect_' in d.name])
    
    if not run_dirs:
        print(f"ERROR: No run directories found in {runs_root}")
        sys.exit(1)
    
    print(f"Found {len(run_dirs)} run directories")
    print("=" * 80)
    
    # Analyze each run
    print("Analyzing runs...")
    results = []
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"[{i}/{len(run_dirs)}] {run_dir.name}")
        res = analyze_run(run_dir)
        if res is not None:
            results.append(res)
    
    if not results:
        print("ERROR: No valid results found")
        sys.exit(1)
    
    print(f"\nSuccessfully analyzed {len(results)}/{len(run_dirs)} runs")
    print("=" * 80)
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {outdir}")
    
    # Generate summary table
    print("\nGenerating summary table...")
    summary_df = make_summary_table(results, outdir)
    
    # Generate plots if requested
    if args.make_plots:
        print("\nGenerating plots...")
        plot_ke_by_aspect_ratio(results, outdir)
        plot_ke_statistics_vs_aspect_ratio(summary_df, outdir)
        plot_contact_count_vs_aspect_ratio(results, outdir)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files in: {outdir}")
    print(f"  - summary_table.csv")
    if args.make_plots:
        print(f"  - ke_traces_by_aspect_ratio.png")
        print(f"  - statistics_vs_aspect_ratio.png")
        print(f"  - contacts_vs_aspect_ratio.png")
    print()

if __name__ == "__main__":
    main()
