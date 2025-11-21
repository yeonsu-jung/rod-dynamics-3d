#!/usr/bin/env python3
"""
post_analyze_soft_contact.py

Analyze results from soft contact parametric sweep: Effects of friction coefficient (mu) and noise amplitude on KE evolution.

Scans run directories, extracts KE data from profile.csv, computes statistics, generates comprehensive plots.

Usage:
    python post_analyze_soft_contact.py --job-name soft_contact_sweep --make-plots --outdir analysis_soft_contact
"""

import argparse
from pathlib import Path

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
    
    # Load COM data if available
    com_csv = run_dir / "com.csv"
    if com_csv.exists():
        try:
            com_df = pd.read_csv(com_csv)
            if 'x' in com_df.columns and 'y' in com_df.columns and 'z' in com_df.columns:
                # Calculate displacement from initial position
                x0, y0, z0 = com_df['x'].iloc[0], com_df['y'].iloc[0], com_df['z'].iloc[0]
                dx = com_df['x'] - x0
                dy = com_df['y'] - y0
                dz = com_df['z'] - z0
                result['com_displacement'] = np.sqrt(dx**2 + dy**2 + dz**2).values
        except Exception as e:
            pass  # COM data is optional
    
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

def plot_com_displacement(results, outdir):
    """Plot COM displacement over time grouped by friction and noise."""
    results_with_com = [r for r in results if 'com_displacement' in r]
    
    if not results_with_com:
        print("  [WARN] No COM displacement data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: By friction
    ax = axes[0]
    frictions = sorted(set(r['friction'] for r in results_with_com))
    colors_f = plt.cm.plasma(np.linspace(0, 1, len(frictions)))
    
    for friction, color in zip(frictions, colors_f):
        runs = [r for r in results_with_com if abs(r['friction'] - friction) < 1e-6]
        for r in runs:
            label = f'μ={friction:.2f}' if r == runs[0] else None
            ax.plot(r['frames'], r['com_displacement'], color=color, alpha=0.5, linewidth=1, label=label)
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('COM Displacement', fontsize=12)
    ax.set_title('COM Displacement by Friction', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Right: By noise
    ax = axes[1]
    noises = sorted(set(r['noise'] for r in results_with_com))
    colors_n = plt.cm.coolwarm(np.linspace(0, 1, len(noises)))
    
    for noise, color in zip(noises, colors_n):
        runs = [r for r in results_with_com if abs(r['noise'] - noise) < 1e-10]
        for r in runs:
            label = f'σ={noise:.1e}' if r == runs[0] else None
            ax.plot(r['frames'], r['com_displacement'], color=color, alpha=0.5, linewidth=1, label=label)
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('COM Displacement', fontsize=12)
    ax.set_title('COM Displacement by Noise', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / 'com_displacement.png', dpi=150)
    plt.close()
    print(f"Saved plot: {outdir / 'com_displacement.png'}")

def plot_aggregate_statistics(results, outdir):
    """Plot aggregate KE and COM statistics with mean ± std bands."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top left: KE by friction (mean ± std)
    ax = axes[0, 0]
    frictions = sorted(set(r['friction'] for r in results))
    colors_f = plt.cm.plasma(np.linspace(0, 1, len(frictions)))
    
    for friction, color in zip(frictions, colors_f):
        runs = [r for r in results if abs(r['friction'] - friction) < 1e-6]
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
    ax.set_ylabel('Kinetic Energy (J)', fontsize=11)
    ax.set_title('KE by Friction (mean ± std)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Top right: KE by noise (mean ± std)
    ax = axes[0, 1]
    noises = sorted(set(r['noise'] for r in results))
    colors_n = plt.cm.coolwarm(np.linspace(0, 1, len(noises)))
    
    for noise, color in zip(noises, colors_n):
        runs = [r for r in results if abs(r['noise'] - noise) < 1e-10]
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
    ax.set_ylabel('Kinetic Energy (J)', fontsize=11)
    ax.set_title('KE by Noise (mean ± std)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Bottom left: COM displacement by friction (if available)
    ax = axes[1, 0]
    results_with_com = [r for r in results if 'com_displacement' in r]
    if results_with_com:
        for friction, color in zip(frictions, colors_f):
            runs = [r for r in results_with_com if abs(r['friction'] - friction) < 1e-6]
            if not runs:
                continue
            
            min_len = min(len(r['com_displacement']) for r in runs)
            frames = runs[0]['frames'][:min_len]
            com_stack = np.array([r['com_displacement'][:min_len] for r in runs])
            com_mean = np.mean(com_stack, axis=0)
            com_std = np.std(com_stack, axis=0)
            
            ax.plot(frames, com_mean, color=color, linewidth=2, label=f'μ={friction:.2f}')
            ax.fill_between(frames, com_mean - com_std, com_mean + com_std, color=color, alpha=0.2)
        
        ax.set_xlabel('Frame', fontsize=11)
        ax.set_ylabel('COM Displacement', fontsize=11)
        ax.set_title('COM Displacement by Friction (mean ± std)', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No COM data', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    
    # Bottom right: COM displacement by noise (if available)
    ax = axes[1, 1]
    if results_with_com:
        for noise, color in zip(noises, colors_n):
            runs = [r for r in results_with_com if abs(r['noise'] - noise) < 1e-10]
            if not runs:
                continue
            
            min_len = min(len(r['com_displacement']) for r in runs)
            frames = runs[0]['frames'][:min_len]
            com_stack = np.array([r['com_displacement'][:min_len] for r in runs])
            com_mean = np.mean(com_stack, axis=0)
            com_std = np.std(com_stack, axis=0)
            
            ax.plot(frames, com_mean, color=color, linewidth=2, label=f'σ={noise:.1e}')
            ax.fill_between(frames, com_mean - com_std, com_mean + com_std, color=color, alpha=0.2)
        
        ax.set_xlabel('Frame', fontsize=11)
        ax.set_ylabel('COM Displacement', fontsize=11)
        ax.set_title('COM Displacement by Noise (mean ± std)', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No COM data', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(outdir / 'aggregate_statistics.png', dpi=150)
    plt.close()
    print(f"Saved plot: {outdir / 'aggregate_statistics.png'}")

def submit_analysis_job(args):
    """Submit this analysis script as a SLURM job."""
    import subprocess
    import os
    
    # Build command to run this script
    cmd_parts = [
        'python3', __file__,
        '--job-name', args.job_name,
        '--outdir', args.outdir,
        '--runs-root', args.runs_root,
    ]
    if args.make_plots:
        cmd_parts.append('--make-plots')
    
    analysis_cmd = ' '.join(cmd_parts)
    
    # Create SLURM script
    sbatch_script = f"""#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 0-2:00
#SBATCH -p seas_compute
#SBATCH --mem=16000
#SBATCH -o analysis_{args.job_name}_%j.out
#SBATCH -e analysis_{args.job_name}_%j.err
#SBATCH --job-name=analyze_{args.job_name}

set -euo pipefail
module load python
mamba activate mujoco-env

echo "======================================"
echo "Post-Analysis Job"
echo "======================================"
echo "Job name: {args.job_name}"
echo "Output directory: {args.outdir}"
echo "Make plots: {args.make_plots}"
echo "PWD: $(pwd)"
echo "======================================"
echo ""

{analysis_cmd}

echo ""
echo "======================================"
echo "Analysis complete"
echo "======================================"
"""
    
    # Write to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(sbatch_script)
        sbatch_file = f.name
    
    try:
        # Make executable
        os.chmod(sbatch_file, 0o755)
        
        # Submit job
        print("Submitting analysis job to SLURM...")
        print(f"Command: {analysis_cmd}")
        result = subprocess.run(['sbatch', sbatch_file], capture_output=True, text=True, check=True)
        print(result.stdout)
        print(f"Job submitted successfully!")
        print(f"Monitor with: squeue -u $USER")
    finally:
        # Clean up temporary file
        os.unlink(sbatch_file)

def main():
    parser = argparse.ArgumentParser(description="Analyze soft contact parametric sweep results.")
    parser.add_argument('--job-name', type=str, required=True, help='Job name used in submission.')
    parser.add_argument('--runs-root', type=str, 
                       default='/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs',
                       help='Root directory for runs.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for plots and analysis.')
    parser.add_argument('--make-plots', action='store_true', help='Generate plots.')
    parser.add_argument('--submit', action='store_true', help='Submit analysis as SLURM job instead of running locally.')
    args = parser.parse_args()
    
    # If --submit is specified, create and submit SLURM job
    if args.submit:
        submit_analysis_job(args)
        return
    
    # Import heavy dependencies only when actually running analysis
    import os
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # Make these available globally for the analysis functions
    globals()['os'] = os
    globals()['json'] = json
    globals()['np'] = np
    globals()['pd'] = pd
    globals()['plt'] = plt
    globals()['curve_fit'] = curve_fit

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
        plot_com_displacement(data, outdir)
        plot_aggregate_statistics(data, outdir)
        
        print()
        print("=" * 80)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {outdir}")
        print("  - summary_table.csv")
        print("  - heatmap_ke_friction_vs_noise.png")
        print("  - heatmap_growth_friction_vs_noise.png")
        print("  - ke_by_friction.png")
        print("  - ke_by_noise.png")
        print("  - growth_rate_vs_noise.png")
        print("  - com_displacement.png")
        print("  - aggregate_statistics.png")
        print("=" * 80)
    else:
        print()
        print("Skipping plots (use --make-plots to generate)")

if __name__ == "__main__":
    main()
