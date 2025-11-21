#!/usr/bin/env python3
"""
Analysis script for soft contact parametric sweep results.
Analyzes kinetic energy, entanglement metrics, and contact statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Parameters matching the sweep
FRICTION_COEFFS = [0.0, 0.05, 0.1, 0.2, 0.4]
NOISE_AMPLITUDES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# Paths
REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/soft_first")
PLOT_DIR = OUTPUT_DIR / "plots"

def load_kinetic_energy(run_id):
    """Load kinetic energy data from a simulation run."""
    ke_file = OUTPUT_DIR / f"{run_id}_ke.csv"
    
    if not ke_file.exists():
        print(f"Warning: KE file not found: {ke_file}")
        return None, None
    
    try:
        data = np.loadtxt(ke_file, delimiter=',', skiprows=1)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        time = data[:, 0]
        ke_total = data[:, 1]
        return time, ke_total
    except Exception as e:
        print(f"Error loading {ke_file}: {e}")
        return None, None

def load_positions(run_id):
    """Load position data from a simulation run."""
    pos_file = OUTPUT_DIR / f"{run_id}_pos.csv"
    
    if not pos_file.exists():
        return None
    
    try:
        data = np.loadtxt(pos_file, delimiter=',', skiprows=1)
        return data
    except Exception as e:
        print(f"Error loading {pos_file}: {e}")
        return None

def compute_mean_ke_late(time, ke, start_frac=0.5):
    """Compute mean KE from the latter half of the simulation."""
    if time is None or ke is None:
        return np.nan
    
    start_idx = int(len(time) * start_frac)
    return np.mean(ke[start_idx:])

def compute_ke_growth_rate(time, ke, start_frac=0.3, end_frac=0.9):
    """Compute linear growth rate of log(KE) in the middle region."""
    if time is None or ke is None:
        return np.nan
    
    start_idx = int(len(time) * start_frac)
    end_idx = int(len(time) * end_frac)
    
    t_fit = time[start_idx:end_idx]
    ke_fit = ke[start_idx:end_idx]
    
    # Filter out zeros/negatives
    valid = ke_fit > 0
    if np.sum(valid) < 10:
        return np.nan
    
    log_ke = np.log(ke_fit[valid])
    t_fit = t_fit[valid]
    
    # Linear fit in log space
    try:
        coeffs = np.polyfit(t_fit, log_ke, 1)
        return coeffs[0]  # slope = growth rate
    except:
        return np.nan

def analyze_all_runs():
    """Analyze all runs and create summary data structures."""
    
    results = {
        'friction': [],
        'noise': [],
        'mean_ke': [],
        'ke_growth_rate': [],
        'run_id': []
    }
    
    for friction in FRICTION_COEFFS:
        for noise in NOISE_AMPLITUDES:
            run_id = f"mu{friction:.2f}_noise{noise:.1e}"
            
            time, ke = load_kinetic_energy(run_id)
            
            mean_ke = compute_mean_ke_late(time, ke)
            growth_rate = compute_ke_growth_rate(time, ke)
            
            results['friction'].append(friction)
            results['noise'].append(noise)
            results['mean_ke'].append(mean_ke)
            results['ke_growth_rate'].append(growth_rate)
            results['run_id'].append(run_id)
            
            print(f"{run_id}: mean_KE={mean_ke:.2e}, growth_rate={growth_rate:.4e}")
    
    # Convert to numpy arrays
    for key in results:
        if key != 'run_id':
            results[key] = np.array(results[key])
    
    return results

def plot_ke_heatmap(results):
    """Create heatmap of mean KE vs friction and noise."""
    
    n_friction = len(FRICTION_COEFFS)
    n_noise = len(NOISE_AMPLITUDES)
    
    ke_matrix = np.full((n_noise, n_friction), np.nan)
    
    for i, noise in enumerate(NOISE_AMPLITUDES):
        for j, friction in enumerate(FRICTION_COEFFS):
            mask = (results['friction'] == friction) & (results['noise'] == noise)
            if np.any(mask):
                ke_matrix[i, j] = results['mean_ke'][mask][0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(np.log10(ke_matrix), aspect='auto', cmap='viridis', origin='lower')
    
    ax.set_xticks(range(n_friction))
    ax.set_xticklabels([f"{f:.2f}" for f in FRICTION_COEFFS])
    ax.set_xlabel('Friction Coefficient μ', fontsize=12)
    
    ax.set_yticks(range(n_noise))
    ax.set_yticklabels([f"{n:.0e}" for n in NOISE_AMPLITUDES])
    ax.set_ylabel('Noise Amplitude σ', fontsize=12)
    
    ax.set_title('Mean Kinetic Energy (log10)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(KE)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ke_heatmap.png', dpi=150)
    print(f"Saved: {PLOT_DIR / 'ke_heatmap.png'}")
    plt.close()

def plot_growth_rate_heatmap(results):
    """Create heatmap of KE growth rate vs friction and noise."""
    
    n_friction = len(FRICTION_COEFFS)
    n_noise = len(NOISE_AMPLITUDES)
    
    rate_matrix = np.full((n_noise, n_friction), np.nan)
    
    for i, noise in enumerate(NOISE_AMPLITUDES):
        for j, friction in enumerate(FRICTION_COEFFS):
            mask = (results['friction'] == friction) & (results['noise'] == noise)
            if np.any(mask):
                rate_matrix[i, j] = results['ke_growth_rate'][mask][0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(rate_matrix, aspect='auto', cmap='RdYlGn_r', origin='lower')
    
    ax.set_xticks(range(n_friction))
    ax.set_xticklabels([f"{f:.2f}" for f in FRICTION_COEFFS])
    ax.set_xlabel('Friction Coefficient μ', fontsize=12)
    
    ax.set_yticks(range(n_noise))
    ax.set_yticklabels([f"{n:.0e}" for n in NOISE_AMPLITUDES])
    ax.set_ylabel('Noise Amplitude σ', fontsize=12)
    
    ax.set_title('KE Growth Rate (1/time)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Growth Rate', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'growth_rate_heatmap.png', dpi=150)
    print(f"Saved: {PLOT_DIR / 'growth_rate_heatmap.png'}")
    plt.close()

def plot_ke_traces_by_friction():
    """Plot KE traces grouped by friction coefficient."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, friction in enumerate(FRICTION_COEFFS):
        ax = axes[i]
        
        for noise in NOISE_AMPLITUDES:
            run_id = f"mu{friction:.2f}_noise{noise:.1e}"
            time, ke = load_kinetic_energy(run_id)
            
            if time is not None:
                ax.semilogy(time, ke, label=f'σ={noise:.0e}', alpha=0.7)
        
        ax.set_title(f'μ = {friction:.2f}', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Kinetic Energy')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplot
    if len(FRICTION_COEFFS) < len(axes):
        axes[-1].axis('off')
    
    plt.suptitle('KE Evolution: Soft Contact Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ke_traces_by_friction.png', dpi=150)
    print(f"Saved: {PLOT_DIR / 'ke_traces_by_friction.png'}")
    plt.close()

def plot_ke_traces_by_noise():
    """Plot KE traces grouped by noise amplitude."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, noise in enumerate(NOISE_AMPLITUDES):
        ax = axes[i]
        
        for friction in FRICTION_COEFFS:
            run_id = f"mu{friction:.2f}_noise{noise:.1e}"
            time, ke = load_kinetic_energy(run_id)
            
            if time is not None:
                ax.semilogy(time, ke, label=f'μ={friction:.2f}', alpha=0.7)
        
        ax.set_title(f'σ = {noise:.0e}', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Kinetic Energy')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('KE Evolution: Soft Contact Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ke_traces_by_noise.png', dpi=150)
    print(f"Saved: {PLOT_DIR / 'ke_traces_by_noise.png'}")
    plt.close()

def main():
    # Create plot directory
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Analyzing soft contact parametric sweep results...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 80)
    
    # Analyze all runs
    results = analyze_all_runs()
    
    print("\n" + "-" * 80)
    print("Creating plots...")
    
    # Generate plots
    plot_ke_heatmap(results)
    plot_growth_rate_heatmap(results)
    plot_ke_traces_by_friction()
    plot_ke_traces_by_noise()
    
    # Save results to file
    results_file = OUTPUT_DIR / 'analysis_results.npz'
    np.savez(results_file, **results)
    print(f"\nSaved results to: {results_file}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Plots saved to: {PLOT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
