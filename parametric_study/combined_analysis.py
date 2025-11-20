#!/usr/bin/env python3
"""
Combined analysis: Rigid motion decomposition + pairwise metrics

Analyzes rod ensemble dynamics from simulation trajectories:
1. Rigid motion decomposition (global vs deformational)
2. Pairwise structural metrics
3. Temporal correlations
4. Energy partitioning

Usage:
  python combined_analysis.py <per_rod_csv> [--output-dir DIR]
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Import from our modules
from rigid_motion_decomposition import (
    load_trajectory_csv,
    decompose_velocities,
    compute_kinetic_energies,
    analyze_rigid_motion_decomposition
)
from pairwise_metrics import (
    compute_pairwise_distances,
    compute_pairwise_angles,
    compute_all_metrics
)


def combined_analysis(csv_path, output_dir=None):
    """
    Perform combined rigid decomposition + pairwise analysis.
    
    Args:
        csv_path: Path to per-rod trajectory CSV
        output_dir: Directory to save all results
    
    Returns:
        Dictionary with all analysis results
    """
    # Setup output
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_dir = f"combined_analysis_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"COMBINED ANALYSIS: {os.path.basename(csv_path)}")
    print(f"{'='*60}\n")
    
    # Load trajectory
    print("Loading trajectory...")
    positions_time, velocities_time, ang_velocities_time, orientations_time, frames, masses = load_trajectory_csv(csv_path)
    N_timesteps, N_rods, _ = positions_time.shape
    print(f"  {N_timesteps} timesteps, {N_rods} rods\n")
    
    # ============================================================
    # 1. RIGID MOTION DECOMPOSITION
    # ============================================================
    print("1. Rigid Motion Decomposition")
    print("-" * 40)
    
    rigid_results = {
        'frames': frames,
        'omega': np.zeros((N_timesteps, 3)),
        'v0': np.zeros((N_timesteps, 3)),
        'KE_total': np.zeros(N_timesteps),
        'KE_global': np.zeros(N_timesteps),
        'KE_def': np.zeros(N_timesteps),
        'residual_rms': np.zeros(N_timesteps),
    }
    
    for t_idx in range(N_timesteps):
        v_rigid, v_def, omega, v0, residual = decompose_velocities(
            positions_time[t_idx], velocities_time[t_idx], masses
        )
        
        rigid_results['omega'][t_idx] = omega
        rigid_results['v0'][t_idx] = v0
        rigid_results['residual_rms'][t_idx] = residual
        rigid_results['KE_total'][t_idx] = compute_kinetic_energies(velocities_time[t_idx], masses)
        rigid_results['KE_global'][t_idx] = compute_kinetic_energies(v_rigid, masses)
        rigid_results['KE_def'][t_idx] = compute_kinetic_energies(v_def, masses)
    
    mean_global_frac = rigid_results['KE_global'].mean() / (rigid_results['KE_total'].mean() + 1e-10)
    print(f"  Mean global KE fraction: {mean_global_frac:.3f}")
    print(f"  Mean deformational KE fraction: {1-mean_global_frac:.3f}")
    print(f"  Mean |ω|: {np.linalg.norm(rigid_results['omega'], axis=1).mean():.4f} rad/s")
    print(f"  Mean |v₀|: {np.linalg.norm(rigid_results['v0'], axis=1).mean():.4f} units/s\n")
    
    # ============================================================
    # 2. PAIRWISE METRICS
    # ============================================================
    print("2. Pairwise Structural Metrics")
    print("-" * 40)
    
    pairwise_results = compute_all_metrics(positions_time, orientations_time)
    
    print(f"  Average pairwise distance: {pairwise_results['avg_distance']:.4f}")
    print(f"  Average pairwise angle: {pairwise_results['avg_angle']:.2f}°")
    print(f"  Orientation autocorr (lag=1): {pairwise_results['avg_orientation_correlation']:.4f}")
    print(f"  Distance autocorr (lag=1): {pairwise_results['avg_distance_correlation']:.4f}\n")
    
    # ============================================================
    # 3. TIME-RESOLVED PAIRWISE DISTANCES
    # ============================================================
    print("3. Computing time-resolved pairwise distances...")
    
    avg_distances = []
    std_distances = []
    min_distances = []
    
    for t_idx in range(N_timesteps):
        dists = compute_pairwise_distances(positions_time[t_idx])
        # Extract upper triangle (unique pairs)
        upper_tri_idx = np.triu_indices(N_rods, k=1)
        dists_unique = dists[upper_tri_idx]
        
        avg_distances.append(dists_unique.mean())
        std_distances.append(dists_unique.std())
        min_distances.append(dists_unique.min())
    
    avg_distances = np.array(avg_distances)
    std_distances = np.array(std_distances)
    min_distances = np.array(min_distances)
    
    print(f"  Mean distance: {avg_distances.mean():.4f} ± {avg_distances.std():.4f}")
    print(f"  Mean min distance: {min_distances.mean():.4f}\n")
    
    # ============================================================
    # 4. TIME-RESOLVED PAIRWISE ANGLES
    # ============================================================
    print("4. Computing time-resolved pairwise angles...")
    
    avg_angles = []
    
    for t_idx in range(N_timesteps):
        angles_mat = compute_pairwise_angles(orientations_time[t_idx])
        upper_tri_idx = np.triu_indices(N_rods, k=1)
        angles_unique = angles_mat[upper_tri_idx]
        avg_angles.append(np.degrees(angles_unique.mean()))
    
    avg_angles = np.array(avg_angles)
    
    print(f"  Mean angle: {avg_angles.mean():.2f}° ± {avg_angles.std():.2f}°\n")
    
    # ============================================================
    # 5. COMBINED VISUALIZATION
    # ============================================================
    print("5. Creating visualizations...")
    
    create_combined_plots(
        frames, rigid_results, avg_distances, std_distances, 
        min_distances, avg_angles, output_dir
    )
    
    # Save summary CSV
    summary_df = pd.DataFrame({
        'frame': frames,
        'KE_total': rigid_results['KE_total'],
        'KE_global': rigid_results['KE_global'],
        'KE_def': rigid_results['KE_def'],
        'omega_mag': np.linalg.norm(rigid_results['omega'], axis=1),
        'v0_mag': np.linalg.norm(rigid_results['v0'], axis=1),
        'avg_pairwise_dist': avg_distances,
        'std_pairwise_dist': std_distances,
        'min_pairwise_dist': min_distances,
        'avg_pairwise_angle': avg_angles,
        'residual_rms': rigid_results['residual_rms']
    })
    
    summary_csv = os.path.join(output_dir, 'combined_analysis.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Summary saved to {summary_csv}")
    
    # Save full results
    results = {
        'rigid': rigid_results,
        'pairwise': pairwise_results,
        'time_series': {
            'avg_distances': avg_distances,
            'std_distances': std_distances,
            'min_distances': min_distances,
            'avg_angles': avg_angles
        },
        'N_rods': N_rods,
        'N_timesteps': N_timesteps
    }
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results in: {output_dir}/")
    print(f"{'='*60}\n")
    
    return results


def create_combined_plots(frames, rigid_results, avg_distances, std_distances, 
                         min_distances, avg_angles, output_dir):
    """Create comprehensive combined analysis plots."""
    
    # Figure 1: Energy decomposition + pairwise distances
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Energy
    ax = axes[0]
    ax.plot(frames, rigid_results['KE_total'], 'k-', label='Total KE', linewidth=2, alpha=0.7)
    ax.plot(frames, rigid_results['KE_global'], 'b-', label='Global KE', linewidth=1.5)
    ax.plot(frames, rigid_results['KE_def'], 'r-', label='Deformational KE', linewidth=1.5)
    ax.set_ylabel('Kinetic Energy')
    ax.set_title('Energy Decomposition')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    # Bottom: Pairwise distances
    ax = axes[1]
    ax.plot(frames, avg_distances, 'purple', label='Mean distance', linewidth=2)
    ax.fill_between(frames, avg_distances - std_distances, avg_distances + std_distances,
                    alpha=0.3, color='purple', label='±1 std')
    ax.plot(frames, min_distances, 'orange', label='Min distance', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance')
    ax.set_title('Pairwise Distances Over Time')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_energy_distance.png'), dpi=150)
    plt.close()
    
    # Figure 2: Angular velocity vs pairwise angles
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Global rotation
    ax = axes[0]
    omega_mag = np.linalg.norm(rigid_results['omega'], axis=1)
    ax.plot(frames, omega_mag, 'g-', linewidth=2)
    ax.set_ylabel('|ω| (rad/s)')
    ax.set_title('Global Angular Velocity')
    ax.grid(alpha=0.3)
    
    # Bottom: Pairwise angles
    ax = axes[1]
    ax.plot(frames, avg_angles, 'brown', linewidth=2)
    ax.axhline(45, color='gray', linestyle='--', alpha=0.5, label='Random (45°)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Average Pairwise Angle Between Rod Axes')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_rotation_angles.png'), dpi=150)
    plt.close()
    
    # Figure 3: Phase space - energy fractions vs structural order
    fig, ax = plt.subplots(figsize=(10, 8))
    
    total_nonzero = rigid_results['KE_total'] > 1e-10
    frac_def = np.zeros_like(rigid_results['KE_total'])
    frac_def[total_nonzero] = rigid_results['KE_def'][total_nonzero] / rigid_results['KE_total'][total_nonzero]
    
    # Normalize distances to [0, 1] for colormap
    dist_norm = (avg_distances - avg_distances.min()) / (avg_distances.max() - avg_distances.min() + 1e-10)
    
    scatter = ax.scatter(frac_def, avg_angles, c=frames, cmap='viridis', 
                        s=30, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax.set_xlabel('Deformational KE Fraction')
    ax.set_ylabel('Average Pairwise Angle (degrees)')
    ax.set_title('Phase Space: Energy vs Structure (colored by time)')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Frame')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_space.png'), dpi=150)
    plt.close()
    
    # Figure 4: 4-panel summary
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Energy fractions pie chart (average)
    ax = fig.add_subplot(gs[0, 0])
    mean_global = rigid_results['KE_global'].mean()
    mean_def = rigid_results['KE_def'].mean()
    ax.pie([mean_global, mean_def], labels=['Global', 'Deformational'],
           autopct='%1.1f%%', colors=['blue', 'red'], startangle=90)
    ax.set_title('Mean Energy Partitioning')
    
    # Panel 2: Distance histogram (final frame)
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(avg_distances, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(avg_distances.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('Average Pairwise Distance')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Mean Distances')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 3: Residual RMS over time
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(frames, rigid_results['residual_rms'], 'orange', linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('RMS Residual')
    ax.set_title('Rigid Motion Fit Quality')
    ax.grid(alpha=0.3)
    
    # Panel 4: Min distance over time (collision proxy)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(frames, min_distances, 'red', linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Minimum Pairwise Distance')
    ax.set_title('Closest Approach (Collision Proxy)')
    ax.grid(alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'summary_4panel.png'), dpi=150)
    plt.close()
    
    print(f"  Plots saved to {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python combined_analysis.py <per_rod_csv> [--output-dir DIR]")
        print("\nExample:")
        print("  python combined_analysis.py ../perrod.csv")
        print("  python combined_analysis.py pairwise_sweep_csvs/pairwise_n6_ar50.csv --output-dir results_n6_ar50")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_dir = None
    
    if '--output-dir' in sys.argv:
        idx = sys.argv.index('--output-dir')
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    results = combined_analysis(csv_path, output_dir)
