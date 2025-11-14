#!/usr/bin/env python3
"""
Compute enhanced pairwise properties for rod ensembles:
1. Average pairwise distance d_ij between rod centers
2. Average pairwise angle θ_ij between rod axes
3. Average temporal correlation of orientation autocorrelations
4. Average temporal correlation of pairwise distances
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import os


def rotation_matrix_from_quaternion(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])


def quaternion_to_axis(q):
    """Extract orientation axis (body z-axis) from quaternion."""
    R = rotation_matrix_from_quaternion(q)
    return R[:, 2]  # Third column = local z in world frame


def angle_between_vectors(u, v):
    """Compute angle in radians between unit vectors u and v."""
    cos_theta = np.clip(np.dot(u, v), -1.0, 1.0)
    return np.arccos(np.abs(cos_theta))  # Use abs to get acute angle


def compute_pairwise_distances(positions):
    """
    Compute pairwise distances between all rod centers.
    
    Args:
        positions: (N_rods, 3) array of rod centers
    
    Returns:
        (N_rods, N_rods) distance matrix
    """
    return squareform(pdist(positions))


def compute_pairwise_angles(orientations):
    """
    Compute pairwise angles between all rod axes.
    
    Args:
        orientations: (N_rods, 3) array of rod orientation unit vectors
    
    Returns:
        (N_rods, N_rods) angle matrix in radians
    """
    N = len(orientations)
    angles = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            theta = angle_between_vectors(orientations[i], orientations[j])
            angles[i, j] = theta
            angles[j, i] = theta
    return angles


def compute_autocorrelation(signal, max_lag=None):
    """
    Compute normalized autocorrelation of a 1D signal.
    
    Args:
        signal: 1D array
        max_lag: Maximum lag to compute (default: len(signal)//2)
    
    Returns:
        Autocorrelation array of length max_lag+1
    """
    signal = np.array(signal)
    signal = signal - signal.mean()  # Remove mean
    
    if max_lag is None:
        max_lag = len(signal) // 2
    
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
    autocorr = autocorr[:max_lag+1]
    autocorr = autocorr / autocorr[0]  # Normalize by variance
    
    return autocorr


def compute_orientation_autocorrelations(orientations_time):
    """
    Compute autocorrelation of cos(θ(t, t+δt)) for each rod.
    
    Args:
        orientations_time: (N_timesteps, N_rods, 3) array of rod axes over time
    
    Returns:
        (N_rods, N_lags) array of autocorrelation functions
    """
    N_timesteps, N_rods, _ = orientations_time.shape
    max_lag = N_timesteps // 2
    
    autocorrs = np.zeros((N_rods, max_lag + 1))
    
    for i in range(N_rods):
        # Compute cos(θ(t, t+δt)) for all δt
        cos_series = []
        for t in range(N_timesteps - 1):
            u_t = orientations_time[t, i]
            u_tp1 = orientations_time[t+1, i]
            cos_theta = np.dot(u_t, u_tp1)
            cos_series.append(cos_theta)
        
        cos_series = np.array(cos_series)
        autocorrs[i] = compute_autocorrelation(cos_series, max_lag)
    
    return autocorrs


def compute_distance_autocorrelations(positions_time):
    """
    Compute autocorrelation of pairwise distances d_ij(t).
    
    Args:
        positions_time: (N_timesteps, N_rods, 3) array of rod centers over time
    
    Returns:
        (N_pairs, N_lags) array of distance autocorrelations
    """
    N_timesteps, N_rods, _ = positions_time.shape
    max_lag = N_timesteps // 2
    
    # Compute time series of pairwise distances
    N_pairs = N_rods * (N_rods - 1) // 2
    distance_series = np.zeros((N_timesteps, N_pairs))
    
    for t in range(N_timesteps):
        dists = pdist(positions_time[t])
        distance_series[t] = dists
    
    # Compute autocorrelation for each pair
    autocorrs = np.zeros((N_pairs, max_lag + 1))
    for pair_idx in range(N_pairs):
        autocorrs[pair_idx] = compute_autocorrelation(distance_series[:, pair_idx], max_lag)
    
    return autocorrs


def load_trajectory_csv(csv_path):
    """
    Load per-rod trajectory CSV.
    
    Expected columns: frame, rod, px, py, pz, vx, vy, vz, wx, wy, wz, qw, qx, qy, qz, KE_lin, KE_rot, KE_total
    
    Returns:
        positions_time: (N_timesteps, N_rods, 3)
        orientations_time: (N_timesteps, N_rods, 3) - unit axes
        times: (N_timesteps,) - frame numbers
    """
    df = pd.read_csv(csv_path)
    
    # Get unique frames and rod IDs
    frames = df['frame'].unique()
    rod_ids = df['rod'].unique()
    N_timesteps = len(frames)
    N_rods = len(rod_ids)
    
    positions_time = np.zeros((N_timesteps, N_rods, 3))
    orientations_time = np.zeros((N_timesteps, N_rods, 3))
    
    for t_idx, frame in enumerate(frames):
        df_t = df[df['frame'] == frame].sort_values('rod')
        
        positions_time[t_idx] = df_t[['px', 'py', 'pz']].values
        
        # Convert quaternions to orientation axes
        for rod_idx, row in enumerate(df_t.itertuples()):
            q = [row.qw, row.qx, row.qy, row.qz]
            axis = quaternion_to_axis(q)
            orientations_time[t_idx, rod_idx] = axis
    
    return positions_time, orientations_time, frames


def compute_all_metrics(positions_time, orientations_time) -> Dict[str, float]:
    """
    Compute all four enhanced metrics.
    
    Returns:
        Dictionary with:
        - avg_distance: Mean pairwise distance <d_ij>
        - avg_angle: Mean pairwise angle <θ_ij> in degrees
        - avg_orientation_correlation: Mean of orientation autocorrelation at lag=1
        - avg_distance_correlation: Mean of distance autocorrelation at lag=1
    """
    N_timesteps, N_rods, _ = positions_time.shape
    
    # 1. Average pairwise distance (time-averaged)
    avg_dists_per_time = []
    for t in range(N_timesteps):
        dists = pdist(positions_time[t])
        avg_dists_per_time.append(dists.mean())
    avg_distance = np.mean(avg_dists_per_time)
    
    # 2. Average pairwise angle (time-averaged)
    avg_angles_per_time = []
    for t in range(N_timesteps):
        angles_mat = compute_pairwise_angles(orientations_time[t])
        # Extract upper triangle (unique pairs)
        angles = angles_mat[np.triu_indices(N_rods, k=1)]
        avg_angles_per_time.append(np.degrees(angles.mean()))
    avg_angle = np.mean(avg_angles_per_time)
    
    # 3. Average temporal correlation of orientations
    orientation_autocorrs = compute_orientation_autocorrelations(orientations_time)
    # Use lag=1 correlation as measure of temporal coherence
    avg_orientation_correlation = orientation_autocorrs[:, 1].mean()
    
    # 4. Average temporal correlation of distances
    distance_autocorrs = compute_distance_autocorrelations(positions_time)
    avg_distance_correlation = distance_autocorrs[:, 1].mean()
    
    return {
        'avg_distance': avg_distance,
        'avg_angle': avg_angle,
        'avg_orientation_correlation': avg_orientation_correlation,
        'avg_distance_correlation': avg_distance_correlation,
        'N_rods': N_rods,
        'N_timesteps': N_timesteps
    }


def analyze_parametric_sweep(csv_dir, n_values, ar_values, output_dir='pairwise_analysis'):
    """
    Analyze a parametric sweep over n and AR.
    
    Args:
        csv_dir: Directory containing CSV files named like "n{n}_ar{ar}.csv"
        n_values: List of n (number of rods) values
        ar_values: List of AR (aspect ratio) values
        output_dir: Directory to save plots and results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for n in n_values:
        for ar in ar_values:
            csv_file = os.path.join(csv_dir, f"n{n}_ar{ar}.csv")
            
            if not os.path.exists(csv_file):
                print(f"Warning: {csv_file} not found, skipping...")
                continue
            
            print(f"Processing n={n}, AR={ar}...")
            
            positions_time, orientations_time, times = load_trajectory_csv(csv_file)
            metrics = compute_all_metrics(positions_time, orientations_time)
            metrics['n'] = n
            metrics['AR'] = ar
            
            results.append(metrics)
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, 'pairwise_metrics.csv'), index=False)
    
    print(f"\nResults saved to {output_dir}/pairwise_metrics.csv")
    print(df_results)
    
    # Create plots
    plot_metrics_vs_parameters(df_results, output_dir)
    
    return df_results


def plot_metrics_vs_parameters(df, output_dir):
    """Create comprehensive plots of metrics vs n and AR."""
    metrics = ['avg_distance', 'avg_angle', 'avg_orientation_correlation', 'avg_distance_correlation']
    metric_labels = [
        'Average Pairwise Distance',
        'Average Pairwise Angle (deg)',
        'Orientation Autocorr (lag=1)',
        'Distance Autocorr (lag=1)'
    ]
    
    n_values = sorted(df['n'].unique())
    ar_values = sorted(df['AR'].unique())
    
    # Plot each metric vs n (separate lines for each AR)
    for metric, label in zip(metrics, metric_labels):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for ar in ar_values:
            df_ar = df[df['AR'] == ar].sort_values('n')
            ax.plot(df_ar['n'], df_ar[metric], marker='o', label=f'AR={ar}')
        
        ax.set_xlabel('Number of Rods (n)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Number of Rods')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_vs_n.png'), dpi=150)
        plt.close()
    
    # Plot each metric vs AR (separate lines for each n)
    for metric, label in zip(metrics, metric_labels):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for n in n_values:
            df_n = df[df['n'] == n].sort_values('AR')
            ax.plot(df_n['AR'], df_n[metric], marker='s', label=f'n={n}')
        
        ax.set_xlabel('Aspect Ratio (AR)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Aspect Ratio')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_vs_ar.png'), dpi=150)
        plt.close()
    
    # Create heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        # Create pivot table
        pivot = df.pivot(index='AR', columns='n', values=metric)
        
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xticks(range(len(n_values)))
        ax.set_xticklabels(n_values)
        ax.set_yticks(range(len(ar_values)))
        ax.set_yticklabels(ar_values)
        ax.set_xlabel('Number of Rods (n)')
        ax.set_ylabel('Aspect Ratio (AR)')
        ax.set_title(label)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add text annotations
        for i in range(len(ar_values)):
            for j in range(len(n_values)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                           color='white' if val < pivot.values.mean() else 'black',
                           fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pairwise_metrics.py <csv_file>")
        print("   Or: python pairwise_metrics.py --sweep <csv_dir>")
        sys.exit(1)
    
    if sys.argv[1] == '--sweep':
        if len(sys.argv) < 3:
            csv_dir = 'pairwise_sweep_csvs'
        else:
            csv_dir = sys.argv[2]
        
        n_values = [6, 7, 8]
        ar_values = [50, 150, 500]
        
        analyze_parametric_sweep(csv_dir, n_values, ar_values)
    else:
        # Single file analysis
        csv_file = sys.argv[1]
        
        positions_time, orientations_time, times = load_trajectory_csv(csv_file)
        metrics = compute_all_metrics(positions_time, orientations_time)
        
        print("\n=== Pairwise Metrics ===")
        for key, value in metrics.items():
            print(f"{key}: {value}")
