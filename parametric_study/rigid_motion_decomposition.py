#!/usr/bin/env python3
"""
Rigid Motion Decomposition for Rod Ensembles

Implements the theory from RodMotionDecomposition.md:
- Twist fitting (ω, v₀) from velocity field
- Energy decomposition: T_total = T_global + T_def
- Kabsch algorithm for position alignment
- Visualization of global vs deformational motion

Given per-rod trajectories from simulation, computes:
1. Best-fit global rigid motion at each timestep
2. Deformational velocity (residual after removing global motion)
3. Global and deformational kinetic energies
4. Screw motion parameters (axis, pitch, rotation angle)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.linalg import svd
from typing import Tuple, Dict, List
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


def skew_matrix(v):
    """Convert vector to skew-symmetric matrix [v]ₓ for cross product."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def fit_twist(positions, velocities, masses):
    """
    Fit best rigid motion twist (ω, v₀) to velocity field.
    
    Minimizes: Σᵢ mᵢ ‖vᵢ - (ω × xᵢ + v₀)‖²
    
    Args:
        positions: (N, 3) array of marker positions
        velocities: (N, 3) array of marker velocities
        masses: (N,) array of marker masses
    
    Returns:
        omega: (3,) angular velocity vector
        v0: (3,) translational velocity
        residual: RMS residual velocity magnitude
    """
    N = len(positions)
    
    # Center of mass and mean velocity
    total_mass = masses.sum()
    x_cm = (masses[:, None] * positions).sum(axis=0) / total_mass
    v_cm = (masses[:, None] * velocities).sum(axis=0) / total_mass
    
    # Centered positions and velocities
    x_tilde = positions - x_cm
    v_tilde = velocities - v_cm
    
    # Build linear system J ω = b
    # J = Σᵢ mᵢ [r̃ᵢ]ₓ² = Σᵢ mᵢ [r̃ᵢ]ₓᵀ [r̃ᵢ]ₓ
    # b = Σᵢ mᵢ [r̃ᵢ]ₓ ṽᵢ
    
    J = np.zeros((3, 3))
    b = np.zeros(3)
    
    for i in range(N):
        r_skew = skew_matrix(x_tilde[i])
        J += masses[i] * (r_skew.T @ r_skew)
        b += masses[i] * (r_skew.T @ v_tilde[i])
    
    # Solve for ω
    # Add small regularization for numerical stability
    J_reg = J + 1e-10 * np.eye(3)
    omega = np.linalg.solve(J_reg, b)
    
    # Compute v₀
    v0 = v_cm - np.cross(omega, x_cm)
    
    # Compute residual
    v_rigid = np.array([np.cross(omega, x) + v0 for x in positions])
    residuals = velocities - v_rigid
    residual_rms = np.sqrt((masses[:, None] * residuals**2).sum() / total_mass)
    
    return omega, v0, residual_rms


def decompose_velocities(positions, velocities, masses):
    """
    Decompose velocity field into rigid + deformational components.
    
    Returns:
        v_rigid: (N, 3) rigid motion velocities
        v_def: (N, 3) deformational velocities
        omega: (3,) angular velocity
        v0: (3,) translational velocity
        residual: RMS residual
    """
    omega, v0, residual = fit_twist(positions, velocities, masses)
    
    v_rigid = np.array([np.cross(omega, x) + v0 for x in positions])
    v_def = velocities - v_rigid
    
    return v_rigid, v_def, omega, v0, residual


def compute_kinetic_energies(velocities, masses):
    """
    Compute total kinetic energy from velocities.
    
    Args:
        velocities: (N, 3) velocities
        masses: (N,) masses
    
    Returns:
        Total kinetic energy
    """
    return 0.5 * (masses[:, None] * velocities**2).sum()


def kabsch_algorithm(P, Q, masses=None):
    """
    Find optimal rotation R and translation t to align P onto Q.
    
    Minimizes: Σᵢ mᵢ ‖qᵢ - (R pᵢ + t)‖²
    
    Args:
        P: (N, 3) source points
        Q: (N, 3) target points
        masses: (N,) optional weights (default: uniform)
    
    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        rmsd: Root mean squared deviation
    """
    N = len(P)
    if masses is None:
        masses = np.ones(N)
    
    total_mass = masses.sum()
    
    # Centroids
    p_cm = (masses[:, None] * P).sum(axis=0) / total_mass
    q_cm = (masses[:, None] * Q).sum(axis=0) / total_mass
    
    # Centered coordinates
    P_tilde = P - p_cm
    Q_tilde = Q - q_cm
    
    # Covariance matrix H = Σᵢ mᵢ P̃ᵢ Q̃ᵢᵀ
    H = (masses[:, None, None] * P_tilde[:, :, None] * Q_tilde[:, None, :]).sum(axis=0)
    
    # SVD
    U, S, Vt = svd(H)
    V = Vt.T
    
    # Rotation matrix (handle reflection)
    d = np.linalg.det(V @ U.T)
    R = V @ np.diag([1, 1, d]) @ U.T
    
    # Translation
    t = q_cm - R @ p_cm
    
    # RMSD
    Q_fit = (R @ P.T).T + t
    rmsd = np.sqrt((masses[:, None] * (Q - Q_fit)**2).sum() / total_mass)
    
    return R, t, rmsd


def screw_parameters(R, t):
    """
    Extract screw motion parameters from rigid transform.
    
    Args:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    
    Returns:
        axis: (3,) screw axis direction (unit vector)
        angle: rotation angle in radians
        pitch: translation along axis
        point: (3,) point on screw axis
    """
    # Rotation angle from trace
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    angle = np.arccos(cos_theta)
    
    if angle < 1e-6:
        # Pure translation
        axis = np.array([0, 0, 1]) if np.linalg.norm(t) < 1e-10 else t / np.linalg.norm(t)
        pitch = np.linalg.norm(t)
        point = np.zeros(3)
        return axis, angle, pitch, point
    
    # Axis is eigenvector with eigenvalue 1
    eigenvalues, eigenvectors = np.linalg.eig(R)
    idx = np.argmin(np.abs(eigenvalues - 1))
    axis = np.real(eigenvectors[:, idx])
    axis = axis / np.linalg.norm(axis)
    
    # Pitch (translation along axis)
    pitch = np.dot(axis, t)
    
    # Point on axis: solve (I - R)p = t - (n·t)n
    I_minus_R = np.eye(3) - R
    rhs = t - pitch * axis
    
    # Use pseudo-inverse since I-R is singular
    point = np.linalg.lstsq(I_minus_R, rhs, rcond=None)[0]
    
    return axis, angle, pitch, point


def load_trajectory_csv(csv_path):
    """
    Load per-rod trajectory CSV from simulation.
    
    Expected columns: frame, rod, px, py, pz, vx, vy, vz, wx, wy, wz, qw, qx, qy, qz, KE_lin, KE_rot, KE_total
    
    Returns:
        positions_time: (N_timesteps, N_rods, 3)
        velocities_time: (N_timesteps, N_rods, 3)
        ang_velocities_time: (N_timesteps, N_rods, 3)
        orientations_time: (N_timesteps, N_rods, 3) - unit axes
        frames: (N_timesteps,) frame numbers
        masses: (N_rods,) rod masses (assumed uniform = 1.0)
    """
    df = pd.read_csv(csv_path)
    
    frames = sorted(df['frame'].unique())
    rod_ids = sorted(df['rod'].unique())
    N_timesteps = len(frames)
    N_rods = len(rod_ids)
    
    positions_time = np.zeros((N_timesteps, N_rods, 3))
    velocities_time = np.zeros((N_timesteps, N_rods, 3))
    ang_velocities_time = np.zeros((N_timesteps, N_rods, 3))
    orientations_time = np.zeros((N_timesteps, N_rods, 3))
    
    for t_idx, frame in enumerate(frames):
        df_t = df[df['frame'] == frame].sort_values('rod')
        
        positions_time[t_idx] = df_t[['px', 'py', 'pz']].values
        velocities_time[t_idx] = df_t[['vx', 'vy', 'vz']].values
        ang_velocities_time[t_idx] = df_t[['wx', 'wy', 'wz']].values
        
        # Convert quaternions to orientation axes
        for rod_idx, row in enumerate(df_t.itertuples()):
            q = [row.qw, row.qx, row.qy, row.qz]
            axis = quaternion_to_axis(q)
            orientations_time[t_idx, rod_idx] = axis
    
    # Assume uniform mass = 1.0 per rod
    masses = np.ones(N_rods)
    
    return positions_time, velocities_time, ang_velocities_time, orientations_time, frames, masses


def analyze_rigid_motion_decomposition(csv_path, output_dir=None):
    """
    Perform full rigid motion decomposition analysis on simulation data.
    
    Args:
        csv_path: Path to per-rod trajectory CSV
        output_dir: Directory to save plots and results (default: create from csv name)
    
    Returns:
        results: Dictionary with time series of decomposition quantities
    """
    # Load data
    print(f"Loading trajectory from {csv_path}...")
    positions_time, velocities_time, ang_velocities_time, orientations_time, frames, masses = load_trajectory_csv(csv_path)
    N_timesteps, N_rods, _ = positions_time.shape
    
    print(f"  {N_timesteps} timesteps, {N_rods} rods")
    
    # Setup output
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_dir = f"rigid_decomp_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for results
    results = {
        'frames': frames,
        'omega': np.zeros((N_timesteps, 3)),
        'v0': np.zeros((N_timesteps, 3)),
        'residual_rms': np.zeros(N_timesteps),
        'KE_total': np.zeros(N_timesteps),
        'KE_global': np.zeros(N_timesteps),
        'KE_def': np.zeros(N_timesteps),
        'screw_axis': np.zeros((N_timesteps, 3)),
        'screw_angle': np.zeros(N_timesteps),
        'screw_pitch': np.zeros(N_timesteps),
        'v_rigid_time': np.zeros((N_timesteps, N_rods, 3)),
        'v_def_time': np.zeros((N_timesteps, N_rods, 3))
    }
    
    print("\nDecomposing motion at each timestep...")
    
    for t_idx in range(N_timesteps):
        positions = positions_time[t_idx]
        velocities = velocities_time[t_idx]
        
        # Decompose velocities
        v_rigid, v_def, omega, v0, residual = decompose_velocities(positions, velocities, masses)
        
        results['omega'][t_idx] = omega
        results['v0'][t_idx] = v0
        results['residual_rms'][t_idx] = residual
        results['v_rigid_time'][t_idx] = v_rigid
        results['v_def_time'][t_idx] = v_def
        
        # Compute energies
        KE_total = compute_kinetic_energies(velocities, masses)
        KE_global = compute_kinetic_energies(v_rigid, masses)
        KE_def = compute_kinetic_energies(v_def, masses)
        
        results['KE_total'][t_idx] = KE_total
        results['KE_global'][t_idx] = KE_global
        results['KE_def'][t_idx] = KE_def
        
        # Screw parameters (fit from previous frame if available)
        if t_idx > 0:
            R, t, rmsd = kabsch_algorithm(positions_time[t_idx-1], positions, masses)
            axis, angle, pitch, point = screw_parameters(R, t)
            results['screw_axis'][t_idx] = axis
            results['screw_angle'][t_idx] = angle
            results['screw_pitch'][t_idx] = pitch
        
        if (t_idx + 1) % 100 == 0:
            print(f"  Frame {t_idx+1}/{N_timesteps}")
    
    # Verify energy decomposition
    energy_error = np.abs(results['KE_total'] - (results['KE_global'] + results['KE_def']))
    max_error = energy_error.max()
    print(f"\nEnergy decomposition verification:")
    print(f"  Max |KE_total - (KE_global + KE_def)|: {max_error:.2e}")
    print(f"  Should be ~0 due to orthogonality")
    
    # Save results
    df_results = pd.DataFrame({
        'frame': frames,
        'omega_x': results['omega'][:, 0],
        'omega_y': results['omega'][:, 1],
        'omega_z': results['omega'][:, 2],
        'v0_x': results['v0'][:, 0],
        'v0_y': results['v0'][:, 1],
        'v0_z': results['v0'][:, 2],
        'residual_rms': results['residual_rms'],
        'KE_total': results['KE_total'],
        'KE_global': results['KE_global'],
        'KE_def': results['KE_def'],
        'screw_angle': results['screw_angle'],
        'screw_pitch': results['screw_pitch']
    })
    
    csv_out = os.path.join(output_dir, 'rigid_decomposition.csv')
    df_results.to_csv(csv_out, index=False)
    print(f"\nResults saved to {csv_out}")
    
    # Create plots
    plot_decomposition_results(results, output_dir)
    
    return results


def plot_decomposition_results(results, output_dir):
    """Create comprehensive plots of rigid motion decomposition."""
    frames = results['frames']
    
    # 1. Energy decomposition over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frames, results['KE_total'], 'k-', label='Total KE', linewidth=2)
    ax.plot(frames, results['KE_global'], 'b-', label='Global KE', linewidth=1.5)
    ax.plot(frames, results['KE_def'], 'r-', label='Deformational KE', linewidth=1.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Kinetic Energy')
    ax.set_title('Energy Decomposition: Total = Global + Deformational')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_decomposition.png'), dpi=150)
    plt.close()
    
    # 2. Energy fractions
    fig, ax = plt.subplots(figsize=(10, 6))
    total_nonzero = results['KE_total'] > 1e-10
    frac_global = np.zeros_like(results['KE_total'])
    frac_def = np.zeros_like(results['KE_total'])
    frac_global[total_nonzero] = results['KE_global'][total_nonzero] / results['KE_total'][total_nonzero]
    frac_def[total_nonzero] = results['KE_def'][total_nonzero] / results['KE_total'][total_nonzero]
    
    ax.plot(frames, frac_global, 'b-', label='Global KE / Total', linewidth=1.5)
    ax.plot(frames, frac_def, 'r-', label='Deformational KE / Total', linewidth=1.5)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Fraction of Total KE')
    ax.set_title('Energy Partitioning')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_fractions.png'), dpi=150)
    plt.close()
    
    # 3. Angular velocity magnitude
    fig, ax = plt.subplots(figsize=(10, 6))
    omega_mag = np.linalg.norm(results['omega'], axis=1)
    ax.plot(frames, omega_mag, 'g-', linewidth=1.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('|ω| (rad/s)')
    ax.set_title('Global Angular Velocity Magnitude')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'angular_velocity.png'), dpi=150)
    plt.close()
    
    # 4. Translational velocity magnitude
    fig, ax = plt.subplots(figsize=(10, 6))
    v0_mag = np.linalg.norm(results['v0'], axis=1)
    ax.plot(frames, v0_mag, 'm-', linewidth=1.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('|v₀| (units/s)')
    ax.set_title('Global Translational Velocity Magnitude')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'translational_velocity.png'), dpi=150)
    plt.close()
    
    # 5. Screw motion parameters
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(frames, np.degrees(results['screw_angle']), 'b-', linewidth=1.5)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Rotation Angle (degrees)')
    ax1.set_title('Screw Motion: Rotation per Timestep')
    ax1.grid(alpha=0.3)
    
    ax2.plot(frames, results['screw_pitch'], 'r-', linewidth=1.5)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Pitch (translation along axis)')
    ax2.set_title('Screw Motion: Translation along Rotation Axis')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'screw_parameters.png'), dpi=150)
    plt.close()
    
    # 6. Residual (goodness of rigid motion fit)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frames, results['residual_rms'], 'orange', linewidth=1.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('RMS Residual Velocity')
    ax.set_title('Rigid Motion Fit Quality (lower = more rigid-like)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fit_residual.png'), dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python rigid_motion_decomposition.py <per_rod_csv_file>")
        print("\nExample:")
        print("  python rigid_motion_decomposition.py pairwise_sweep_csvs/pairwise_n6_ar50.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    results = analyze_rigid_motion_decomposition(csv_path)
    
    print("\n=== Summary Statistics ===")
    print(f"Mean KE_global: {results['KE_global'].mean():.4f}")
    print(f"Mean KE_def: {results['KE_def'].mean():.4f}")
    print(f"Mean global fraction: {(results['KE_global'] / (results['KE_total'] + 1e-10)).mean():.3f}")
    print(f"Mean residual RMS: {results['residual_rms'].mean():.4e}")
