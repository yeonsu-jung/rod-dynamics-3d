#!/usr/bin/env python3
"""
Quick analysis script for a single run: KE dynamics and COM movement.

Usage:
    python quick_analyze_run.py <run_directory>
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_run(run_dir):
    """Analyze KE dynamics and COM movement for a single run."""
    run_dir = Path(run_dir)
    
    print(f"\nAnalyzing: {run_dir.name}")
    print("=" * 80)
    
    # Load profile.csv for KE data
    profile_csv = run_dir / "profile.csv"
    if not profile_csv.exists():
        print(f"ERROR: No profile.csv found in {run_dir}")
        return
    
    df_profile = pd.read_csv(profile_csv)
    print(f"\nProfile data: {len(df_profile)} frames")
    print(f"Columns: {list(df_profile.columns)}")
    
    # Load com.csv for COM movement
    com_csv = run_dir / "com.csv"
    if not com_csv.exists():
        print(f"WARNING: No com.csv found in {run_dir}")
        df_com = None
    else:
        df_com = pd.read_csv(com_csv)
        print(f"\nCOM data: {len(df_com)} frames")
        print(f"Columns: {list(df_com.columns)}")
    
    # Compute KE statistics
    print("\n" + "=" * 80)
    print("KINETIC ENERGY STATISTICS")
    print("=" * 80)
    
    ke = df_profile['KE'].to_numpy()
    frames = df_profile['frame'].to_numpy()
    
    # Full statistics
    print(f"Mean KE:   {np.mean(ke):.6f}")
    print(f"Std KE:    {np.std(ke):.6f}")
    print(f"Min KE:    {np.min(ke):.6f}")
    print(f"Max KE:    {np.max(ke):.6f}")
    print(f"Final KE:  {ke[-1]:.6f}")
    
    # Latter half statistics (equilibrated)
    half_idx = len(ke) // 2
    ke_late = ke[half_idx:]
    print(f"\nLatter half (frames {frames[half_idx]}-{frames[-1]}):")
    print(f"Mean KE:   {np.mean(ke_late):.6f}")
    print(f"Std KE:    {np.std(ke_late):.6f}")
    print(f"CV:        {np.std(ke_late)/np.mean(ke_late):.4f}")
    
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
        growth_rate = coeffs[0]
        print(f"\nGrowth rate (d(ln KE)/dt): {growth_rate:.6e}")
        if abs(growth_rate) < 1e-6:
            print("  -> System appears equilibrated (near-zero growth)")
        elif growth_rate > 0:
            print("  -> System is heating up")
        else:
            print("  -> System is cooling down")
    
    # Compute COM statistics if available
    if df_com is not None:
        print("\n" + "=" * 80)
        print("CENTER OF MASS STATISTICS")
        print("=" * 80)
        
        com_x = df_com['com_x'].to_numpy()
        com_y = df_com['com_y'].to_numpy()
        com_z = df_com['com_z'].to_numpy()
        
        # Initial and final positions
        print(f"\nInitial COM: ({com_x[0]:.6f}, {com_y[0]:.6f}, {com_z[0]:.6f})")
        print(f"Final COM:   ({com_x[-1]:.6f}, {com_y[-1]:.6f}, {com_z[-1]:.6f})")
        
        # Displacement
        dx = com_x[-1] - com_x[0]
        dy = com_y[-1] - com_y[0]
        dz = com_z[-1] - com_z[0]
        total_displacement = np.sqrt(dx**2 + dy**2 + dz**2)
        
        print(f"\nTotal displacement: {total_displacement:.6f}")
        print(f"  ΔX: {dx:+.6f}")
        print(f"  ΔY: {dy:+.6f}")
        print(f"  ΔZ: {dz:+.6f}")
        
        # Mean position and drift
        print(f"\nMean COM position:")
        print(f"  X: {np.mean(com_x):.6f} ± {np.std(com_x):.6f}")
        print(f"  Y: {np.mean(com_y):.6f} ± {np.std(com_y):.6f}")
        print(f"  Z: {np.mean(com_z):.6f} ± {np.std(com_z):.6f}")
        
        # Drift rate
        com_frames = df_com['frame'].to_numpy()
        drift_rate = total_displacement / (com_frames[-1] - com_frames[0])
        print(f"\nDrift rate: {drift_rate:.6e} units/frame")
    
    # Create plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    figs_dir = run_dir / "figs"
    figs_dir.mkdir(exist_ok=True)
    
    # Plot KE time trace
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frames, ke, 'b-', linewidth=1, alpha=0.7)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Kinetic Energy', fontsize=12)
    ax.set_title(f'Kinetic Energy vs Frame\n{run_dir.name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add equilibration line
    if half_idx > 0:
        ax.axvline(frames[half_idx], color='r', linestyle='--', alpha=0.5, 
                   label=f'Latter half start (frame {frames[half_idx]})')
        ax.legend()
    
    ke_plot = figs_dir / "ke_dynamics.png"
    plt.tight_layout()
    plt.savefig(ke_plot, dpi=150)
    plt.close()
    print(f"  Saved: {ke_plot}")
    
    # Plot COM trajectory if available
    if df_com is not None:
        # 3D trajectory
        fig = plt.figure(figsize=(12, 10))
        
        # 3D plot
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        scatter = ax1.scatter(com_x, com_y, com_z, c=com_frames, cmap='viridis', s=1, alpha=0.6)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('COM 3D Trajectory')
        plt.colorbar(scatter, ax=ax1, label='Frame')
        
        # XY projection
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(com_x, com_y, c=com_frames, cmap='viridis', s=1, alpha=0.6)
        ax2.plot(com_x[0], com_y[0], 'go', markersize=10, label='Start')
        ax2.plot(com_x[-1], com_y[-1], 'ro', markersize=10, label='End')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('COM XY Projection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # XZ projection
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.scatter(com_x, com_z, c=com_frames, cmap='viridis', s=1, alpha=0.6)
        ax3.plot(com_x[0], com_z[0], 'go', markersize=10, label='Start')
        ax3.plot(com_x[-1], com_z[-1], 'ro', markersize=10, label='End')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('COM XZ Projection')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # YZ projection
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.scatter(com_y, com_z, c=com_frames, cmap='viridis', s=1, alpha=0.6)
        ax4.plot(com_y[0], com_z[0], 'go', markersize=10, label='Start')
        ax4.plot(com_y[-1], com_z[-1], 'ro', markersize=10, label='End')
        ax4.set_xlabel('Y')
        ax4.set_ylabel('Z')
        ax4.set_title('COM YZ Projection')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        plt.suptitle(f'Center of Mass Trajectory\n{run_dir.name}', fontsize=14)
        plt.tight_layout()
        
        com_plot = figs_dir / "com_trajectory.png"
        plt.savefig(com_plot, dpi=150)
        plt.close()
        print(f"  Saved: {com_plot}")
        
        # Time evolution plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        axes[0].plot(com_frames, com_x, 'b-', linewidth=1)
        axes[0].set_ylabel('COM X')
        axes[0].set_title('COM Position vs Frame')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(com_frames, com_y, 'g-', linewidth=1)
        axes[1].set_ylabel('COM Y')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(com_frames, com_z, 'r-', linewidth=1)
        axes[2].set_xlabel('Frame')
        axes[2].set_ylabel('COM Z')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'COM Evolution\n{run_dir.name}', fontsize=14)
        plt.tight_layout()
        
        com_evolution_plot = figs_dir / "com_evolution.png"
        plt.savefig(com_evolution_plot, dpi=150)
        plt.close()
        print(f"  Saved: {com_evolution_plot}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll plots saved to: {figs_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_analyze_run.py <run_directory>")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    analyze_run(run_dir)
