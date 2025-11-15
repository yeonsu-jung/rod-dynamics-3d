#!/usr/bin/env python3
"""
Test energy conservation with 10 rods in a confined box using soft contacts.
Goal: Verify no energy dissipation over extended simulation.
"""

import numpy as np
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

def create_scene_n10_confined(filename, box_size=0.80, rod_length=0.10, rod_radius=0.005):
    """
    Create scene with 10 rods randomly placed in a cubic box with periodic boundaries.
    
    Args:
        box_size: Side length of confining box
        rod_length: Length of each rod
        rod_radius: Radius of each rod (diameter = 2*radius)
    """
    np.random.seed(42)  # Reproducible initial conditions
    
    n_rods = 10
    bodies = []
    
    # Random initial positions within box (with margin for rod length)
    half = box_size / 2.0
    margin = rod_length / 2.0
    for i in range(n_rods):
        # Random position
        x = np.random.uniform(-half + margin, half - margin)
        y = np.random.uniform(-half + margin, half - margin)
        z = np.random.uniform(-half + margin, half - margin)
        
        # Random orientation (uniform on sphere)
        # Use axis-angle representation
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # Random initial velocities (small)
        vx = np.random.uniform(-0.1, 0.1)
        vy = np.random.uniform(-0.1, 0.1)
        vz = np.random.uniform(-0.1, 0.1)
        
        # Random angular velocities (small)
        wx = np.random.uniform(-0.5, 0.5)
        wy = np.random.uniform(-0.5, 0.5)
        wz = np.random.uniform(-0.5, 0.5)
        
        body = {
            "pos": [x, y, z],
            "length": rod_length,
            "diameter": 2.0 * rod_radius,
            "density": 1000.0,
            "rot_axis": axis.tolist(),
            "rot_deg": 0.0,
            "v_lin": [vx, vy, vz],
            "v_ang": [wx, wy, wz]
        }
        bodies.append(body)
    
    # IMPORTANT: Config loader expects bodies & periodic INSIDE top-level "scene" object.
    # physics keys use snake_case (lin_damp, ang_damp) not linDamp/angDamp.
    scene = {
        "physics": {
            "gravity": [0.0, 0.0, 0.0],
            "dt": 1.0 / 12000.0,
            "lin_damp": 0.0,          # zero linear damping (energy conservation)
            "ang_damp": 0.0,          # zero angular damping
            "friction": 0.0,
            "restitution": 1.0,
            "soft_contact": {
                "enabled": True,
                "delta": 0.005,
                "k_scaler": 500,
                "mu": 0.0,
                "enable_friction": False,
                "verbose": False
            }
        },
        "scene": {
            "periodic": {
                "enabled": True,
                "min": [-half, -half, -half],
                "max": [half, half, half]
            },
            "bodies": bodies
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(scene, f, indent=2)
    
    print(f"Created scene: {n_rods} rods in {box_size}x{box_size}x{box_size} box")
    print(f"  Rod dimensions: length={rod_length}, diameter={2*rod_radius}")
    print(f"  Timestep: dt={scene['physics']['dt']:.6f}")
    print(f"  Periodic boundaries: [{-half}, {half}]^3")


def run_simulation(scene_file, output_csv, steps=24000, k_scaler=500, delta=0.005, mu=0.0):
    """Run headless simulation with soft contacts."""
    
    cmd = [
        "../build/rigidbody_viewer_3d",
        "--headless",
        "--scene", scene_file,
        "--steps", str(steps),
        "--perrod", output_csv,
        "--perrod-max", "2000",  # Maximum frames to save (skip if too many)
        "--soft-contact",
        "--k-scaler", str(k_scaler),
        "--delta", str(delta),
        "--mu", str(mu),
        "--verbose", "false"
    ]
    
    print(f"\nRunning simulation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Simulation failed with code {result.returncode}")
    
    print("Simulation complete!")
    return output_csv


def analyze_energy_conservation(csv_file):
    """Analyze energy conservation from per-rod CSV (one row per rod per logged frame)."""
    df = pd.read_csv(csv_file)

    required = {"frame","rod","KE_total"}
    if not required.issubset(df.columns):
        print("ERROR: CSV missing required columns for energy analysis.")
        print("Columns present:", df.columns.tolist())
        return None

    # Aggregate total kinetic energy per logged frame across all rods
    ke_by_frame = df.groupby('frame')['KE_total'].sum()
    ke_initial = ke_by_frame.iloc[0]
    ke_final = ke_by_frame.iloc[-1]
    ke_mean = ke_by_frame.mean()
    ke_std = ke_by_frame.std()
    ke_min = ke_by_frame.min()
    ke_max = ke_by_frame.max()

    energy_change = ((ke_final - ke_initial) / ke_initial) * 100.0
    energy_drift = ((ke_max - ke_min) / ke_mean) * 100.0

    print("\n" + "="*60)
    print("ENERGY CONSERVATION ANALYSIS (Aggregated)")
    print("="*60)
    print(f"Logged frames:    {len(ke_by_frame)} (sampling skip applied)")
    print(f"Initial KE:       {ke_initial:.9f} J")
    print(f"Final KE:         {ke_final:.9f} J")
    print(f"Mean KE:          {ke_mean:.9f} J")
    print(f"Std Dev:          {ke_std:.9f} J ({(ke_std/ke_mean)*100:.3f}%)")
    print(f"Min KE:           {ke_min:.9f} J")
    print(f"Max KE:           {ke_max:.9f} J")
    print(f"\nEnergy change:    {energy_change:+.6f}%")
    print(f"Energy drift:     {energy_drift:.6f}%")
    print("="*60)

    # Attach aggregated series for downstream plotting
    df_ke = ke_by_frame.reset_index().rename(columns={'KE_total':'KE_sum'})
    return df_ke


def plot_energy(df, output_file="confined_n10_energy.png"):
    """Plot total kinetic energy over time."""
    
    if df is None or 'KE_total' not in df.columns:
        print("Cannot plot: No energy data")
        return
    
    dt = 1.0 / 12000.0
    time = np.arange(len(df)) * dt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(time, df['KE_total'], 'b-', linewidth=1.5, label='Total KE')
    ax.axhline(df['KE_total'].iloc[0], color='r', linestyle='--', 
               linewidth=1, label='Initial KE', alpha=0.7)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Kinetic Energy (J)', fontsize=12)
    ax.set_title('Energy Conservation: 10 Rods in Confined Box (Soft Contacts)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nPlot saved: {output_file}")
    plt.close()


def main():
    scene_file = "scene_confined_n10.json"
    output_csv = "confined_n10.csv"
    
    # Create scene with small rods in large box for low density
    create_scene_n10_confined(
        scene_file,
        box_size=0.80,      # 80cm box
        rod_length=0.10,    # 10cm rods
        rod_radius=0.005    # 5mm radius
    )
    
    # Run simulation with optimized soft contact parameters
    # Run for 2 seconds (24000 steps at dt=1/12000)
    run_simulation(
        scene_file,
        output_csv,
        steps=24000,     # 2 seconds
        k_scaler=500,    # Tuned stiffness
        delta=0.005,     # Transition width
        mu=0.0           # No friction (pure elastic)
    )
    
    # Analyze results
    df = analyze_energy_conservation(output_csv)
    
    # Visualize
    if df is not None:
        plot_energy(df)


if __name__ == "__main__":
    main()
