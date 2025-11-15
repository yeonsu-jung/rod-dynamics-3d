#!/usr/bin/env python3
"""
Study confined rod systems with kinetic energy tracking.

Creates and analyzes small systems of N rods in confined periodic boxes.
Box size is adjustable as a multiple of rod length.

Usage:
  python3 confined_study.py --n-rods 10 --box-factor 0.7 --steps 10000
  python3 confined_study.py --generate-only  # Just create scene
  python3 confined_study.py --analyze <csv>  # Analyze existing data
"""
import json
import numpy as np
import argparse
import os
import subprocess
import matplotlib.pyplot as plt
import pandas as pd


def generate_confined_scene(n_rods=10, box_factor=0.7, rod_length=0.5, rod_radius=0.025,
                           initial_ke=0.1, output_path=None, seed=42, 
                           perfect_conservation=False):
    """
    Generate a scene with N rods in a confined periodic box.
    
    Args:
        n_rods: Number of rods
        box_factor: Box linear size as fraction of rod length (e.g., 0.7 means box = 0.7 * L)
        rod_length: Total rod length (2 * halfLength)
        rod_radius: Rod radius
        initial_ke: Target initial kinetic energy per rod
        output_path: JSON output path
        seed: Random seed for reproducibility
        perfect_conservation: If True, use settings for best energy conservation
    
    Returns:
        Scene dictionary
    """
    np.random.seed(seed)
    
    half_length = rod_length / 2.0
    box_size = box_factor * rod_length
    box_half = box_size / 2.0
    
    print(f"\n{'='*60}")
    print(f"CONFINED ROD SYSTEM GENERATION")
    print(f"{'='*60}")
    print(f"  Number of rods: {n_rods}")
    print(f"  Rod length: {rod_length:.4f} (halfLength = {half_length:.4f})")
    print(f"  Rod radius: {rod_radius:.4f}")
    print(f"  Box size: {box_size:.4f} (factor = {box_factor:.2f})")
    print(f"  Box bounds: [{-box_half:.3f}, {box_half:.3f}]³")
    print(f"  Target KE per rod: {initial_ke:.4f}")
    
    # Calculate packing fraction
    rod_volume = np.pi * rod_radius**2 * rod_length
    box_volume = box_size**3
    packing_fraction = (n_rods * rod_volume) / box_volume
    print(f"  Packing fraction: {packing_fraction:.4f}")
    
    if packing_fraction > 0.5:
        print(f"  WARNING: High packing fraction! Expect many collisions.")
    
    # Generate rods
    bodies = []
    
    for i in range(n_rods):
        # Random position in box
        pos = np.random.uniform(-box_half, box_half, 3).tolist()
        
        # Random orientation (uniform on sphere)
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        axis = [
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ]
        
        # Random velocity (scaled to achieve target KE)
        # KE_lin = 0.5 * m * v²  =>  v = sqrt(2 * KE / m)
        # Split KE between linear and rotational
        ke_lin = initial_ke * 0.5
        ke_rot = initial_ke * 0.5
        
        mass = 1.0
        v_mag = np.sqrt(2 * ke_lin / mass)
        v_dir = np.random.randn(3)
        v_dir = v_dir / np.linalg.norm(v_dir)
        velocity = (v_mag * v_dir).tolist()
        
        # Angular velocity (rough approximation for rod inertia)
        # For rod: I ~ m * L² / 12
        I_approx = mass * rod_length**2 / 12.0
        w_mag = np.sqrt(2 * ke_rot / I_approx)
        w_dir = np.random.randn(3)
        w_dir = w_dir / np.linalg.norm(w_dir)
        ang_velocity = (w_mag * w_dir).tolist()
        
        body = {
            "pos": pos,
            "length": rod_length,
            "diameter": 2 * rod_radius,
            "density": mass / rod_volume,  # mass = density * volume
            "rot_axis": axis,
            "rot_deg": 0.0,
            "v_lin": velocity,
            "v_ang": ang_velocity
        }
        bodies.append(body)
    
    # Scene configuration
    physics_config = {
        "gravity": [0, 0, 0],
        "dt": 1.0/600.0,
        "solverIterations": 20,
        "lin_damp": 0.0,
        "ang_damp": 0.0,
        "friction": 0.0,
        "restitution": 1.0,
        "floor": False
    }
    
    # For perfect conservation, tune solver parameters
    if perfect_conservation:
        print(f"  Using PERFECT CONSERVATION mode:")
        print(f"    - Higher solver iterations (100)")
        print(f"    - Baumgarte = 0.0 (no artificial damping)")
        print(f"    - allowedPen = 0.0 (no drift tolerance)")
        print(f"    - splitImpulse = False (velocity-based correction)")
        print(f"    - w_max = 1e6 (no angular velocity clamping)")
        
        physics_config["solverIterations"] = 100
        physics_config["w_max"] = 1e6  # Effectively disable angular velocity clamping
        physics_config["solver"] = {
            "baumgarte": 0.0,
            "allowedPen": 0.0,
            "velIters": 100,
            "splitImpulse": False,
            "splitOrient": False,
            "ngsNormalSweeps": 0
        }
    
    scene = {
        "scene": {
            "bodies": bodies,
            "periodic": {
                "enabled": True,
                "min": [-box_half, -box_half, -box_half],
                "max": [box_half, box_half, box_half]
            }
        },
        "physics": physics_config
    }
    
    # Write JSON
    if output_path is None:
        output_path = f"../assets/scenes/confined_n{n_rods}_box{box_factor:.2f}.json"
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(scene, f, indent=2)
    
    print(f"\n  Scene saved to: {output_path}")
    print(f"{'='*60}\n")
    
    return scene, output_path


def run_simulation(scene_path, steps=10000, output_csv=None, perrod_csv=None, 
                   perrod_max_frames=1000, build_dir="../build"):
    """
    Run the simulation headless.
    
    Args:
        scene_path: Path to scene JSON
        steps: Number of simulation steps
        output_csv: Path for summary CSV (total KE over time)
        perrod_csv: Path for per-rod trajectory CSV
        perrod_max_frames: Max frames to sample in per-rod output
        build_dir: Build directory with executable
    """
    executable = os.path.join(build_dir, "rigidbody_viewer_3d")
    
    if not os.path.exists(executable):
        print(f"ERROR: Executable not found at {executable}")
        print(f"Please build the project first:")
        print(f"  cd {build_dir} && cmake .. && make")
        return False
    
    cmd = [
        executable,
        "--scene", scene_path,
        "--headless", str(steps)
    ]
    
    if output_csv:
        cmd.extend(["--csv", output_csv])
    
    if perrod_csv:
        cmd.extend(["--perrod", perrod_csv, "--perrod-max", str(perrod_max_frames)])
    
    print(f"Running simulation...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Steps: {steps}")
    
    result = subprocess.run(cmd, cwd=build_dir)
    
    if result.returncode == 0:
        print(f"  Simulation completed successfully!")
        return True
    else:
        print(f"  Simulation failed with return code {result.returncode}")
        return False


def analyze_ke_evolution(perrod_csv, output_dir=None):
    """
    Analyze kinetic energy evolution from per-rod CSV.
    
    Args:
        perrod_csv: Path to per-rod trajectory CSV
        output_dir: Directory to save analysis plots
    """
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(perrod_csv))[0]
        output_dir = f"confined_analysis_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"KINETIC ENERGY ANALYSIS")
    print(f"{'='*60}")
    print(f"  Input: {perrod_csv}")
    
    # Load data
    df = pd.read_csv(perrod_csv)
    frames = sorted(df['frame'].unique())
    n_rods = len(df['rod'].unique())
    
    print(f"  Frames: {len(frames)}")
    print(f"  Rods: {n_rods}")
    
    # Compute total KE per frame
    ke_total = []
    ke_lin_total = []
    ke_rot_total = []
    
    for frame in frames:
        df_frame = df[df['frame'] == frame]
        ke_total.append(df_frame['KE_total'].sum())
        ke_lin_total.append(df_frame['KE_lin'].sum())
        ke_rot_total.append(df_frame['KE_rot'].sum())
    
    ke_total = np.array(ke_total)
    ke_lin_total = np.array(ke_lin_total)
    ke_rot_total = np.array(ke_rot_total)
    
    # Statistics
    ke_initial = ke_total[0]
    ke_final = ke_total[-1]
    ke_loss = ke_initial - ke_final
    ke_loss_pct = 100 * ke_loss / ke_initial
    
    print(f"\n  Initial total KE: {ke_initial:.6f}")
    print(f"  Final total KE: {ke_final:.6f}")
    print(f"  KE loss: {ke_loss:.6f} ({ke_loss_pct:.2f}%)")
    print(f"  Mean KE: {ke_total.mean():.6f} ± {ke_total.std():.6f}")
    
    # Save summary
    summary_df = pd.DataFrame({
        'frame': frames,
        'KE_total': ke_total,
        'KE_lin': ke_lin_total,
        'KE_rot': ke_rot_total
    })
    summary_csv = os.path.join(output_dir, 'ke_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  Summary saved to: {summary_csv}")
    
    # Plots
    create_ke_plots(frames, ke_total, ke_lin_total, ke_rot_total, output_dir)
    
    print(f"{'='*60}\n")
    
    return summary_df


def create_ke_plots(frames, ke_total, ke_lin, ke_rot, output_dir):
    """Create kinetic energy visualization plots."""
    
    # Plot 1: Total, linear, rotational KE
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(frames, ke_total, 'k-', label='Total KE', linewidth=2, alpha=0.8)
    ax.plot(frames, ke_lin, 'b-', label='Linear KE', linewidth=1.5, alpha=0.7)
    ax.plot(frames, ke_rot, 'r-', label='Rotational KE', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Kinetic Energy')
    ax.set_title('Kinetic Energy Evolution in Confined System')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ke_evolution.png'), dpi=150)
    plt.close()
    
    # Plot 2: KE fractions
    fig, ax = plt.subplots(figsize=(12, 6))
    frac_lin = ke_lin / (ke_total + 1e-10)
    frac_rot = ke_rot / (ke_total + 1e-10)
    ax.plot(frames, frac_lin, 'b-', label='Linear fraction', linewidth=1.5)
    ax.plot(frames, frac_rot, 'r-', label='Rotational fraction', linewidth=1.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3, label='Equal partition')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Fraction of Total KE')
    ax.set_title('Energy Partitioning: Linear vs Rotational')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ke_fractions.png'), dpi=150)
    plt.close()
    
    # Plot 3: KE loss over time
    fig, ax = plt.subplots(figsize=(12, 6))
    ke_loss = ke_total[0] - ke_total
    ax.plot(frames, ke_loss, 'purple', linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('KE Lost (from initial)')
    ax.set_title('Cumulative Kinetic Energy Loss')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ke_loss.png'), dpi=150)
    plt.close()
    
    # Plot 4: Log scale
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(frames, ke_total, 'k-', linewidth=2, label='Total KE')
    ax.semilogy(frames, ke_lin, 'b-', linewidth=1.5, alpha=0.7, label='Linear KE')
    ax.semilogy(frames, ke_rot, 'r-', linewidth=1.5, alpha=0.7, label='Rotational KE')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Kinetic Energy (log scale)')
    ax.set_title('KE Evolution (Log Scale)')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ke_evolution_log.png'), dpi=150)
    plt.close()
    
    print(f"  Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Study confined rod systems")
    parser.add_argument('--n-rods', type=int, default=10, help='Number of rods')
    parser.add_argument('--box-factor', type=float, default=0.7, 
                       help='Box size as fraction of rod length')
    parser.add_argument('--rod-length', type=float, default=0.5, help='Rod length')
    parser.add_argument('--initial-ke', type=float, default=0.1, 
                       help='Initial KE per rod')
    parser.add_argument('--steps', type=int, default=10000, help='Simulation steps')
    parser.add_argument('--perrod-frames', type=int, default=1000, 
                       help='Max frames in per-rod output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--perfect-conservation', action='store_true',
                       help='Use settings for best energy conservation (no Baumgarte, high iterations)')
    parser.add_argument('--generate-only', action='store_true', 
                       help='Only generate scene, do not run')
    parser.add_argument('--analyze', type=str, help='Analyze existing per-rod CSV')
    
    args = parser.parse_args()
    
    if args.analyze:
        # Analyze mode
        analyze_ke_evolution(args.analyze)
    else:
        # Generate and optionally run
        scene, scene_path = generate_confined_scene(
            n_rods=args.n_rods,
            box_factor=args.box_factor,
            rod_length=args.rod_length,
            initial_ke=args.initial_ke,
            seed=args.seed,
            perfect_conservation=args.perfect_conservation
        )
        
        if not args.generate_only:
            base_name = f"confined_n{args.n_rods}_box{args.box_factor:.2f}"
            perrod_csv = f"{base_name}.csv"
            
            success = run_simulation(
                scene_path,
                steps=args.steps,
                perrod_csv=perrod_csv,
                perrod_max_frames=args.perrod_frames
            )
            
            if success and os.path.exists(perrod_csv):
                analyze_ke_evolution(perrod_csv)


if __name__ == "__main__":
    main()
