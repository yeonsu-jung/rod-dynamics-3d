#!/usr/bin/env python3
"""
submit_aspect_ratio.py

Submit SLURM jobs for aspect ratio parametric sweep: Effects of length-to-diameter ratio.

- Sweeps length-to-diameter ratio (L/D), friction coefficient, and noise amplitude.
- For each parameter set, creates a run directory with generated scene.json.
- Rod length is fixed at 1.0, diameter varies to achieve desired L/D ratios.

Usage:
  
  For SLURM cluster submission:
    python submit_aspect_ratio.py --job-name aspect_ratio_test
  
  For local debugging (runs one random parameter set):
    python submit_aspect_ratio.py --local --job-name debug_test
  
  For dry run (creates files but doesn't submit):
    python submit_aspect_ratio.py --dry-run --job-name test

Then, for analysis: python post_analyze_aspect_ratio.py --job-name aspect_ratio_test --make-plots --outdir analysis_aspect_ratio_test

Prereq: 
  For cluster: Build headless binary first
    mkdir -p build && cd build && cmake .. -DBUILD_HEADLESS=ON && cmake --build . -j
  
  For local: Build debug binary
    mkdir -p build-debug && cd build-debug && cmake .. && make -j
"""

from pathlib import Path
from datetime import datetime
import shutil, subprocess, os, stat, sys, json, copy
import argparse

# ------------------- USER CONFIG -------------------

# Length-to-diameter ratios to test
ASPECT_RATIOS = [10, 50, 100, 200, 500]
ROD_LENGTH = 1.0  # Fixed rod length
SYSTEM_SIZE = 1.1  # Linear size of the system (L)

# Optional: sweep friction as well
# FRICTION_COEFFS = [0.2]  # Single friction value, or add more: [0.0, 0.1, 0.2, 0.4]

FRICTION_COEFFS = [0.0, 0.05, 0.1, 0.2, 0.4]
NOISE_AMPLITUDES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# Fixed noise amplitude
# NOISE_AMPLITUDE = 1e-3

# Calculate number of rods based on aspect ratio
# Formula: N/L^3 * (l/AR) * l^2 = 6
# Solving for N: N = 6 * L^3 * AR / l^3
def calculate_n_rods(aspect_ratio, rod_length=ROD_LENGTH, system_size=SYSTEM_SIZE):
    """
    Calculate number of rods to maintain constant number density.
    
    Formula: N/L^3 * (l/AR) * l^2 = 6
    Rearranging: N = 6 * L^3 * AR / l^3
    
    Args:
        aspect_ratio: L/D ratio
        rod_length: length of each rod (l)
        system_size: linear size of periodic box (L)
    
    Returns:
        Number of rods (rounded to nearest integer)
    """
    N = 6.0 * (system_size ** 3) * aspect_ratio / (rod_length ** 3)
    return int(round(N))

N_RODS = 200  # Default, will be overridden by calculate_n_rods() in param generation
STEPS = 20000
OUTPUT_INTERVAL = 100

# For local debug, use fewer steps
LOCAL_DEBUG_STEPS = 10000
LOCAL_DEBUG_OUTPUT_INTERVAL = 10

# SLURM defaults
SLURM = {
    "partition":  "seas_compute",
    "time":       "0-12:00",
    "mem":        "2000",
    "ntasks":     1,
    "cpus":       4,
    "nodes":      1,
    "mail_user":  os.environ.get("USER_EMAIL", ""),
    "mail_type":  "END",
    "module_line":"module load python",
    "conda_env":  "mujoco-env",
}

BASE_SCENE = {
    "scene": {
        "bodies": [
            {
                "length": 1.0,        # will be set to ROD_LENGTH
                "diameter": 0.05,     # will vary based on aspect ratio
                "density": 1000.0,
                "restitution": 1.0,
                "friction": 0.2,      # will vary if multiple friction coeffs
                "friction_s": 0.2,
                "friction_d": 0.2
            }
        ],
        "populate": {
            "count": N_RODS,
            "mode": "nonoverlap",
            "spacingMul": 2.0,
            "seed": 12345,
            "maxAttempts": 100000
        },
        "periodic": {
            "enabled": True,
            "min": [-0.55, -0.55, -0.55],
            "max": [ 0.55,  0.55,  0.55],
            "cellSize": 2.0
        },
        "randomInit": {
            "enabled": False,
            "vSigma": 0.1,
            "wSpeed": 0.1,
            "seed": 42
        },
        "randomForce": {
            "enabled": True,
            "fSigma": 0.001,      # will be set to NOISE_AMPLITUDE
            "tauMag": 0.001,
            "seed": 123
        }
    },
    "physics": {
        "dt": 0.001,
        "gravity": [0.0, 0.0, 0.0],
        "lin_damp": 0.0,
        "ang_damp": 0.0,
        "soft_contact": {
            "enabled": True,
            "k_scaler": 1e1,
            "delta": 0.0002,
            "nu": 0.1,
            "mu": 0.2,            # will vary if multiple friction coeffs
            "enable_friction": True,
            "verbose": False
        },
        "solver": {
            "baumgarte": 0.0,
            "allowed_pen": 0.0,
            "velIters": 10,
            "split_impulse": False,
            "split_orient": True,
            "ngsNormalSweeps": 0,
            "ngsHighVThresh": 0.5
        }
    }
}

# ------------------- HELPERS -------------------

def find_root_dir(start=None, target_name="rod-dynamics-3d"):
    p = Path.cwd() if start is None else Path(start).resolve()
    for ancestor in [p, *p.parents]:
        if ancestor.name == target_name:
            return ancestor
    raise SystemExit(f"Could not find repository root named '{target_name}' starting from {p}")

def now_ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_executable(path):
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if not os.access(path, os.X_OK):
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)

def compute_param_sets():
    ps = []
    for aspect_ratio in ASPECT_RATIOS:
        diameter = ROD_LENGTH / aspect_ratio
        n_rods = calculate_n_rods(aspect_ratio)
        for friction in FRICTION_COEFFS:
            for noise in NOISE_AMPLITUDES:
                ps.append({
                    "aspect_ratio": float(aspect_ratio),
                    "length": float(ROD_LENGTH),
                    "diameter": float(diameter),
                    "friction": float(friction),
                    "noise": float(noise),
                    "n_rods": n_rods,
                    "steps": STEPS,
                    "output_interval": OUTPUT_INTERVAL,
                })
    return ps

def make_run_dir(runs_root, params, job_name):
    name = (
        f"{now_ts()}_RUN_aspect"
        f"_LD{params['aspect_ratio']:.0f}"
        f"_mu{params['friction']:.2f}"
        f"_noise{params['noise']:.1e}"
        f"_{job_name}"
    )
    run_dir = runs_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def generate_scene(run_dir, params):
    scene = copy.deepcopy(BASE_SCENE)
    
    # Set rod dimensions
    scene["scene"]["bodies"][0]["length"] = params["length"]
    scene["scene"]["bodies"][0]["diameter"] = params["diameter"]
    
    # Set number of rods
    scene["scene"]["populate"]["count"] = params["n_rods"]
    
    # Set friction
    friction = params["friction"]
    scene["scene"]["bodies"][0]["friction"] = friction
    scene["scene"]["bodies"][0]["friction_s"] = friction
    scene["scene"]["bodies"][0]["friction_d"] = friction
    scene["physics"]["soft_contact"]["mu"] = friction
    
    # Set noise
    noise = params["noise"]
    scene["scene"]["randomForce"]["fSigma"] = noise
    scene["scene"]["randomForce"]["tauMag"] = noise

    scene_path = run_dir / "scene.json"
    save_json(scene, scene_path)
    return scene_path

def write_readme_and_options(run_dir, params, sim_cmd):
    (run_dir / "README.txt").write_text(
        "Aspect Ratio Parametric Sweep\n"
        "==============================\n\n"
        "Parameters:\n" + "\n".join(f"  {k}: {v}" for k, v in params.items()) + 
        f"\n\nSimulation Command:\n  {sim_cmd}\n"
    )
    with open(run_dir / "options.txt", "w") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

def write_sbatch(run_dir, sim_cmd, params):
    post_py = r"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

csv = 'profile.csv'
if os.path.exists(csv):
    df = pd.read_csv(csv)
    os.makedirs('figs', exist_ok=True)
    
    # KE plot
    if 'frame' in df.columns and 'KE' in df.columns:
        plt.figure(figsize=(10,6))
        plt.semilogy(df['frame'], df['KE'])
        plt.xlabel('Frame')
        plt.ylabel('Kinetic Energy (J)')
        plt.title(f'KE vs Frame (L/D={params['aspect_ratio']:.0f})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figs/ke.png')
        plt.close()
        
        # Compute statistics
        ke_mean = df['KE'].iloc[len(df)//2:].mean()
        ke_std = df['KE'].iloc[len(df)//2:].std()
        
        # Linear fit in log space (growth rate)
        try:
            from scipy.optimize import curve_fit
            t = df['frame'].to_numpy(dtype=float)
            y = df['KE'].to_numpy(dtype=float)
            
            # Use middle 60% of data
            start_idx = int(0.3 * len(t))
            end_idx = int(0.9 * len(t))
            t_fit = t[start_idx:end_idx]
            y_fit = y[start_idx:end_idx]
            
            valid = y_fit > 0
            if np.sum(valid) > 10:
                log_ke = np.log(y_fit[valid])
                t_fit = t_fit[valid]
                coeffs = np.polyfit(t_fit, log_ke, 1)
                growth_rate = coeffs[0]
            else:
                growth_rate = np.nan
        except Exception as e:
            growth_rate = np.nan
        
        with open('analysis.txt', 'w') as f:
            f.write('Aspect Ratio Study Analysis\n')
            f.write('=' * 40 + '\n\n')
            f.write(f'Aspect ratio (L/D): {params['aspect_ratio']:.1f}\n')
            f.write(f'Length: {params['length']:.4f}\n')
            f.write(f'Diameter: {params['diameter']:.6f}\n')
            f.write(f'Mean KE (latter half): {ke_mean:.6e}\n')
            f.write(f'Std KE (latter half): {ke_std:.6e}\n')
            f.write(f'Growth rate: {growth_rate:.6e}\n')
    
    # Contact count plot if available
    if 'n_contacts' in df.columns:
        plt.figure(figsize=(10,6))
        plt.plot(df['frame'], df['n_contacts'])
        plt.xlabel('Frame')
        plt.ylabel('Number of Contacts')
        plt.title(f'Contact Count vs Frame (L/D={params['aspect_ratio']:.0f})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figs/contacts.png')
        plt.close()

print('Post-analysis complete.')
"""
    
    sb = f"""#!/bin/bash
#SBATCH -n {SLURM['ntasks']}
#SBATCH -c {SLURM['cpus']}
#SBATCH -N {SLURM['nodes']}
#SBATCH -t {SLURM['time']}
#SBATCH -p {SLURM['partition']}
#SBATCH --mem={SLURM['mem']}
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err
#SBATCH --mail-type={SLURM['mail_type']}
{f"#SBATCH --mail-user={SLURM['mail_user']}" if SLURM['mail_user'] else ""}
#SBATCH --job-name=LD{params['aspect_ratio']:.0f}_mu{params['friction']:.2f}_n{params['noise']:.1e}

set -euo pipefail
{SLURM['module_line']}
{('mamba activate '+SLURM['conda_env']) if SLURM['conda_env'] else "# (no conda activation)"}

echo "======================================"
echo "Aspect Ratio Parametric Sweep"
echo "======================================"
echo "Aspect ratio (L/D): {params['aspect_ratio']}"
echo "Rod length: {params['length']}"
echo "Rod diameter: {params['diameter']}"
echo "Friction coefficient: {params['friction']}"
echo "Noise amplitude: {params['noise']}"
echo "Number of rods: {params['n_rods']}"
echo "PWD: $(pwd)"
echo "======================================"
echo ""

# --- run simulation ---
echo "Running simulation..."
{sim_cmd}

echo ""
echo "Simulation complete."
echo ""

# --- post-analysis (inline) ---
echo "Running post-analysis..."
python3 - <<'PY'
{post_py}
PY

echo "Post-analysis complete."
echo "======================================"
"""
    sbatch_path = run_dir / "Sbatch.sh"
    sbatch_path.write_text(sb)
    os.chmod(sbatch_path, 0o755)
    return sbatch_path

def submit(run_dir):
    print(f"  Submitting: {run_dir.name}")
    subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir, check=True)

def run_local(run_dir, params):
    """Run simulation locally without SLURM."""
    print(f"\n  Running locally in: {run_dir.name}")
    print(f"  Aspect ratio (L/D): {params['aspect_ratio']}")
    print(f"  Friction: {params['friction']}")
    print(f"  Noise: {params['noise']}")
    print(f"  Diameter: {params['diameter']:.6f}")
    print()
    
    # Use reduced steps for local debug
    local_steps = LOCAL_DEBUG_STEPS
    local_output_interval = LOCAL_DEBUG_OUTPUT_INTERVAL
    
    print(f"  Using reduced steps for local debug: {local_steps} (output every {local_output_interval})")
    print()
    
    # Build command
    sim_cmd = [
        "./rigidbody_viewer_3d",
        "--headless",
        "--scene", "scene.json",
        "--steps", str(local_steps),
        "--output-interval", str(local_output_interval),
        "--csv", "profile.csv",
        "--com", "com.csv",
        "--network", "network.csv",
        "--threads", "4"
    ]
    
    print(f"  Command: {' '.join(sim_cmd)}")
    print()
    
    # Run simulation
    print("=" * 60)
    result = subprocess.run(sim_cmd, cwd=run_dir)
    print("=" * 60)
    
    if result.returncode != 0:
        print(f"\n  ERROR: Simulation failed with exit code {result.returncode}")
        return False
    
    print("\n  Simulation complete. Running post-analysis...")
    
    # Run post-analysis
    post_analysis_local(run_dir, params)
    
    print(f"\n  Results saved in: {run_dir}")
    print(f"    - profile.csv")
    print(f"    - com.csv")
    print(f"    - network.csv")
    print(f"    - figs/ke.png")
    print(f"    - figs/com_evolution.png")
    print(f"    - figs/com_displacement.png")
    print(f"    - figs/contact_count.png")
    print(f"    - figs/contact_frequency.png")
    print(f"    - figs/contact_spatial.png")
    print(f"    - analysis.txt")
    
    return True

def post_analysis_local(run_dir, params):
    """Run post-analysis locally using inline Python."""
    csv_path = run_dir / "profile.csv"
    if not csv_path.exists():
        print("  Warning: profile.csv not found, skipping post-analysis")
        return
    
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    df = pd.read_csv(csv_path)
    figs_dir = run_dir / 'figs'
    figs_dir.mkdir(exist_ok=True)
    
    # KE plot
    if 'frame' in df.columns and 'KE' in df.columns:
        plt.figure(figsize=(10,6))
        plt.semilogy(df['frame'], df['KE'])
        plt.xlabel('Frame')
        plt.ylabel('Kinetic Energy (J)')
        plt.title(f'KE vs Frame (L/D={params["aspect_ratio"]:.0f}, μ={params["friction"]:.2f}, σ={params["noise"]:.1e})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figs_dir / 'ke.png', dpi=150)
        plt.close()
        
        # Compute statistics
        ke_mean = df['KE'].iloc[len(df)//2:].mean()
        ke_std = df['KE'].iloc[len(df)//2:].std()
        
        # Linear fit in log space (growth rate)
        try:
            from scipy.optimize import curve_fit
            t = df['frame'].to_numpy(dtype=float)
            y = df['KE'].to_numpy(dtype=float)
            
            # Use middle 60% of data
            start_idx = int(0.3 * len(t))
            end_idx = int(0.9 * len(t))
            t_fit = t[start_idx:end_idx]
            y_fit = y[start_idx:end_idx]
            
            valid = y_fit > 0
            if np.sum(valid) > 10:
                log_ke = np.log(y_fit[valid])
                t_fit = t_fit[valid]
                coeffs = np.polyfit(t_fit, log_ke, 1)
                growth_rate = coeffs[0]
            else:
                growth_rate = np.nan
        except Exception as e:
            growth_rate = np.nan
        
        with open(run_dir / 'analysis.txt', 'w') as f:
            f.write('Aspect Ratio Study Analysis\n')
            f.write('=' * 40 + '\n\n')
            f.write(f'Aspect ratio (L/D): {params["aspect_ratio"]:.1f}\n')
            f.write(f'Length: {params["length"]:.4f}\n')
            f.write(f'Diameter: {params["diameter"]:.6f}\n')
            f.write(f'Friction: {params["friction"]:.2f}\n')
            f.write(f'Noise: {params["noise"]:.1e}\n')
            f.write(f'Mean KE (latter half): {ke_mean:.6e}\n')
            f.write(f'Std KE (latter half): {ke_std:.6e}\n')
            f.write(f'Growth rate: {growth_rate:.6e}\n')
        
        print(f"  Mean KE: {ke_mean:.6e}, Growth rate: {growth_rate:.6e}")
    
    # Contact count plot if available
    if 'n_contacts' in df.columns:
        plt.figure(figsize=(10,6))
        plt.plot(df['frame'], df['n_contacts'])
        plt.xlabel('Frame')
        plt.ylabel('Number of Contacts')
        plt.title(f'Contact Count vs Frame (L/D={params["aspect_ratio"]:.0f})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figs_dir / 'contacts.png', dpi=150)
        plt.close()
    
    # COM analysis if available
    com_path = run_dir / 'com.csv'
    if com_path.exists():
        com_df = pd.read_csv(com_path)
        
        # Plot COM position evolution
        if all(col in com_df.columns for col in ['frame', 'com_x', 'com_y', 'com_z']):
            fig, axes = plt.subplots(3, 1, figsize=(10, 10))
            
            axes[0].plot(com_df['frame'], com_df['com_x'], 'r-', linewidth=0.5)
            axes[0].set_ylabel('COM X')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title(f'Center of Mass Evolution (L/D={params["aspect_ratio"]:.0f}, μ={params["friction"]:.2f}, σ={params["noise"]:.1e})')
            
            axes[1].plot(com_df['frame'], com_df['com_y'], 'g-', linewidth=0.5)
            axes[1].set_ylabel('COM Y')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(com_df['frame'], com_df['com_z'], 'b-', linewidth=0.5)
            axes[2].set_ylabel('COM Z')
            axes[2].set_xlabel('Frame')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(figs_dir / 'com_evolution.png', dpi=150)
            plt.close()
            
            # Plot COM displacement from initial position
            com_x0 = com_df['com_x'].iloc[0]
            com_y0 = com_df['com_y'].iloc[0]
            com_z0 = com_df['com_z'].iloc[0]
            
            displacement = np.sqrt(
                (com_df['com_x'] - com_x0)**2 + 
                (com_df['com_y'] - com_y0)**2 + 
                (com_df['com_z'] - com_z0)**2
            )
            
            plt.figure(figsize=(10, 6))
            plt.plot(com_df['frame'], displacement, 'k-', linewidth=0.8)
            plt.xlabel('Frame')
            plt.ylabel('COM Displacement from Initial')
            plt.title(f'COM Displacement (L/D={params["aspect_ratio"]:.0f}, μ={params["friction"]:.2f}, σ={params["noise"]:.1e})')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(figs_dir / 'com_displacement.png', dpi=150)
            plt.close()
            
            # Compute COM statistics
            com_displacement_mean = displacement.mean()
            com_displacement_std = displacement.std()
            com_displacement_max = displacement.max()
            
            # Append to analysis file
            with open(run_dir / 'analysis.txt', 'a') as f:
                f.write('\n\nCenter of Mass Analysis\n')
                f.write('=' * 40 + '\n')
                f.write(f'Initial position: ({com_x0:.6f}, {com_y0:.6f}, {com_z0:.6f})\n')
                f.write(f'Final position: ({com_df["com_x"].iloc[-1]:.6f}, {com_df["com_y"].iloc[-1]:.6f}, {com_df["com_z"].iloc[-1]:.6f})\n')
                f.write(f'Mean displacement: {com_displacement_mean:.6e}\n')
                f.write(f'Std displacement: {com_displacement_std:.6e}\n')
                f.write(f'Max displacement: {com_displacement_max:.6e}\n')
            
            print(f"  COM displacement - mean: {com_displacement_mean:.6e}, max: {com_displacement_max:.6e}")
    
    # Contact network analysis if available
    network_path = run_dir / 'network.csv'
    if network_path.exists():
        network_df = pd.read_csv(network_path)
        
        if len(network_df) > 0 and 'frame' in network_df.columns:
            # Count contacts per frame
            contacts_per_frame = network_df.groupby('frame').size().reset_index(name='num_contacts')
            
            # Plot contact count evolution
            plt.figure(figsize=(10, 6))
            plt.plot(contacts_per_frame['frame'], contacts_per_frame['num_contacts'], 'b-', linewidth=0.8)
            plt.xlabel('Frame')
            plt.ylabel('Number of Contacts')
            plt.title(f'Contact Network Size (L/D={params["aspect_ratio"]:.0f}, μ={params["friction"]:.2f}, σ={params["noise"]:.1e})')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(figs_dir / 'contact_count.png', dpi=150)
            plt.close()
            
            # Analyze contact pairs
            if all(col in network_df.columns for col in ['rod_i', 'rod_j']):
                # Create unique contact pairs (unordered)
                network_df['pair'] = network_df.apply(
                    lambda row: tuple(sorted([row['rod_i'], row['rod_j']])), axis=1
                )
                
                # Count how many times each pair contacts
                pair_contacts = network_df.groupby('pair').size().reset_index(name='contact_count')
                pair_contacts = pair_contacts.sort_values('contact_count', ascending=False)
                
                # Plot contact frequency distribution
                plt.figure(figsize=(10, 6))
                plt.hist(pair_contacts['contact_count'], bins=50, edgecolor='black', alpha=0.7)
                plt.xlabel('Number of Frames in Contact')
                plt.ylabel('Number of Rod Pairs')
                plt.title(f'Contact Frequency Distribution (L/D={params["aspect_ratio"]:.0f})')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(figs_dir / 'contact_frequency.png', dpi=150)
                plt.close()
                
                # Compute network statistics
                total_contacts = len(network_df)
                unique_pairs = len(pair_contacts)
                mean_contacts_per_frame = contacts_per_frame['num_contacts'].mean()
                max_contacts_per_frame = contacts_per_frame['num_contacts'].max()
                
                # Most frequent contacts (top 10)
                top_pairs = pair_contacts.head(10)
                
                # Append to analysis file
                with open(run_dir / 'analysis.txt', 'a') as f:
                    f.write('\n\nContact Network Analysis\n')
                    f.write('=' * 40 + '\n')
                    f.write(f'Total contact events: {total_contacts}\n')
                    f.write(f'Unique rod pairs in contact: {unique_pairs}\n')
                    f.write(f'Mean contacts per frame: {mean_contacts_per_frame:.2f}\n')
                    f.write(f'Max contacts per frame: {max_contacts_per_frame}\n')
                    f.write(f'\nTop 10 most frequent contact pairs:\n')
                    for idx, row in top_pairs.iterrows():
                        pair = row['pair']
                        count = row['contact_count']
                        f.write(f'  Pair ({pair[0]}, {pair[1]}): {count} frames\n')
                    
                    # Additional contact info if available
                    if all(col in network_df.columns for col in ['contact_x', 'contact_y', 'contact_z']):
                        f.write(f'\nContact spatial information available\n')
                    if all(col in network_df.columns for col in ['normal_x', 'normal_y', 'normal_z']):
                        f.write(f'Contact normal vectors available\n')
                    if 'distance' in network_df.columns:
                        mean_dist = network_df['distance'].mean()
                        f.write(f'Mean contact distance: {mean_dist:.6e}\n')
                
                print(f"  Contact network - unique pairs: {unique_pairs}, mean contacts/frame: {mean_contacts_per_frame:.2f}")
            
            # Visualize contact locations if available
            if all(col in network_df.columns for col in ['contact_x', 'contact_y', 'contact_z']) and len(network_df) > 0:
                fig = plt.figure(figsize=(12, 4))
                
                # XY projection
                ax1 = fig.add_subplot(131)
                ax1.scatter(network_df['contact_x'], network_df['contact_y'], alpha=0.3, s=1)
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
                ax1.set_title('Contact Positions (XY)')
                ax1.grid(True, alpha=0.3)
                ax1.set_aspect('equal')
                
                # XZ projection
                ax2 = fig.add_subplot(132)
                ax2.scatter(network_df['contact_x'], network_df['contact_z'], alpha=0.3, s=1)
                ax2.set_xlabel('X')
                ax2.set_ylabel('Z')
                ax2.set_title('Contact Positions (XZ)')
                ax2.grid(True, alpha=0.3)
                ax2.set_aspect('equal')
                
                # YZ projection
                ax3 = fig.add_subplot(133)
                ax3.scatter(network_df['contact_y'], network_df['contact_z'], alpha=0.3, s=1)
                ax3.set_xlabel('Y')
                ax3.set_ylabel('Z')
                ax3.set_title('Contact Positions (YZ)')
                ax3.grid(True, alpha=0.3)
                ax3.set_aspect('equal')
                
                plt.tight_layout()
                plt.savefig(figs_dir / 'contact_spatial.png', dpi=150)
                plt.close()
        else:
            print(f"  No contacts detected in simulation")

# ------------------- MAIN -------------------

def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for aspect ratio parametric sweep.")
    parser.add_argument('--job-name', type=str, default='aspect_ratio', help='Job name for organizing runs.')
    parser.add_argument('--dry-run', action='store_true', help='Generate files but do not submit jobs.')
    parser.add_argument('--local', action='store_true', help='Run locally for debugging (picks one random parameter set).')
    args = parser.parse_args()

    root_dir = find_root_dir()
    
    # Choose runs directory based on mode
    if args.local:
        runs_root = root_dir / "runs" / f"{args.job_name}_local"
    else:
        runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs") / args.job_name
    runs_root.mkdir(parents=True, exist_ok=True)

    # Choose binary based on mode
    if args.local:
        binary_src = root_dir / "build-debug" / "rigidbody_viewer_3d"
    else:
        binary_src = root_dir / "build" / "rigidbody_viewer_3d"
    ensure_executable(binary_src)

    param_sets = compute_param_sets()
    
    # If local mode, pick one random parameter set
    if args.local:
        import random
        param_sets = [random.choice(param_sets)]
        print(f"\n*** LOCAL DEBUG MODE: Running single parameter set ***\n")

    print("=" * 80)
    print("ASPECT RATIO PARAMETRIC SWEEP SUBMISSION")
    print("=" * 80)
    print(f"Job name: {args.job_name}")
    print(f"Runs directory: {runs_root}")
    print(f"System size: {SYSTEM_SIZE}")
    print(f"Rod length (fixed): {ROD_LENGTH}")
    print(f"\nAspect ratios and corresponding rod counts:")
    print(f"  Formula: N/L³ * (l/AR) * l² = 6  =>  N = 6*L³*AR/l³")
    for ar in ASPECT_RATIOS:
        n = calculate_n_rods(ar)
        d = ROD_LENGTH / ar
        print(f"  L/D = {ar:4d}  ->  N = {n:4d}  (diameter = {d:.6f})")
    print(f"\nFriction coefficients: {FRICTION_COEFFS}")
    print(f"Noise amplitudes: {NOISE_AMPLITUDES}")
    print(f"Total parameter combinations: {len(param_sets)}")
    print(f"Steps per run: {STEPS}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)
    print()

    run_dirs = []
    for i, params in enumerate(param_sets, 1):
        print(f"[{i}/{len(param_sets)}] Setting up: L/D={params['aspect_ratio']:.0f}, μ={params['friction']:.2f}, σ={params['noise']:.1e}, D={params['diameter']:.6f}")
        
        run_dir = make_run_dir(runs_root, params, args.job_name)
        run_dirs.append(run_dir)

        # copy binary
        binary_dst = run_dir / "rigidbody_viewer_3d"
        shutil.copy2(binary_src, binary_dst)
        os.chmod(binary_dst, 0o755)

        # copy this submission script
        shutil.copy2(Path(__file__), run_dir / Path(__file__).name)

        # generate scene.json in run dir
        scene_path = generate_scene(run_dir, params)

        # build simulation command
        sim_cmd = (
            f"./rigidbody_viewer_3d "
            f"--headless "
            f"--scene scene.json "
            f"--steps {int(params['steps'])} "
            f"--output-interval {int(params['output_interval'])} "
            f"--csv profile.csv "
            f"--com com.csv "
            f"--network network.csv "
            f"--threads ${{SLURM_CPUS_PER_TASK:-{SLURM['cpus']}}}"
        )

        # write docs and sbatch
        write_readme_and_options(run_dir, params, sim_cmd)
        
        # Execute based on mode
        if args.local:
            # Run locally without SLURM
            if not run_local(run_dir, params):
                print("  Local execution failed!")
                sys.exit(1)
        else:
            # Write sbatch and submit to cluster
            write_sbatch(run_dir, sim_cmd, params)
            if not args.dry_run:
                submit(run_dir)
            else:
                print(f"  [DRY RUN] Would submit: {run_dir / 'Sbatch.sh'}")

    # Save list of run directories
    run_dirs_file = runs_root / "run_dirs.txt"
    with open(run_dirs_file, 'w') as f:
        for rd in run_dirs:
            f.write(str(rd) + '\n')
    print()
    print(f"Saved run directories to: {run_dirs_file}")

    # Save parameter summary
    summary_file = runs_root / "parameter_summary.json"
    summary = {
        "job_name": args.job_name,
        "aspect_ratios": ASPECT_RATIOS,
        "rod_length": ROD_LENGTH,
        "rod_diameters": [ROD_LENGTH/ar for ar in ASPECT_RATIOS],
        "friction_coeffs": FRICTION_COEFFS,
        "noise_amplitudes": NOISE_AMPLITUDES,
        "n_rods": N_RODS,
        "steps": STEPS,
        "output_interval": OUTPUT_INTERVAL,
        "total_runs": len(param_sets),
        "run_dirs": [str(rd) for rd in run_dirs]
    }
    save_json(summary, summary_file)
    print(f"Saved parameter summary to: {summary_file}")

    # Save combined analysis command
    combined_cmd = f"python3 post_analyze_aspect_ratio.py --job-name {args.job_name} --make-plots --outdir analysis_aspect_{args.job_name}"
    combined_cmd_file = runs_root / "combined_analysis_command.txt"
    with open(combined_cmd_file, 'w') as f:
        f.write(combined_cmd + '\n')
    print(f"Saved combined analysis command to: {combined_cmd_file}")
    
    print()
    print("=" * 80)
    if args.local:
        print("LOCAL DEBUG MODE COMPLETE")
        print(f"Results directory: {runs_root}")
        print(f"Check the analysis.txt and figs/ directory in the run folder")
    elif args.dry_run:
        print("DRY RUN COMPLETE - No jobs submitted")
    else:
        print(f"SUBMISSION COMPLETE - {len(param_sets)} jobs submitted")
        print(f"Monitor with: squeue -u $USER")
        print(f"After completion, run: {combined_cmd}")
    print("=" * 80)

if __name__ == "__main__":
    main()
