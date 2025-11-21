#!/usr/bin/env python3
"""
submit_soft_contact.py

Submit SLURM jobs for soft contact parametric sweep: Effects of friction coefficient and noise amplitude.

- Sweeps friction (mu) and noise amplitude (fSigma), submits headless simulations as separate jobs.
- For each parameter set, creates a run directory under /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/{job_name} with a generated scene.json.
- Copies the built executable (build/rigidbody_viewer_3d) into each run directory.
- Writes an sbatch script that runs the simulation and a lightweight post-analysis.
- Submits the job via sbatch.

Usage: python submit_soft_contact.py --job-name soft_contact_test

Then, for analysis: python post_analyze_soft_contact.py --job-name soft_contact_test --make-plots --outdir analysis_soft_contact_test

Prereq: Build headless binary first
  mkdir -p build && cd build && cmake .. -DBUILD_HEADLESS=ON && cmake --build . -j
"""

from pathlib import Path
from datetime import datetime
import shutil, subprocess, os, stat, sys, json, copy
import argparse

# ------------------- USER CONFIG -------------------

FRICTION_COEFFS = [0.0, 0.05, 0.1, 0.2, 0.4]
NOISE_AMPLITUDES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
N_RODS = 200
STEPS = 100000
OUTPUT_INTERVAL = 100

# Output options
ENABLE_NETWORK_OUTPUT = False  # Set to True to enable contact network tracking (uses more memory)

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
                "length": 1.5,
                "diameter": 0.05,
                "density": 1000.0,
                "restitution": 1.0,
                "friction": 0.4,      # will vary
                "friction_s": 0.4,    # will vary
                "friction_d": 0.4     # will vary
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
            "min": [-1.0, -1.0, -1.0],
            "max": [1.0, 1.0, 1.0],
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
            "fSigma": 0.01,       # will vary
            "tauMag": 0.01,       # will vary
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
            "mu": 0.4,            # will vary
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
    for friction in FRICTION_COEFFS:
        for noise in NOISE_AMPLITUDES:
            ps.append({
                "friction": float(friction),
                "noise": float(noise),
                "n_rods": N_RODS,
                "steps": STEPS,
                "output_interval": OUTPUT_INTERVAL,
            })
    return ps

def make_run_dir(runs_root, params, job_name):
    name = (
        f"{now_ts()}_RUN_soft"
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
        "Soft Contact Model Parametric Sweep\n"
        "====================================\n\n"
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
        plt.title('KE vs Frame (Soft Contact)')
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
            f.write('Soft Contact Analysis\n')
            f.write('=' * 40 + '\n\n')
            f.write(f'Mean KE (latter half): {ke_mean:.6e}\n')
            f.write(f'Std KE (latter half): {ke_std:.6e}\n')
            f.write(f'Growth rate: {growth_rate:.6e}\n')
    
    # Contact count plot if available
    if 'n_contacts' in df.columns:
        plt.figure(figsize=(10,6))
        plt.plot(df['frame'], df['n_contacts'])
        plt.xlabel('Frame')
        plt.ylabel('Number of Contacts')
        plt.title('Contact Count vs Frame')
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
#SBATCH --job-name=soft_mu{params['friction']:.2f}_n{params['noise']:.1e}

set -euo pipefail
{SLURM['module_line']}
{('mamba activate '+SLURM['conda_env']) if SLURM['conda_env'] else "# (no conda activation)"}

echo "======================================"
echo "Soft Contact Parametric Sweep"
echo "======================================"
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

# ------------------- MAIN -------------------

def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for soft contact parametric sweep.")
    parser.add_argument('--job-name', type=str, default='soft_contact', help='Job name for organizing runs.')
    parser.add_argument('--dry-run', action='store_true', help='Generate files but do not submit jobs.')
    args = parser.parse_args()

    root_dir = find_root_dir()
    runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs") / args.job_name
    runs_root.mkdir(parents=True, exist_ok=True)

    binary_src = root_dir / "build/rigidbody_viewer_3d"
    ensure_executable(binary_src)

    param_sets = compute_param_sets()

    print("=" * 80)
    print("SOFT CONTACT PARAMETRIC SWEEP SUBMISSION")
    print("=" * 80)
    print(f"Job name: {args.job_name}")
    print(f"Runs directory: {runs_root}")
    print(f"Friction coefficients: {FRICTION_COEFFS}")
    print(f"Noise amplitudes: {NOISE_AMPLITUDES}")
    print(f"Number of rods: {N_RODS}")
    print(f"Total parameter combinations: {len(param_sets)}")
    print(f"Steps per run: {STEPS}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)
    print()

    run_dirs = []
    for i, params in enumerate(param_sets, 1):
        print(f"[{i}/{len(param_sets)}] Setting up: μ={params['friction']:.2f}, σ={params['noise']:.1e}")
        
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
        )
        if ENABLE_NETWORK_OUTPUT:
            sim_cmd += f"--network network.csv "
        sim_cmd += f"--threads ${{SLURM_CPUS_PER_TASK:-{SLURM['cpus']}}}"

        # write docs and sbatch
        write_readme_and_options(run_dir, params, sim_cmd)
        write_sbatch(run_dir, sim_cmd, params)

        # submit
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
    combined_cmd = f"python3 post_analyze_soft_contact.py --job-name {args.job_name} --make-plots --outdir analysis_soft_{args.job_name}"
    combined_cmd_file = runs_root / "combined_analysis_command.txt"
    with open(combined_cmd_file, 'w') as f:
        f.write(combined_cmd + '\n')
    print(f"Saved combined analysis command to: {combined_cmd_file}")
    
    print()
    print("=" * 80)
    if args.dry_run:
        print("DRY RUN COMPLETE - No jobs submitted")
    else:
        print(f"SUBMISSION COMPLETE - {len(param_sets)} jobs submitted")
    print(f"Monitor with: squeue -u $USER")
    print(f"After completion, run: {combined_cmd}")
    print("=" * 80)

if __name__ == "__main__":
    main()
