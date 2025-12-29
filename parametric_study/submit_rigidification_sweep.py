#!/usr/bin/env python3
"""
submit_rigidification_sweep.py

Submit SLURM jobs to find subsets of parameters related to rigidification.
Related parameters include lin/ang dampling, random force, aspect ratio, density.
"""


from pathlib import Path
from datetime import datetime
import shutil, subprocess, os, stat, sys, json
import argparse
import numpy as np

# Test cases: dt and corresponding steps for constant physical time
# "fSigma": 1.0,
#   "tauMag": 0.0,

# Generate fSigma values from 1e-2 to 1e0
f_sigmas = np.geomspace(1e-4, 1e0, num=5)
# f_sigmas = np.geomspace(1e-3, 1e1, num=20)
# f_sigmas = np.array([1e-1])

TEST_CASES = []
for f in f_sigmas:
    TEST_CASES.append({
        "dt": 0.0005, 
        "steps": 100000,
        "lin_damp": 0.0,
        "ang_damp": 0.0, 
        "fSigma": float(f), 
        "tauMag": 0.0, 
        "aspect_ratio": 100.0, 
        "num_rods": 3375,
        "density": 2500.0
    })


# SLURM defaults
SLURM = {
    "partition":  "seas_compute",
    "time":       "7-00:00", # Increased time for longer runs
    "mem":        "100",
    "ntasks":     1,
    "cpus":       8,
    "nodes":      1,
    "mail_user":  os.environ.get("USER_EMAIL", ""),
    "mail_type":  "END",
    "module_line":"module load python",
}

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

def main():
    parser = argparse.ArgumentParser(description="Submit rigidification sweep jobs.")
    parser.add_argument('--job-name', type=str, default='rigid_sweep', help='Job name prefix.')
    parser.add_argument('--dry-run', action='store_true', help='Generate files but do not submit.')
    args = parser.parse_args()

    root_dir = find_root_dir()
    # Target runs directory in holylabs
    runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs") / args.job_name
    runs_root.mkdir(parents=True, exist_ok=True)
    
    # copy this file to runs_root for reference
    shutil.copy2(Path(__file__), runs_root / Path(__file__).name)
    
    timestamp = now_ts()
    print(f"Submitting rigidification sweep jobs to: {runs_root}")

    # Source Paths
    binary_src = root_dir / "build" / "rigidbody_viewer_3d"
    # Use experiment_bigger_mu_0.2.json as base
    scene_src = root_dir / "assets/scenes/experiment_bigger_mu_0.2.json"

    ensure_executable(binary_src)
    if not scene_src.exists():
        raise SystemExit(f"Scene file not found: {scene_src}")

    # Load base scene
    with open(scene_src, 'r') as f:
        base_scene = json.load(f)

    for i, case in enumerate(TEST_CASES):
        dt = case['dt']
        steps = case['steps']
        ar = case['aspect_ratio']
        fSigma = case['fSigma']
        
        # Create unique run dir
        run_name = f"{timestamp}_run{i}_dt{dt:.0e}_ar{ar}_fSig{fSigma:.1e}"
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Setting up: dt={dt}, ar={ar}, fSigma={fSigma:.1e} in {run_dir.name}")

        # Copy binary
        shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")

        # Modify and save scene
        scene = base_scene.copy()
        if 'physics' not in scene:
            scene['physics'] = {}
        scene['physics']['dt'] = dt
        scene['physics']['lin_damp'] = case['lin_damp']
        scene['physics']['ang_damp'] = case['ang_damp']
        
        if 'scene' not in scene:
            scene['scene'] = {}
            
        if 'randomForce' not in scene['scene']:
            scene['scene']['randomForce'] = {}
        scene['scene']['randomForce']['fSigma'] = case['fSigma']
        scene['scene']['randomForce']['tauMag'] = case['tauMag']
        
        if 'populate' not in scene['scene']:
            scene['scene']['populate'] = {}
        scene['scene']['populate']['count'] = case['num_rods']
        scene['scene']['populate']['density'] = case['density']
        
        # Calculate length from aspect ratio
        # Default radius is 0.005 (D=0.01) in default.json
        radius = scene['scene']['populate'].get('radius', 0.005)
        
        # If bodies template exists, use its diameter
        if 'bodies' in scene['scene'] and len(scene['scene']['bodies']) > 0:
            if 'diameter' in scene['scene']['bodies'][0]:
                radius = scene['scene']['bodies'][0]['diameter'] / 2.0

        diameter = 2.0 * radius
        length = case['aspect_ratio'] * diameter
        scene['scene']['populate']['length'] = length
        
        # Update bodies template if it exists
        if 'bodies' in scene['scene'] and len(scene['scene']['bodies']) > 0:
            scene['scene']['bodies'][0]['length'] = length
            scene['scene']['bodies'][0]['diameter'] = diameter
        
        with open(run_dir / "scene.json", 'w') as f:
            json.dump(scene, f, indent=2)

        # Calculate stride for ~500 frames
        target_frames = 200
        snap_stride = max(1, steps // target_frames)

        # Build Command
        sim_cmd = (
            "time "
            "./rigidbody_viewer_3d "
            "--headless "
            "--scene scene.json "
            "--output output.csv "
            f"--snap-stride {snap_stride} "
            f"--snap-frames {target_frames} "
            "--use-spatial-hash "
            f"--perrod perrod.csv "
            f"--perrod-max {target_frames} "
            f"--perrod-stride {snap_stride} "
            f"--steps {steps} "
            f"--threads {SLURM['cpus']}"
        )

        # Create Sbatch script
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
#SBATCH --job-name={args.job_name}_fSig{fSigma:.1e}

set -euo pipefail
{SLURM['module_line']}

echo "======================================"
echo "Rigidification Sweep: dt={dt}, ar={ar}, fSigma={fSigma}"
echo "PWD: $(pwd)"
echo "======================================"

echo "Running simulation..."
echo "{sim_cmd}"
{sim_cmd}

echo ""
echo "Job complete."
"""
        
        sbatch_path = run_dir / "Sbatch.sh"
        sbatch_path.write_text(sb)
        os.chmod(sbatch_path, 0o755)

        if not args.dry_run:
            subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir, check=True)
            print(f"    Submitted job for ar={ar}")
        else:
            print(f"    [Dry Run] Created Sbatch.sh")

    print("\nAll jobs processed.")

if __name__ == "__main__":
    main()
