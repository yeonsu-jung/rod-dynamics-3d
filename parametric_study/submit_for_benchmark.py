#!/usr/bin/env python3
"""
submit_for_benchmark.py

Submit SLURM jobs to test timestep effects.
Runs simulations with different dt and step counts to cover the same physical time.
Separates execution directory (home) from data storage (lab).
"""

from pathlib import Path
from datetime import datetime
import shutil, subprocess, os, stat, sys, json
import argparse

# Test cases: dt and corresponding steps for constant physical time
TEST_CASES = [
    {"dt": 1e-5, "steps": 1_000_000_000},
    # {"dt": 1e-6, "steps": 1_000_000_000},
    # {"dt": 1e-6, "steps": 1000},
    # {"dt": 1e-7, "steps": 1_000_000},
]

# SLURM defaults
SLURM = {
    "partition":  "seas_compute",
    "time":       "7-00:00", # Increased time for longer runs
    "mem":        "100",
    "ntasks":     1,
    "cpus":       1,
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
    parser = argparse.ArgumentParser(description="Submit timestep test jobs.")
    parser.add_argument('--job-name', type=str, default='benchmark6_after_recompile', help='Job name prefix.')
    parser.add_argument('--dry-run', action='store_true', help='Generate files but do not submit.')
    args = parser.parse_args()

    root_dir = find_root_dir()
    
    # 1. Execution Directory (Home) - Lightweight
    home_runs_root = root_dir / "runs" / args.job_name
    home_runs_root.mkdir(parents=True, exist_ok=True)

    # 2. Data Directory (Lab) - Heavyweight
    lab_runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs") / args.job_name
    lab_runs_root.mkdir(parents=True, exist_ok=True)
    
    timestamp = now_ts()
    print(f"Submitting jobs:")
    print(f"  Execution Dir: {home_runs_root}")
    print(f"  Data Dir:      {lab_runs_root}")

    # Source Paths
    binary_src = root_dir / "build" / "rigidbody_viewer_3d"
    scene_src = root_dir / "assets/scenes/default_with_csv.json"
    csv_src = root_dir / "initial-configs/rods_bf_1132.csv"

    ensure_executable(binary_src)
    if not scene_src.exists():
        raise SystemExit(f"Scene file not found: {scene_src}")
    if not csv_src.exists():
        raise SystemExit(f"CSV file not found: {csv_src}")

    # Load base scene
    with open(scene_src, 'r') as f:
        base_scene = json.load(f)

    for case in TEST_CASES:
        dt = case['dt']
        steps = case['steps']
        
        # Create unique run name
        run_name = f"{timestamp}_dt{dt:.0e}_steps{steps}"
        
        # Create directories
        home_run_dir = home_runs_root / run_name
        home_run_dir.mkdir(parents=True, exist_ok=True)
        
        lab_run_dir = lab_runs_root / run_name
        lab_run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Setting up: dt={dt}, steps={steps}")

        # Copy CSV to HOME (Execution dir)
        # shutil.copy2(binary_src, home_run_dir / "rigidbody_viewer_3d")
        shutil.copy2(csv_src, home_run_dir / "init_config.csv")

        # Modify and save scene to HOME
        scene = base_scene.copy()
        if 'physics' not in scene:
            scene['physics'] = {}
        scene['physics']['dt'] = dt
        
        with open(home_run_dir / "scene.json", 'w') as f:
            json.dump(scene, f, indent=2)

        # Calculate per-rod max frames
        perrod_stride = 30000
        perrod_max = max(1, steps // perrod_stride)

        # Define output paths in LAB directory
        output_csv = lab_run_dir / "output.csv"
        perrod_csv = lab_run_dir / "perrod.csv"
        com_csv    = lab_run_dir / "com_debug.csv"

        # Build Command
        # Note: We run in home_run_dir, but point outputs to lab_run_dir
        sim_cmd = (
            "time "
            f"{binary_src} "            
            "--scene scene.json "
            "--init-csv init_config.csv "
            f"--output {output_csv} "
            # f"--com {com_csv} "  # Using --com instead of --debug-com to specify path
            # "--profile "
            "--use-spatial-hash "
            f"--perrod {perrod_csv} "
            f"--perrod-max {perrod_max} "
            f"--perrod-stride {perrod_stride} "
            f"--steps {steps} "
            f"--threads {SLURM['cpus']}"
        )

        # Create Sbatch script in HOME
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
#SBATCH --job-name={args.job_name}_dt{dt:.0e}

set -euo pipefail
{SLURM['module_line']}

# Ensure we are in the execution directory
cd {home_run_dir}

echo "======================================"
echo "Timestep Test: dt={dt}, steps={steps}"
echo "Execution Dir: $(pwd)"
echo "Data Dir:      {lab_run_dir}"
echo "======================================"

echo "Running simulation..."
echo "{sim_cmd}"
{sim_cmd}

echo ""
echo "Job complete."
"""
        
        sbatch_path = home_run_dir / "Sbatch.sh"
        sbatch_path.write_text(sb)
        os.chmod(sbatch_path, 0o755)

        if not args.dry_run:
            subprocess.run(["sbatch", "Sbatch.sh"], cwd=home_run_dir, check=True)
            print(f"    Submitted job for dt={dt}")
        else:
            print(f"    [Dry Run] Created Sbatch.sh")

    print("\nAll jobs processed.")

if __name__ == "__main__":
    main()


