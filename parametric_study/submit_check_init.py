#!/usr/bin/env python3
"""
submit_check_init.py

Submit a single SLURM job to check initialization non-penetration.
Creates a run directory in /n/holylabs/.../runs/, copies necessary files, and submits.
"""

from pathlib import Path
from datetime import datetime
import shutil, subprocess, os, stat, sys
import argparse

# SLURM defaults
SLURM = {
    "partition":  "seas_compute",
    "time":       "0-12:00", # Short time for check
    "mem":        "4000",
    "ntasks":     1,
    "cpus":       4,
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
    parser = argparse.ArgumentParser(description="Submit check init job.")
    parser.add_argument('--job-name', type=str, default='check_init', help='Job name.')
    parser.add_argument('--dry-run', action='store_true', help='Generate files but do not submit.')
    args = parser.parse_args()

    root_dir = find_root_dir()
    # Target runs directory in holylabs
    runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs") / args.job_name
    
    # Create unique run dir
    run_dir = runs_root / f"{now_ts()}_{args.job_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Setting up run in: {run_dir}")

    # Source Paths
    binary_src = root_dir / "build" / "rigidbody_viewer_3d"
    scene_src = root_dir / "assets/scenes/default_with_csv.json"
    csv_src = root_dir / "initial-configs/rods932_bruteforce_checked_v3.csv"

    ensure_executable(binary_src)
    if not scene_src.exists():
        raise SystemExit(f"Scene file not found: {scene_src}")
    if not csv_src.exists():
        raise SystemExit(f"CSV file not found: {csv_src}")

    # Copy files to run directory
    print("Copying files...")
    shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")
    shutil.copy2(scene_src, run_dir / "scene.json")
    shutil.copy2(csv_src, run_dir / "init_config.csv")

    # Build Command
    # Note: paths are now local to run_dir
    sim_cmd = (
        "./rigidbody_viewer_3d "
        "--headless "
        "--scene scene.json "
        "--init-csv init_config.csv "
        "--output output_checked.csv "
        "--steps 1000 "
        "--debug-min-gap "
        "--check-init-nonpenetration "
        "--snap-frames 200"
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
#SBATCH --job-name={args.job_name}

set -euo pipefail
{SLURM['module_line']}

echo "======================================"
echo "Check Init Job"
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
        print("Submitting job...")
        subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir, check=True)
        print(f"Submitted. Output will be in {run_dir}")
    else:
        print(f"Dry run: Sbatch.sh created in {run_dir} but not submitted.")

if __name__ == "__main__":
    main()
