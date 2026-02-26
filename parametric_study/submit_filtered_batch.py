#!/usr/bin/env python3
"""submit_filtered_batch.py

Submits a batch of simulations from `initial-configs/relaxation_3rd_multithreading`.
Filters: N > 50, Alpha > 100.
Output: Minimal (No per-rod, no network, sparse output.csv).
"""

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

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

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", s)

class SlurmCfg:
    def __init__(self, partition="seas_compute", time="1-00:00", mem_gb=1, ntasks=1, cpus=4, nodes=1, mail_user=os.environ.get("USER_EMAIL", ""), mail_type="END", module_line="module load python"):
        self.partition = partition
        self.time = time
        self.mem_gb = mem_gb
        self.ntasks = ntasks
        self.cpus = cpus
        self.nodes = nodes
        self.mail_user = mail_user
        self.mail_type = mail_type
        self.module_line = module_line

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Generate files but do not submit")
    ap.add_argument("--steps", type=int, default=200000, help="Simulation steps")
    # New name for the batch folder
    ap.add_argument("--job-name", type=str, default="filtered_batch_Ngt50_ARgt100_mu1", help="Job name / output folder")
    ap.add_argument("--friction", type=float, default=1.0, help="Friction coefficient")
    args = ap.parse_args()

    root_dir = find_root_dir()
    input_root = root_dir / "initial-configs/relaxation_3rd_multithreading"
    runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs") / args.job_name

    if not input_root.exists():
        raise SystemExit(f"Input dir not found: {input_root}")

    # Binary and Scene
    binary_src = root_dir / "build" / "rigidbody_viewer_3d"
    ensure_executable(binary_src)
    
    # Use default entangled scene as base
    scene_src = root_dir / "assets/scenes/default_entangled.json"
    if not scene_src.exists():
        raise SystemExit(f"Scene not found: {scene_src}")
    
    base_scene = json.loads(scene_src.read_text())

    runs_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), runs_root / Path(__file__).name)

    timestamp = now_ts()
    
    # 1. Scan for jobs
    # Structure: input_root / Nxxx / ... / x_relaxed_ARxxx.txt
    
    jobs = []
    
    # List N folders
    for n_dir in input_root.iterdir():
        if not n_dir.is_dir() or not n_dir.name.startswith("N"):
            continue
        
        # Parse N
        try:
            n_val = int(n_dir.name[1:])
        except ValueError:
            continue
            
        # Filter N > 50
        if n_val <= 50:
            continue
            
        # Search for x_relaxed files recursively inside this N folder
        # rglob ensures we find files in all subfolders (seeds)
        for x_path in n_dir.rglob("x_relaxed_AR*.txt"):
            # Parse AR
            m = re.search(r"x_relaxed_AR(\d+)\.txt$", x_path.name)
            if not m:
                continue
            ar_val = int(m.group(1))
            
            # Filter Alpha > 100
            if ar_val <= 100:
                continue
                
            jobs.append((n_val, ar_val, x_path))

    # Sort jobs
    jobs.sort(key=lambda t: (t[0], t[1]))
    
    print(f"Found {len(jobs)} jobs matching N > 50 and AR > 100 from all seeds.")
    
    if not jobs:
        print("No jobs found.")
        return

    submitted = 0
    slurm = SlurmCfg()

    for n_val, ar_val, x_path in jobs:
        seed_folder = x_path.parent.name
        
        # Unique run name
        run_name = safe_name(f"{timestamp}_N{n_val}_{seed_folder}_AR{ar_val}_mu{args.friction}")
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy binary
        shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")
        
        # Scene setup
        scene_data = json.loads(json.dumps(base_scene)) # Deep copy
        
        # Update N
        if "scene" in scene_data and "populate" in scene_data["scene"]:
            scene_data["scene"]["populate"]["count"] = n_val
        
        # Update Friction
        if "physics" in scene_data and "soft_contact" in scene_data["physics"]:
             scene_data["physics"]["soft_contact"]["mu"] = args.friction
             scene_data["physics"]["soft_contact"]["mu_static"] = args.friction
            
        # Write scene
        (run_dir / "scene.json").write_text(json.dumps(scene_data, indent=2))
        
        # Link input
        sym_x = run_dir / "x_relaxed.txt"
        if sym_x.exists(): sym_x.unlink()
        sym_x.symlink_to(x_path)
        
        # Command
        # Output stride large to minimize IO
        # No network, No per-rod
        # Entanglement ON
        output_stride = 20000 
        
        sim_parts = [
            "./rigidbody_viewer_3d",
            "--headless",
            "--scene scene.json",
            "--init-csv x_relaxed.txt",
            "--output output.csv",
            f"--steps {args.steps}",
            f"--output-stride {output_stride}",
            "--threads 4",
            # Disable per-rod explicitly? Default is usually off if stride not set, but let's be safe if default changed
            "--perrod-stride 0",
            # Entanglement
            "--entanglement",
            "--entanglement-period 60", # Standard frequency? Or just end? 
                                        # Usually we need it somewhat frequent if we want time evolution, 
                                        # but user asked for "final normalized entanglement".
                                        # If we only want final, we could set period = steps/2?
                                        # But getting time series is cheaply stored in output.csv (just numbers).
                                        # Let's keep 60 or 1000. 60 is fine, it just adds columns to output.csv.
                                        # The CSV size is dominated by rows. With stride 20000, we have 10 rows.
            "--entanglement-threads 0",
        ]
        
        sim_cmd = " ".join(sim_parts)
        
        # Sbatch
        sb = f"""#!/bin/bash
#SBATCH -n {slurm.ntasks}
#SBATCH -c {slurm.cpus}
#SBATCH -N {slurm.nodes}
#SBATCH -t {slurm.time}
#SBATCH -p {slurm.partition}
#SBATCH --mem={slurm.mem_gb}G
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err
#SBATCH --mail-type={slurm.mail_type}
{f"#SBATCH --mail-user={slurm.mail_user}" if slurm.mail_user else ""}
#SBATCH --job-name={run_name}

set -euo pipefail
{slurm.module_line}

echo "Starting {run_name}"
echo "N={n_val}, AR={ar_val}"
echo "CMD: {sim_cmd}"

{sim_cmd}

echo "Done."
"""
        (run_dir / "Sbatch.sh").write_text(sb)
        
        if not args.dry_run:
            subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir, check=True)
            submitted += 1
            print(f"Submitted {run_name}")
        else:
            # print(f"[Dry Run] Prepared {run_dir}")
            pass

    if args.dry_run:
        print(f"[Dry Run] Prepared {len(jobs)} folders in {runs_root}")
    else:
        print(f"Submitted {submitted} jobs to {runs_root}")

if __name__ == "__main__":
    main()
