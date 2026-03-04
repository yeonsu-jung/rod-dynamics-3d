#!/usr/bin/env python3
"""submit_mujoco.py

Submit SLURM jobs to run MuJoCo rod dynamics starting from entangled packings.
Refactored to match the hierarchical structure and self-contained folder strategy
of example_mujoco_run_scripts.

Structure: runs_mujoco/{iteration_title}/{N}/{individual_run_folders}
Where individual_run_folders are named: {timestamp}_RUN_{note}
Preserves:
  - x_relaxed.txt
  - options.txt / options.yml
  - perturb_rod_packings.py (copy of run_sims_with_mujoco.py)
  - util.py, packing_initialization.py, first_analysis.py
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import yaml # pip install PyYAML
from datetime import datetime
from pathlib import Path

# Defaults
DEFAULT_RUNS_ROOT = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs_mujoco")

def find_root_dir(start=None, target_name="rod-dynamics-3d"):
    p = (Path.cwd() if start is None else start).resolve()
    for ancestor in [p, *p.parents]:
        if ancestor.name == target_name:
            return ancestor
    raise SystemExit(f"Could not find repository root named '{target_name}' starting from {p}")

def now_ts():
    return datetime.now().strftime("%Y%m%d-%H%M")

_AR_RE = re.compile(r"x_relaxed_AR(\d+)\.txt$")
def iter_x_relaxed_files(root: Path):
    for p in root.rglob("x_relaxed_AR*.txt"):
        m = _AR_RE.search(p.name)
        if not m: continue
        yield (p, int(m.group(1)))

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", str(s))

def main():
    ap = argparse.ArgumentParser(description="Submit MuJoCo jobs (Hierarchical).")
    ap.add_argument("--n-rods", type=int, required=True, help="Number of rods (N).")
    ap.add_argument("--job-name", type=str, default=None, help="Job name (e.g. mujoco_sweep).")
    ap.add_argument("--iteration-title", type=str, default=None, help="Iteration title for organizing runs.")
    ap.add_argument("--input-root", type=Path, default=None, help="Root for input files.")
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="Output root.")
    
    # Sim Params
    ap.add_argument("--steps", type=int, default=10619, help="Simulation steps.")
    ap.add_argument("--dt", type=float, default=0.01, help="Timestep.") # matched options.yml
    ap.add_argument("--stride", type=int, default=100, help="Output stride.")
    ap.add_argument("--frictions", type=str, default="0.0,0.05,0.1,0.15,0.2,0.4,1.0", help="Comma-separated frictions.")
    ap.add_argument("--init-velocity-sigma", type=float, default=0.1, help="Kick sigma (random_amplitude).")


    # Slurm
    ap.add_argument("--time", type=str, default="0-12:00", help="Time limit.")
    ap.add_argument("--dry-run", action="store_true", help="Dry run.")
    ap.add_argument("--limit", type=int, default=0, help="Limit seeds.")
    
    args = ap.parse_args()
    
    root_dir = find_root_dir()
    
    # Define Source Paths
    src_sim = root_dir / "parametric_study" / "run_sims_with_mujoco.py"
    src_util = root_dir / "study" / "util.py" 
    src_analysis = root_dir / "example_mujoco_run_scripts" / "utils" / "first_analysis.py"
    src_packing = root_dir / "example_mujoco_run_scripts" / "utils" / "packing_initialization.py"
    
    for f in [src_sim, src_util, src_analysis, src_packing]:
        if not f.exists():
            print(f"Warning: Source file {f} not found! Check paths.") # Don't exit hard, maybe user fixed paths
            
    # Input Root
    # /n/home01/yjung/Github/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading
    input_root = args.input_root or (root_dir / "initial-configs" / "relaxation_3rd_multithreading" / f"N{args.n_rods}")
    
    if not input_root.exists():
        print(f"Input root {input_root} does not exist.")
        sys.exit(1)
        
    job_name = args.job_name or f"mujoco_entangled_N{args.n_rods}_{now_ts()}"
    iteration_title = args.iteration_title or job_name
    
    # New structure: runs_mujoco/{iteration_title}/{N}/{individual_run_folders}
    base_runs_dir = args.runs_root / iteration_title / f"N{args.n_rods}"
    
    # Collect jobs
    raw_jobs = sorted(iter_x_relaxed_files(input_root), key=lambda t: (t[0].parent.name, t[1]))
    jobs_by_seed = {}
    for p, ar in raw_jobs:
        seed = p.parent.name
        jobs_by_seed.setdefault(seed, []).append((p, ar))
        
    all_seeds = sorted(jobs_by_seed.keys())
    if args.limit > 0:
        selected_seeds = all_seeds[:args.limit]
    else:
        selected_seeds = all_seeds
        
    frictions = [float(f) for f in args.frictions.split(",") if f.strip()]
    
    print(f"Submitting {len(selected_seeds)} seeds. Output: {base_runs_dir}")
    
    submitted_count = 0
    
    for seed in selected_seeds:
        for x_path, ar in jobs_by_seed[seed]:
            for mu in frictions:
                # Prepare Run Directory
                # New structure: runs_mujoco/{iteration_title}/{N}/{individual_run_folders}
                # individual_run_folder like: {timestamp}_RUN_{note}
                # note like: keys{seed}_N{n}_mu{mu}_AR{ar}_A{kick}_seed{seed}_nonperiodic
                
                timestamp = now_ts()
                note = f"keys{seed}_N{args.n_rods}_mu{mu:.4f}_AR{ar}_A{args.init_velocity_sigma:.3f}"
                run_folder_name = f"{timestamp}_RUN_{note}"
                
                run_dir = base_runs_dir / run_folder_name
                
                if run_dir.exists():
                    print(f"Skipping existing {run_dir}")
                    continue
                    
                if not args.dry_run:
                    run_dir.mkdir(parents=True)
                    
                    # COPY FILES
                    shutil.copy(src_sim, run_dir / "perturb_rod_packings.py")
                    if src_util.exists(): shutil.copy(src_util, run_dir / "util.py")
                    if src_analysis.exists(): shutil.copy(src_analysis, run_dir / "first_analysis.py")
                    if src_packing.exists(): shutil.copy(src_packing, run_dir / "packing_initialization.py")
                    shutil.copy(x_path, run_dir / "x_relaxed.txt")
                    
                    # OPTIONS
                    # Construct options dict compatible with analysis expectation
                    # Analysis expects: file_path, friction, random_amplitude (list), AR, etc?
                    # The legacy analysis parses 'file_path' to get AR. 
                    options = {
                        "file_path": str(x_path), # Original absolute path
                        "random_keys": seed,
                        "n_val": args.n_rods,
                        "friction": mu,
                        "random_amplitude": [args.init_velocity_sigma] * 6,
                        "timestep": args.dt,
                        "max_steps": args.steps,
                        "all_data_interval": args.stride,
                        "save_all_data": False,
                        "add_ground_plane": False,
                        "add_box_boundaries": False,
                        "periodic": False,
                        "job_name": job_name
                    }
                    
                    # Dump YAML
                    with open(run_dir / "options.yml", "w") as f:
                        yaml.dump(options, f)
                        
                    # Dump TXT (string dict) for legacy analysis
                    with open(run_dir / "options.txt", "w") as f:
                        f.write(str(options)) # This creates the python-dict string format
                        
                    # SBATCH
                    # Command must use local 'perturb_rod_packings.py' and 'x_relaxed.txt'
                    # But wait! 'perturb_rod_packings.py' (aka run_sims_with_mujoco) expects CLI args.
                    # Does it read options.yml? No, unless we modify it.
                    # So we pass CLI args matching the options.
                    
                    # Extract numeric seed from seed string (e.g., "199,97,131" -> 199)
                    # Use first number as the random seed
                    seed_num = int(seed.split(',')[0])
                    
                    cmd_sim = (
                        f"python3 perturb_rod_packings.py "
                        f"--input x_relaxed.txt "
                        f"--output . "
                        f"--friction {mu} "
                        f"--dt {args.dt} "
                        f"--steps {args.steps} "
                        f"--stride {args.stride} "
                        f"--kick {args.init_velocity_sigma} "
                        f"--ar {ar} "
                        f"--seed {seed_num} "
                    )
                    
                    cmd_analysis = "python3 first_analysis.py"
                    
                    sb = f"""#!/bin/bash
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t {args.time}
#SBATCH -p seas_compute
#SBATCH --mem=4G
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err
#SBATCH --job-name={note}

module load python
mamba activate mujoco-env

echo "Starting Simulation..."
time {cmd_sim}

echo "Starting Analysis..."
time {cmd_analysis}

echo "Done."
"""
                    
                    with open(run_dir / "Sbatch.sh", "w") as f:
                        f.write(sb)
                        
                    subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir, check=True)
                    submitted_count += 1
                else:
                    print(f"[Dry Run] Would submit to {run_dir}")
                    
    print(f"Submitted {submitted_count} jobs.")

if __name__ == "__main__":
    main()
