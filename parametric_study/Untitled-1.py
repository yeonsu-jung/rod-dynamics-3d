#!/usr/bin/env python3
"""
submit_box_sweep.py

Submit SLURM jobs using initial configurations from initial-configs/sweep_box3.0.
Selects a subset of rods based on N = L^3 / (l^2 * d).
"""

from pathlib import Path
from datetime import datetime
import shutil, subprocess, os, stat, sys, json
import argparse
# import numpy as np

# Generate fSigma values from 1e-4 to 1e0
# f_sigmas = np.geomspace(1e-4, 1e0, num=5)
f_sigmas = [0.0001, 0.001, 0.01, 0.1, 1.0]

# SLURM defaults
SLURM = {
    "partition":  "seas_compute",
    "time":       "7-00:00",
    "mem":        "100",
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

def parse_csv_metadata(csv_path):
    metadata = {}
    with open(csv_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                parts = line.strip('# \n').split('=')
                if len(parts) == 2:
                    key, val = parts
                    try:
                        metadata[key.strip()] = float(val.strip())
                    except ValueError:
                        metadata[key.strip()] = val.strip()
            else:
                break # Stop at first non-comment line
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Submit box sweep jobs.")
    parser.add_argument('--job-name', type=str, default='box_sweep', help='Job name prefix.')
    parser.add_argument('--seed-offset', type=int, default=1000, help='Offset to add to random seed.')
    parser.add_argument('--dry-run', action='store_true', help='Generate files but do not submit.')
    args = parser.parse_args()

    root_dir = find_root_dir()
    runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs") / args.job_name
    runs_root.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(Path(__file__), runs_root / Path(__file__).name)
    
    timestamp = now_ts()
    print(f"Submitting box sweep jobs to: {runs_root}")

    binary_src = root_dir / "build" / "rigidbody_viewer_3d"
    scene_src = root_dir / "assets/scenes/experiment_bigger_mu_0.2.json"
    configs_dir = root_dir / "initial-configs/sweep_box1.1"

    ensure_executable(binary_src)
    if not scene_src.exists():
        raise SystemExit(f"Scene file not found: {scene_src}")

    with open(scene_src, 'r') as f:
        base_scene = json.load(f)

    # Iterate over subfolders
    subfolders = [d for d in configs_dir.iterdir() if d.is_dir()]
    # run only for aspect ratio 30, 100 and 300
    subfolders = [d for d in subfolders if any(d.name.startswith(f"ar{r}_") for r in ["30", "100", "200"])]
    
    
    normalized_density = 3.0
    
    for subdir in subfolders:
        csv_path = subdir / "attempts.csv"
        if not csv_path.exists():
            # print(f"Skipping {subdir.name}: attempts.csv not found")
            continue
            
        meta = parse_csv_metadata(csv_path)
        required_keys = ['rod_length', 'rod_diameter', 'box_size']
        if not all(k in meta for k in required_keys):
            print(f"Skipping {subdir.name}: Missing metadata. Found: {meta.keys()}")
            continue
            
        L = meta['rod_length']
        d = meta['rod_diameter']
        box_size = meta['box_size']
        
        # Calculate N = box_size^3 / (L^2 * d)
        num_rods = int(normalized_density * box_size**3 / (L**2 * d))
        
        print(f"Processing {subdir.name}: L={L}, d={d}, box={box_size} -> N={num_rods}")

        for fSigma in f_sigmas:
            # Create run dir
            run_name = f"{timestamp}_{subdir.name}_fSig{fSigma:.1e}"
            run_dir = runs_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")
            
            scene = base_scene.copy()
            
            # soft_physics_options = scene['physics']['soft_contact'].copy()
            
            # Update Physics/Scene params
            if 'physics' not in scene: scene['physics'] = {}
            scene['physics']['dt'] = 0.0005 # From reference script
            scene['physics']['lin_damp'] = 0.0
            scene['physics']['ang_damp'] = 0.0
            
            if 'scene' not in scene: scene['scene'] = {}
            
            if 'randomForce' not in scene['scene']: scene['scene']['randomForce'] = {}
            scene['scene']['randomForce']['fSigma'] = float(fSigma)
            scene['scene']['randomForce']['tauMag'] = 0.0
            # Base seed from scene file + offset + some variation based on params if desired, 
            # but user asked for "different random forcing seed", so we shift it globally.
            base_seed = 123
            if 'seed' in scene['scene']['randomForce']:
                 base_seed = scene['scene']['randomForce']['seed']
            scene['scene']['randomForce']['seed'] = base_seed + args.seed_offset
            
            if 'populate' not in scene['scene']: scene['scene']['populate'] = {}
            scene['scene']['populate']['count'] = num_rods
            scene['scene']['populate']['length'] = L
            scene['scene']['populate']['radius'] = d / 2.0
            scene['scene']['populate']['density'] = 2500.0 # Default
            
            # Update bodies template
            if 'bodies' in scene['scene'] and len(scene['scene']['bodies']) > 0:
                scene['scene']['bodies'][0]['length'] = L
                scene['scene']['bodies'][0]['diameter'] = d
                
            # Update Periodic Box
            half_box = box_size / 2.0
            scene['scene']['periodic'] = {
                "enabled": True,
                "min": [-half_box, -half_box, -half_box],
                "max": [half_box, half_box, half_box],
                "cellSize": 2.0
            }
            
                       
            # Set initCsv
            scene['scene']['initCsv'] = str(csv_path)
            
            with open(run_dir / "scene.json", 'w') as f:
                json.dump(scene, f, indent=2)
                
            # Sbatch generation (same as reference)
            steps = 200000 # From reference
            target_frames = 500
            snap_stride = max(1, steps // target_frames)
            
            sim_cmd = (
                "time "
                "./rigidbody_viewer_3d "
                "--headless "
                "--scene scene.json "
                "--output output.csv "
                f"--snap-stride {snap_stride} "
                f"--snap-frames {target_frames} "                
                f"--perrod perrod.csv "
                f"--perrod-max {target_frames} "
                f"--perrod-stride {snap_stride} "
                f"--steps {steps} "
                f"--threads {SLURM['cpus']}"
            )
            
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
#SBATCH --job-name={args.job_name}_{subdir.name}

set -euo pipefail
{SLURM['module_line']}

echo "======================================"
echo "Box Sweep: {subdir.name}, fSigma={fSigma}"
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
                print(f"    Submitted job for {subdir.name} f={fSigma:.1e}")
            else:
                print(f"    [Dry Run] Created Sbatch.sh for {subdir.name} f={fSigma:.1e}")

if __name__ == "__main__":
    main()
