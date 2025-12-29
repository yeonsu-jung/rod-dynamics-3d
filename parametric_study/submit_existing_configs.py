#!/usr/bin/env python3
"""
submit_existing_configs.py

Submit SLURM jobs using existing relaxed configurations from initial-configs/6,7,8.
Filters for N=200 and disabled PBC.
"""

from pathlib import Path
from datetime import datetime
import shutil, subprocess, os, stat, sys, json
import argparse
import math

# Generate target acceleration values (previously fSigma)
# interpreting this list as 'target_acceleration' now
# f_sigmas = [0.01] 
target_accelerations = [0.01]
target_vSigma = [10.0]

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

def main():
    parser = argparse.ArgumentParser(description="Submit jobs from existing relaxed configs.")
    parser.add_argument('--job-name', type=str, default='existing_configs_N200_const_acc', help='Job name prefix.')
    parser.add_argument('--dry-run', action='store_true', help='Generate files but do not submit.')
    args = parser.parse_args()

    root_dir = find_root_dir()
    runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs") / args.job_name
    # runs_root = root_dir / "runs" / args.job_name
    runs_root.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(Path(__file__), runs_root / Path(__file__).name)
    
    timestamp = now_ts()
    print(f"Submitting jobs to: {runs_root}")

    binary_src = root_dir / "build" / "rigidbody_viewer_3d"
    # scene_src = root_dir / "assets/scenes/experiment_bigger_mu_0.2.json"
    scene_src = root_dir / "assets/scenes/default.json"
    configs_dir = root_dir / "initial-configs/6,7,8"

    ensure_executable(binary_src)
    if not scene_src.exists():
        raise SystemExit(f"Scene file not found: {scene_src}")

    with open(scene_src, 'r') as f:
        base_scene = json.load(f)

    # Iterate over subfolders
    if not configs_dir.exists():
        raise SystemExit(f"Configs directory not found: {configs_dir}")

    subfolders = [d for d in configs_dir.iterdir() if d.is_dir()]
    # Filter for N0200
    subfolders = [d for d in subfolders if "N0200" in d.name]
    
    if not subfolders:
        print("No subfolders matching 'N0200' found.")
        sys.exit(0)

    for subdir in subfolders:
        # Expected format: ...-N0200-AR0100-Scale1...
        # Parse AR and Scale
        ar_val = None
        scale_val = None
        
        parts = subdir.name.split('-')
        for p in parts:
            if p.startswith("AR"):
                try:
                    ar_val = float(p[2:])
                except ValueError:
                    pass
            elif p.startswith("Scale"):
                try:
                    scale_val = float(p[5:])
                except ValueError:
                    pass
        
        if ar_val is None or scale_val is None:
            print(f"Skipping {subdir.name}: Could not parse AR or Scale")
            continue

        L = scale_val
        d = L / ar_val
        
        input_file = subdir / "x_relaxed.txt"
        if not input_file.exists():
            print(f"Skipping {subdir.name}: x_relaxed.txt not found")
            continue

        # Calculate Mass to ensure constant acceleration
        density = 2500.0
        radius = d / 2.0
        # Volume approx (cylinder): pi * r^2 * L
        volume = math.pi * (radius**2) * L
        mass = density * volume
        
        print(f"Processing {subdir.name}: L={L}, d={d} (AR={ar_val}) | Mass~={mass:.4e}")

        for target_acc in target_accelerations:
            # fSigma = mass * acceleration
            fSigma = mass * target_acc
            
            # Create run dir
            run_name = f"{timestamp}_{subdir.name}_acc{target_acc:.1e}"
            run_dir = runs_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")
            
            scene = base_scene.copy()
            
            # Update Physics/Scene params
            if 'physics' not in scene: scene['physics'] = {}
            scene['physics']['dt'] = 0.0005
            scene['physics']['lin_damp'] = 0.0
            scene['physics']['ang_damp'] = 0.0
            
            if 'scene' not in scene: scene['scene'] = {}
            
            if 'randomForce' not in scene['scene']: scene['scene']['randomForce'] = {}
            scene['scene']['randomForce']['fSigma'] = float(fSigma)
            # scene['scene']['randomForce']['fSigma'] = 0.001
            scene['scene']['randomForce']['tauMag'] = 0.0
            scene['scene']['randomForce']['enabled'] = True

            # randomInit
            if 'randomInit' not in scene['scene']: scene['scene']['randomInit'] = {}
            scene['scene']['randomInit']['enabled'] = False
            scene['scene']['randomInit']['vSigma'] = 0.1 # should change to 10x the scale of the rod diameter
            scene['scene']['randomInit']['wSpeed'] = 0.0
            scene['scene']['randomInit']['seed'] = 42
            
            if 'populate' not in scene['scene']: scene['scene']['populate'] = {}
            scene['scene']['populate']['count'] = 200
            scene['scene']['populate']['length'] = L
            scene['scene']['populate']['radius'] = d / 2.0
            scene['scene']['populate']['density'] = 2500.0

            # Update bodies template
            if 'bodies' in scene['scene'] and len(scene['scene']['bodies']) > 0:
                scene['scene']['bodies'][0]['length'] = L
                scene['scene']['bodies'][0]['diameter'] = d
                
            # DISABLE PBC
            scene['scene']['periodic'] = {
                "enabled": False,
                # Values arguably dont matter if enabled=False, but keeping structure valid
                "min": [-100.0, -100.0, -100.0],
                "max": [100.0, 100.0, 100.0],
                "cellSize": 2.0
            }
            
            # Set initCsv
            scene['scene']['initCsv'] = str(input_file)
            
            with open(run_dir / "scene.json", 'w') as f:
                json.dump(scene, f, indent=2)
                
            # Sbatch generation
            steps = 2000000
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
                "--entanglement "
                f"--entanglement-every {snap_stride} "  
                "--entanglement-threads 4 "
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
echo "Config: {subdir.name}, fSigma={fSigma}"
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
