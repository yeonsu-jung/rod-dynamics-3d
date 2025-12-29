#!/usr/bin/env python3
import shutil, subprocess, os, stat, sys, json
from pathlib import Path
from datetime import datetime
import numpy as np

def find_root_dir(start=None, target_name="rod-dynamics-3d"):
    p = Path.cwd() if start is None else Path(start).resolve()
    for ancestor in [p, *p.parents]:
        if ancestor.name == target_name:
            return ancestor
    raise SystemExit(f"Could not find repository root named '{target_name}' starting from {p}")

def parse_csv_metadata(csv_path):
    meta = {}
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('#'):
                parts = line[1:].split('=')
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip()
                    try:
                        if val.lower() == 'true': val = True
                        elif val.lower() == 'false': val = False
                        elif '.' in val or 'e' in val: val = float(val)
                        else: val = int(val)
                    except:
                        pass
                    meta[key] = val
            else:
                break # Stop at first non-comment line
    return meta

def main():
    root_dir = find_root_dir()
    sweep_dir = root_dir / "initial-configs/sweep_box3.0"
    
    if not sweep_dir.exists():
        print(f"Error: {sweep_dir} does not exist.")
        return

    # Base scene
    scene_src = root_dir / "assets/scenes/experiment_bigger_mu_0.2.json"
    if not scene_src.exists():
        print(f"Error: {scene_src} does not exist.")
        return

    with open(scene_src, 'r') as f:
        base_scene = json.load(f)

    # Output directory
    runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/subset_sweep_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    runs_root.mkdir(parents=True, exist_ok=True)
    print(f"Submitting jobs to: {runs_root}")

    binary_src = root_dir / "build" / "rigidbody_viewer_3d"
    if not binary_src.exists():
        print(f"Error: {binary_src} does not exist. Please build the project.")
        return

    for subdir in sweep_dir.iterdir():
        if not subdir.is_dir(): continue
        
        csv_path = subdir / "attempts.csv"
        if not csv_path.exists():
            print(f"Skipping {subdir.name}: attempts.csv not found")
            continue

        meta = parse_csv_metadata(csv_path)
        
        # Required keys
        if 'box_size' not in meta or 'rod_length' not in meta or 'rod_diameter' not in meta:
            print(f"Skipping {subdir.name}: Missing metadata in CSV")
            continue

        L_box = meta['box_size']
        L_rod = meta['rod_length']
        D_rod = meta['rod_diameter']
        
        # Calculate N
        # N = L_box^3 / (L_rod^2 * D_rod)
        N = int(round((L_box**3) / (L_rod**2 * D_rod)))
        
        print(f"Processing {subdir.name}: L_box={L_box}, L_rod={L_rod}, D_rod={D_rod} -> N={N}")

        # Create run directory
        run_dir = runs_root / subdir.name
        run_dir.mkdir(exist_ok=True)

        # Prepare JSON
        scene = base_scene.copy()
        
        # Update scene
        scene['scene']['populate']['count'] = N
        scene['scene']['initCsv'] = str(csv_path)
        
        # Update box size (assuming centered at 0)
        half_box = L_box / 2.0
        scene['scene']['periodic']['min'] = [-half_box, -half_box, -half_box]
        scene['scene']['periodic']['max'] = [half_box, half_box, half_box]
        
        # Write JSON
        json_path = run_dir / "scene.json"
        with open(json_path, 'w') as f:
            json.dump(scene, f, indent=4)

        # Create SLURM script
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name={subdir.name}
#SBATCH --output={run_dir}/output.log
#SBATCH --error={run_dir}/error.log
#SBATCH --partition=seas_compute
#SBATCH --time=7-00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

module load python

{binary_src} {json_path} --headless
"""
        sbatch_path = run_dir / "submit.sh"
        with open(sbatch_path, 'w') as f:
            f.write(slurm_script)
        
        # Submit
        print(f"Submitting {sbatch_path}")
        subprocess.run(["sbatch", str(sbatch_path)])

if __name__ == "__main__":
    main()
