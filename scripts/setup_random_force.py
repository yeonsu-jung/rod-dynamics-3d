#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path

def update_scene_file(scene_path):
    with open(scene_path, 'r') as f:
        data = json.load(f)

    # Extract rod properties from bodies[0] if available, else populate
    try:
        if 'bodies' in data['scene'] and len(data['scene']['bodies']) > 0:
            body = data['scene']['bodies'][0]
            length = body.get('length', 1.0)
            diameter = body.get('diameter', 0.01)
            density = body.get('density', 1000.0)
            radius = diameter / 2.0
        else:
            populate = data['scene']['populate']
            length = populate.get('length', 1.0)
            radius = populate.get('radius', 0.005)
            density = populate.get('density', 1000.0)
    except KeyError as e:
        print(f"Skipping {scene_path}: Missing key {e}")
        return

    # Calculate mass
    vol = math.pi * (radius ** 2) * length
    mass = density * vol
    
    # Calculate fSigma
    # Target acceleration ~ 127.32 m/s^2 (Derived from fSigma=10 for AR=100 => mass=0.0785)
    # fSigma = mass * a
    target_acc = 127.324
    f_sigma = mass * target_acc

    # Update randomInit
    if 'randomInit' not in data['scene']:
        data['scene']['randomInit'] = {}
    data['scene']['randomInit']['enabled'] = False

    # Update randomForce
    if 'randomForce' not in data['scene']:
        data['scene']['randomForce'] = {}
    data['scene']['randomForce']['enabled'] = True
    data['scene']['randomForce']['fSigma'] = f_sigma
    
    # Ensure tauMag is present (optional, usually 0)
    if 'tauMag' not in data['scene']['randomForce']:
        data['scene']['randomForce']['tauMag'] = 0.0

    print(f"Updated {scene_path}: R={radius:.5f} M={mass:.5f} fSigma={f_sigma:.5f}")

    with open(scene_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 setup_random_force.py <sweep_folder_path>")
        sys.exit(1)

    sweep_root = Path(sys.argv[1])
    if not sweep_root.exists():
        print(f"Path not found: {sweep_root}")
        sys.exit(1)

    count = 0
    for run_dir in sweep_root.iterdir():
        if not run_dir.is_dir():
            continue
        scene_path = run_dir / "scene.json"
        if scene_path.exists():
            update_scene_file(scene_path)
            count += 1
    
    print(f"Processed {count} run folders.")

if __name__ == "__main__":
    main()
