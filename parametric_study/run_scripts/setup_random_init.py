#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def update_scene_file(scene_path):
    with open(scene_path, 'r') as f:
        data = json.load(f)

    # Disable randomForce
    if 'randomForce' in data['scene']:
        data['scene']['randomForce']['enabled'] = False

    # Enable randomInit
    if 'randomInit' not in data['scene']:
        data['scene']['randomInit'] = {}
    
    data['scene']['randomInit']['enabled'] = True
    
    # Ensure vSigma is present (default often 0.1)
    if 'vSigma' not in data['scene']['randomInit']:
        data['scene']['randomInit']['vSigma'] = 0.1

    print(f"Updated {scene_path}: randomInit=True, randomForce=False")

    with open(scene_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 setup_random_init.py <sweep_folder_path>")
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
