#!/usr/bin/env python3
import os
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run rod-dynamics-3d sweep for N=200")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    # Configuration
    project_root = Path(__file__).parent.parent
    build_dir = project_root / "build"
    executable = build_dir / "rigidbody_viewer_3d"
    base_config_dir = project_root / "initial-configs/6,7,8"
    output_dir = build_dir / "sweep_output"

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find directories
    print(f"Searching in {base_config_dir}...")
    found_count = 0
    
    for root, dirs, files in os.walk(base_config_dir):
        # We look for directories containing "N0200"
        # The user's example directories had N0200 in their name
        # We also need x_relaxed.txt inside to be a valid target
        
        # Check current root folder name for N0200 (or check if we are inside one)
        # Actually os.walk yields root.
        
        dir_name = os.path.basename(root)
        if "N0200" in dir_name and "x_relaxed.txt" in files:
            found_count += 1
            x_relaxed_path = Path(root) / "x_relaxed.txt"
            
            # Construct output prefix
            # e.g., sweep_2025-02-16_17_EntangledRelaxedPacking-N0200-AR0050-Scale1
            prefix = f"sweep_{dir_name}"
            
            perrod_out = output_dir / f"{prefix}_perrod.csv"
            stats_out = output_dir / f"{prefix}_output.csv"
            network_out = output_dir / f"{prefix}_network.csv"
            
            # Construct command
            cmd = [
                str(executable),
                "--profile",
                "--init-csv", str(x_relaxed_path),
                "--steps", "100000",
                "--csv-stride", "1000",
                "--scene", "../assets/scenes/default_entangled.json",
                "--perrod", str(perrod_out),
                "--output", str(stats_out),
                "--network", str(network_out),
                "--headless"
            ]
            
            print(f"[{found_count}] Found target: {dir_name}")
            print(f"  Command: {' '.join(cmd)}")
            
            if not args.dry_run:
                try:
                    subprocess.run(cmd, check=True, cwd=build_dir)
                    print("  -> Completed.")
                except subprocess.CalledProcessError as e:
                    print(f"  -> Failed with error: {e}")
            print("-" * 40)

    if found_count == 0:
        print("No matching directories found (looking for *N0200* with x_relaxed.txt).")
        # Fallback check: look for subdirectories and recurse? os.walk already recurses.

if __name__ == "__main__":
    main()
