#!/usr/bin/env python3
"""export_subset.py

Exports a subset of simulation runs from a batch directory to a new destination,
filtering by specific parameters (e.g., Friction).

Usage:
    python scripts/export_subset.py <source_dir> <dest_dir> [--frictions F1 F2 ...] [--dry-run] [--heavy]

Example:
    python scripts/export_subset.py runs/sweep runs/representative_sweep --frictions 0.0 0.4 1.0
"""

import argparse
import shutil
import re
from pathlib import Path
from typing import List, Set

def parse_friction(dirname: str) -> float:
    """Extract friction from directory name (e.g. ..._Friction0.4_...)."""
    m = re.search(r"_Friction([0-9.]+)", dirname)
    if m:
        return float(m.group(1))
    return -1.0

def main():
    parser = argparse.ArgumentParser(description="Export subset of runs.")
    parser.add_argument("source_dir", type=Path, help="Source batch directory")
    parser.add_argument("dest_dir", type=Path, help="Destination directory")
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.0, 0.4, 1.0], 
                        help="List of friction coefficients to include (default: 0.0 0.4 1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without copying")
    parser.add_argument("--heavy", action="store_true", help="Include heavy files like network.csv")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Tolerance for float handling")

    args = parser.parse_args()

    if not args.source_dir.exists():
        print(f"Error: Source directory {args.source_dir} does not exist.")
        return

    print(f"Scanning {args.source_dir}...")
    print(f"Target Frictions: {args.frictions}")
    
    runs_to_copy = []
    
    for run_dir in args.source_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        fric = parse_friction(run_dir.name)
        if fric < 0:
            continue
            
        # Check if fric is in target list
        match = False
        for target in args.frictions:
            if abs(fric - target) < args.tolerance:
                match = True
                break
        
        if match:
            runs_to_copy.append(run_dir)

    runs_to_copy.sort(key=lambda p: p.name)
    print(f"Found {len(runs_to_copy)} matching runs.")

    if not args.dry_run:
        args.dest_dir.mkdir(parents=True, exist_ok=True)

    # File patterns to copy
    # Always copy these if present
    essential_files = ["output.csv", "scene.json", "network_metrics.json"]
    
    # Heavy files
    heavy_files = ["network.csv"]

    for src in runs_to_copy:
        dest = args.dest_dir / src.name
        
        if args.dry_run:
            print(f"[Dry Run] would copy {src.name} -> {dest}")
            continue

        print(f"Copying {src.name}...")
        dest.mkdir(exist_ok=True)
        
        # Copy files
        for fname in essential_files:
            s_file = src / fname
            if s_file.exists():
                d_file = dest / fname
                if not d_file.exists() or s_file.stat().st_mtime > d_file.stat().st_mtime:
                    shutil.copy2(s_file, d_file)
        
        if args.heavy:
            for fname in heavy_files:
                s_file = src / fname
                if s_file.exists():
                    d_file = dest / fname
                    # Simple check to avoid re-copying huge files if not changed
                    if not d_file.exists() or s_file.stat().st_size != d_file.stat().st_size:
                        print(f"  Copying heavy file {fname}...")
                        shutil.copy2(s_file, d_file)

        # Copy analysis folder if exists
        s_analysis = src / "analysis"
        if s_analysis.exists():
            d_analysis = dest / "analysis"
            if d_analysis.exists():
                 shutil.rmtree(d_analysis) # Simple overwrite for directories
            shutil.copytree(s_analysis, d_analysis)

    print("Done.")

if __name__ == "__main__":
    main()
