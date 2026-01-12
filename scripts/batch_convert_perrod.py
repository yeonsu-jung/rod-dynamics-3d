#!/usr/bin/env python3
"""batch_convert_perrod.py

Recursively scans a directory for `perrod.csv` files and converts them to
`endpoints.csv` using the C++ tool `convert_perrod_endpoints`.

Usage:
    python scripts/batch_convert_perrod.py <search_dir> --binary <path_to_binary> [--jobs 4] [--force]
"""

import argparse
import subprocess
import multiprocessing
from pathlib import Path
from typing import List, Tuple

def convert_task(args: Tuple[Path, Path, bool]):
    run_dir, binary_path, force = args
    input_csv = run_dir / "perrod.csv"
    output_csv = run_dir / "endpoints.csv"
    
    if not input_csv.exists():
        return
        
    if output_csv.exists() and not force:
        # print(f"Skipping {run_dir.name} (endpoints.csv exists)")
        return
        
    try:
        # rod_length is 1.0 based on scene.json inspection
        cmd = [str(binary_path), str(input_csv), str(output_csv), "1.0"]
        # print(f"Converting {run_dir.name}...")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"Converted: {run_dir.name}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {run_dir.name}: {e.stderr.decode().strip()}")
    except Exception as e:
        print(f"Failed {run_dir.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch convert perrod.csv to endpoints.")
    parser.add_argument("search_dir", type=Path, help="Directory to scan")
    parser.add_argument("--binary", type=Path, required=True, help="Path to convert_perrod_endpoints executable")
    parser.add_argument("--jobs", type=int, default=8, help="Number of parallel jobs (default: 8)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing endpoints.csv")
    args = parser.parse_args()
    
    if not args.search_dir.exists():
        print("Search directory not found.")
        return
        
    if not args.binary.exists():
        print("Binary not found.")
        return
        
    tasks = []
    print(f"Scanning {args.search_dir}...")
    
    # Find all perrod.csv
    # Using rglob
    for p in args.search_dir.rglob("perrod.csv"):
        run_dir = p.parent
        tasks.append((run_dir, args.binary, args.force))
        
    print(f"Found {len(tasks)} runs with perrod.csv.")
    
    if not tasks:
        return

    print(f"Starting conversion with {args.jobs} workers...")
    
    with multiprocessing.Pool(processes=args.jobs) as pool:
        pool.map(convert_task, tasks)
        
    print("Done.")

if __name__ == "__main__":
    main()
