#!/usr/bin/env python3
"""aggregate_output.py

Aggregates output.csv files into a single summary.csv without using pandas.
Minimal dependency version.
"""

import csv
import json
import argparse
from pathlib import Path
import math

def get_metadata_from_scene(run_dir):
    scene_path = run_dir / "scene.json"
    meta = {"N": 0, "friction": 0.4} # Defaults
    if scene_path.exists():
        try:
            data = json.loads(scene_path.read_text())
            # Friction
            phy = data.get("physics", {})
            if "soft_contact" in phy:
                meta["friction"] = phy["soft_contact"].get("mu", 0.4)
            
            # N
            pop = data.get("scene", {}).get("populate", {})
            meta["N"] = pop.get("count", 0)
        except:
            pass
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        print(f"Error: {args.run_dir} is not a directory")
        return

    summary_rows = []
    
    # Header for summary.csv
    # Matching plot_collapsed expectation: AR (or alpha), N, friction, ent_norm_end
    
    print(f"Scanning {args.run_dir}...")
    
    count = 0
    for subdir in args.run_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        out_csv = subdir / "output.csv"
        if not out_csv.exists():
            continue
            
        # Parse AR from folder name (reliable source for this batch)
        # Name format: ..._AR1000...
        import re
        m = re.search(r"_AR(\d+)", subdir.name)
        if m:
            ar = int(m.group(1))
        else:
            # Try x_relaxed name inside?
            # Or pass skip
            continue
            
        # Get metadata (N, Friction)
        meta = get_metadata_from_scene(subdir)
        n_rods = meta["N"]
        friction = meta["friction"]
        
        # Read last line of output.csv
        try:
            last_row = None
            with out_csv.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    last_row = row
            
            if last_row:
                ent_sum = float(last_row.get("ent_sum", 0))
                ent_pairs = float(last_row.get("ent_pairs", 0))
                ent_norm = ent_sum / ent_pairs if ent_pairs > 0 else float("nan")
                
                # We need to construct a row compatible with plot_collapsed.py
                # It looks for: AR, N, friction, ent_norm_end
                summary_rows.append({
                    "run": subdir.name,
                    "AR": ar,
                    "N": n_rods,
                    "friction": friction,
                    "ent_norm_end": ent_norm
                })
                count += 1
                
        except Exception as e:
            print(f"Error processing {subdir.name}: {e}")
            
    print(f"Aggregated {count} runs.")
    
    if summary_rows:
        out_path = args.run_dir / "summary.csv" # Plot script looks for summary.csv inside subfolders mostly, or one big one?
        # analyze_scaling_n.py looks for `summary.csv` recursively.
        # plot_collapsed.py does `list(root_dir.rglob("summary.csv"))`.
        # So one summary.csv at the top level is fine IF `plot_collapsed.py` handles parsing it.
        # Wait, `load_data` iterates rows. So one file with multiple rows is perfect.
        
        with out_path.open("w", newline="") as f:
            fieldnames = ["run", "AR", "N", "friction", "ent_norm_end"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
            
        print(f"Saved summary to {out_path}")

if __name__ == "__main__":
    main()
