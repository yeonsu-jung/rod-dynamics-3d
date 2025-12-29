#!/usr/bin/env python3
"""analyze_scaling_friction.py

Aggregates summary.csv files to plot Normalized Entanglement vs Friction.
Specifically targets:
- Filtered by N (default 200)
- Grouped by Aspect Ratio (AR)

Usage:
  python3 parametric_study/analyze_scaling_friction.py <root_dir> --n-rods 200
"""

import argparse
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir", type=Path, help="Directory containing subfolders with analysis/summary.csv")
    ap.add_argument("--n-rods", type=int, default=200, help="Rod count (N) to filter by (default: 200)")
    args = ap.parse_args()

    if not args.root_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.root_dir}")

    # Data structure: AR -> Friction -> List[values]
    data_map: Dict[int, Dict[float, List[float]]] = {}
    
    # 1. Find all summary.csv files
    summary_files = list(args.root_dir.rglob("summary.csv"))
    if not summary_files:
        raise SystemExit(f"No summary.csv files found within {args.root_dir}")

    print(f"Found {len(summary_files)} summary files. parsing...")

    count = 0
    for csv_path in summary_files:
        try:
            with csv_path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        fric = float(row.get("friction", -1))
                        ar = int(float(row.get("AR", 0)))
                        n_rods = int(float(row.get("N", 0)))
                        
                        # Filter by N
                        if n_rods != args.n_rods:
                            continue
                        
                        val_str = row.get("ent_norm_end", "nan")
                        val = float(val_str)
                        
                        if math.isfinite(val):
                            ar_map = data_map.setdefault(ar, {})
                            ar_map.setdefault(fric, []).append(val)
                            count += 1
                    except ValueError:
                        continue
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")

    print(f"Loaded {count} valid data points for N={args.n_rods}.")
    
    if not data_map:
        print("No matching data found.")
        return

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(3, 2.5))
    
    sorted_ars = sorted(data_map.keys())
    
    has_data = False
    for ar in sorted_ars:
        f_map = data_map[ar]
        sorted_frics = sorted(f_map.keys())
        
        X, Y, Yerr = [], [], []
        for f in sorted_frics:
            vals = f_map[f]
            if not vals:
                continue
            X.append(f)
            Y.append(np.mean(vals))
            Yerr.append(np.std(vals))
            
        if X:
            has_data = True
            ax.errorbar(X, Y, yerr=Yerr, fmt='o-', capsize=3, markersize=4, label=f"AR={ar}")

    if not has_data:
        print("No valid series to plot.")
        return
        
    ax.set_xlabel(r"Friction Coefficient ($\mu$)")
    ax.set_ylabel("Final Norm. Entanglement")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='x-small')
    
    # Log scale if friction spans orders (handle 0 carefully)
    # Usually friction is linear 0..1, so maybe linear is fine.
    # If using logarithmic frictions (0.01, 0.1, 1), log scale helps.
    all_x = [f for ar_map in data_map.values() for f in ar_map]
    if all_x and max(all_x) > 0 and min(x for x in all_x if x>0) < 0.1:
        # Check if 0 is present. If so, symlog or just linear might be safer unless we mask 0
        pass 
    
    out_path_png = args.root_dir / f"scaling_ent_norm_vs_Friction_N{args.n_rods}.png"
    out_path_svg = args.root_dir / f"scaling_ent_norm_vs_Friction_N{args.n_rods}.svg"
    
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=300)
    fig.savefig(out_path_svg)
    print(f"Saved plots to:\n  {out_path_png}\n  {out_path_svg}")
    plt.close(fig)

if __name__ == "__main__":
    main()
