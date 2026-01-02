#!/usr/bin/env python3
"""analyze_scaling_n.py

Aggregates summary.csv files from multiple batch directories to plot metrics vs N.
Specifically targets:
- Final Normalized Entanglement vs N
- Filtered by Friction (default 0.4)
- Grouped by Aspect Ratio (AR)

Usage:
  python3 parametric_study/analyze_scaling_n.py <root_dir_containing_sweeps>
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
    ap.add_argument("--friction", type=float, default=0.4, help="Friction coefficient to filter by (default: 0.4)")
    args = ap.parse_args()

    if not args.root_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.root_dir}")

    # Data structure: AR -> N -> List[values]
    data_map: Dict[int, Dict[int, List[float]]] = {}
    
    # 1. Find all summary.csv files
    # Typically in <root>/<batch>/analysis/summary.csv
    summary_files = list(args.root_dir.rglob("summary.csv"))
    if not summary_files:
        raise SystemExit(f"No summary.csv files found within {args.root_dir}")

    print(f"Found {len(summary_files)} summary files. parsing...")

    count = 0
    for csv_path in summary_files:
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fric = float(row.get("friction", -1))
                    # Filter by friction (approximate float match)
                    if abs(fric - args.friction) > 1e-6:
                        continue
                    
                    ar = int(float(row.get("AR", 0)))
                    n_rods = int(float(row.get("N", 0)))
                    
                    val_str = row.get("ent_norm_end", "nan")
                    val = float(val_str)
                    
                    if math.isfinite(val):
                        ar_map = data_map.setdefault(ar, {})
                        ar_map.setdefault(n_rods, []).append(val)
                        count += 1
                except ValueError:
                    continue

    print(f"Loaded {count} valid data points for mu={args.friction}.")
    
    if not data_map:
        print("No matching data found.")
        return

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(3, 2.5))
    
    sorted_ars = sorted(data_map.keys())
    
    has_data = False
    for ar in sorted_ars:
        n_map = data_map[ar]
        sorted_ns = sorted(n_map.keys())
        
        X, Y, Yerr = [], [], []
        for n in sorted_ns:
            vals = n_map[n]
            if not vals:
                continue
            X.append(n)
            Y.append(np.mean(vals))
            Yerr.append(np.std(vals))
            
        if X:
            has_data = True
            ax.errorbar(X, Y, yerr=Yerr, fmt='o-', capsize=3, markersize=4, label=f"AR={ar}")

    if not has_data:
        print("No valid series to plot.")
        return
        
    ax.set_xlabel("Number of Rods (N)")
    ax.set_ylabel("Final Norm. Entanglement")
    # Title might be too clunky for small plot
    # ax.set_title(f"Normalized Entanglement vs N (mu={args.friction})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='x-small')
    
    # Maybe log scale if N spans orders of magnitude
    all_x = [n for ar_map in data_map.values() for n in ar_map]
    if all_x and max(all_x)/(min(all_x)+1e-9) > 5:
        ax.set_xscale("log")
    
    out_path_png = args.root_dir / f"scaling_ent_norm_vs_N_mu{args.friction}.png"
    out_path_svg = args.root_dir / f"scaling_ent_norm_vs_N_mu{args.friction}.svg"
    
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=300)
    fig.savefig(out_path_svg)
    print(f"Saved plots to:\n  {out_path_png}\n  {out_path_svg}")
    plt.close(fig)

    # 3. Plotting (Metric vs AR, overlaid by N)
    print("Generating vs-AR plot...")
    
    # Restructure data: N -> AR -> List[values]
    data_by_n: Dict[int, Dict[int, List[float]]] = {}
    all_ns = set()
    for ar, n_map in data_map.items():
        for n, vals in n_map.items():
            data_by_n.setdefault(n, {})[ar] = vals
            all_ns.add(n)
            
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sorted_ns = sorted(list(all_ns))
    
    has_data_n = False
    for n in sorted_ns:
        ar_map = data_by_n[n]
        sorted_ars_sub = sorted(ar_map.keys())
        
        X, Y, Yerr = [], [], []
        for ar in sorted_ars_sub:
            vals = ar_map[ar]
            if not vals:
                continue
            X.append(ar)
            Y.append(np.mean(vals))
            Yerr.append(np.std(vals))
            
        if X:
            has_data_n = True
            ax.errorbar(X, Y, yerr=Yerr, fmt='o-', capsize=3, markersize=4, label=f"N={n}")

    if has_data_n:
        ax.set_xlabel("Aspect Ratio (AR)")
        ax.set_ylabel("Final Norm. Entanglement")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='x-small')
        
        # Log scale for AR often makes sense
        all_ar_vals = [ar for n_map in data_map.values() for ar in n_map] # wait, this logic is slightly wrong, gathering keys is safer
        all_ar_keys = list(data_map.keys())
        if all_ar_keys and max(all_ar_keys)/(min(all_ar_keys)+1e-9) > 5:
            ax.set_xscale("log")

        out_path_png_ar = args.root_dir / f"scaling_ent_norm_vs_AR_mu{args.friction}.png"
        out_path_svg_ar = args.root_dir / f"scaling_ent_norm_vs_AR_mu{args.friction}.svg"
        
        fig.tight_layout()
        fig.savefig(out_path_png_ar, dpi=300)
        fig.savefig(out_path_svg_ar, dpi=300) # svg doesn't use dpi but good to keep consistency
        print(f"Saved vs-AR plots to:\n  {out_path_png_ar}\n  {out_path_svg_ar}")
    
    plt.close(fig)

if __name__ == "__main__":
    main()
