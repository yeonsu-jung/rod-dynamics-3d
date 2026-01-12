#!/usr/bin/env python3
"""analyze_scaling_mujoco.py

Aggregates summary.csv files from multiple MuJoCo batch directories (results of analyze_mujoco_batch.py)
to plot metrics vs N and AR.

Targets:
- Max Cluster Fraction vs N (grouped by Friction/AR)
- Avg Contacts (Degree) vs N (grouped by Friction/AR)

Usage:
  python3 parametric_study/analyze_scaling_mujoco.py <root_dir_containing_sweeps>
"""

import argparse
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

def plot_metric(
    data_map: Dict[int, Dict[int, Dict[str, List[float]]]], 
    metric_key: str, 
    ylabel: str,
    out_prefix: str,
    root_dir: Path,
    friction: float
):
    # Plot vs N (Lines = AR)
    fig, ax = plt.subplots(figsize=(4, 3))
    has_data = False
    
    sorted_ars = sorted(data_map.keys())
    for ar in sorted_ars:
        n_map = data_map[ar]
        sorted_ns = sorted(n_map.keys())
        
        X, Y, Yerr = [], [], []
        for n in sorted_ns:
            metrics = n_map[n]
            vals = metrics.get(metric_key, [])
            if not vals: continue
            X.append(n)
            Y.append(np.mean(vals))
            Yerr.append(np.std(vals))
            
        if X:
            has_data = True
            ax.errorbar(X, Y, yerr=Yerr, fmt='o-', capsize=3, markersize=4, label=f"AR={ar}")
            
    if has_data:
        ax.set_xlabel("N Rods")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='x-small')
        
        # Log scale check
        all_ns = [n for ar_map in data_map.values() for n in ar_map]
        if all_ns and max(all_ns)/(min(all_ns)+1e-9) > 5:
            ax.set_xscale("log")

        out = root_dir / f"scaling_{out_prefix}_vs_N_mu{friction}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        print(f"Saved {out.name}")
    plt.close(fig)

    # Plot vs AR (Lines = N)
    # Restructure: N -> AR -> vals
    data_by_n = {}
    for ar, n_map in data_map.items():
        for n, vals_dict in n_map.items():
            vals = vals_dict.get(metric_key, [])
            if vals:
                data_by_n.setdefault(n, {})[ar] = vals
                
    fig, ax = plt.subplots(figsize=(4, 3))
    has_data_ar = False
    
    sorted_ns = sorted(data_by_n.keys())
    for n in sorted_ns:
        ar_map = data_by_n[n]
        sorted_ars_sub = sorted(ar_map.keys())
        
        X, Y, Yerr = [], [], []
        for ar in sorted_ars_sub:
            vals = ar_map[ar]
            X.append(ar)
            Y.append(np.mean(vals))
            Yerr.append(np.std(vals))
            
        if X:
            has_data_ar = True
            ax.errorbar(X, Y, yerr=Yerr, fmt='o-', capsize=3, markersize=4, label=f"N={n}")
            
    if has_data_ar:
        ax.set_xlabel("Aspect Ratio (AR)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='x-small')
        
        all_ars = list(data_map.keys())
        if all_ars and max(all_ars)/(min(all_ars)+1e-9) > 5:
            ax.set_xscale("log")

        out = root_dir / f"scaling_{out_prefix}_vs_AR_mu{friction}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        print(f"Saved {out.name}")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir", type=Path, help="Directory containing subfolders with analysis/summary.csv")
    ap.add_argument("--friction", type=float, default=0.4, help="Friction coefficient to filter by (default: 0.5)")
    args = ap.parse_args()

    if not args.root_dir.is_dir():
        print(f"Not a directory: {args.root_dir}")
        return

    # Data structure: AR -> N -> Dict[metric -> values]
    data_map: Dict[int, Dict[int, Dict[str, List[float]]]] = {}
    
    summary_files = list(args.root_dir.rglob("summary.csv"))
    if not summary_files:
        print(f"No summary.csv files found within {args.root_dir}")
        return

    print(f"Found {len(summary_files)} summary files.")
    
    count = 0
    for csv_path in summary_files:
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fric = float(row.get("friction", -1))
                    if abs(fric - args.friction) > 0.05: # loose match
                        continue
                        
                    ar = int(float(row.get("AR", 0)))
                    n_rods = int(float(row.get("N", 0)))
                    
                    # Metrics
                    cluster_frac = float(row.get("max_cluster_frac_end", "nan"))
                    contacts = float(row.get("avg_contacts_end", "nan"))
                    
                    ar_map = data_map.setdefault(ar, {})
                    metrics = ar_map.setdefault(n_rods, {"cluster": [], "top": []})
                    
                    if math.isfinite(cluster_frac):
                        metrics["cluster"].append(cluster_frac)
                        
                    if math.isfinite(contacts):
                        metrics["top"].append(contacts) # topology/neighbor count
                        
                    count += 1
                except ValueError:
                    continue
                    
    print(f"Loaded {count} data points for mu~={args.friction}")
    
    plot_metric(data_map, "cluster", "Max Cluster Fraction", "cluster_frac", args.root_dir, args.friction)
    plot_metric(data_map, "top", "Avg Contacts (Degree)", "avg_contacts", args.root_dir, args.friction)

if __name__ == "__main__":
    main()
