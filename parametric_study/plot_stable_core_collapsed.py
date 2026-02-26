#!/usr/bin/env python3
"""plot_stable_core_collapsed.py

Plots Stable Core metrics vs N/Alpha (or Alpha/N).
Tailored for 'stable_core_N*.csv' files.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt

class DataPoint:
    def __init__(self, ar: float, n: int, friction: float, y_val: float):
        self.ar = ar
        self.n = n
        self.friction = friction
        self.y_val = y_val

def load_data(root_dir: Path, y_col: str) -> List[DataPoint]:
    data_points = []
    # Find all stable_core_N*.csv files
    csv_files = list(root_dir.rglob("stable_core_N*.csv"))
    print(f"Found {len(csv_files)} stable core CSV files.")
    
    for csv_path in csv_files:
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # N, AR, mu must exist
                    if "N" not in row or "AR" not in row or "mu" not in row:
                        continue
                        
                    n = int(float(row["N"]))
                    ar = float(row["AR"])
                    mu = float(row["mu"])
                    
                    if y_col == "fraction_unchanged":
                        if "fraction_changed" in row:
                            y_val = 1.0 - float(row["fraction_changed"])
                        else:
                            continue
                    elif y_col not in row:
                        continue
                    else:
                        y_val = float(row[y_col])
                    
                    data_points.append(DataPoint(ar, n, mu, y_val))
                except ValueError:
                    continue
    return data_points

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir", type=Path, help="Root directory containing stable core CSVs")
    ap.add_argument("--output", "-o", type=Path, default=Path("stable_core_collapsed"), help="Output filename")
    ap.add_argument("--y-col", type=str, default="core_fraction", help="Column to plot on Y-axis")
    ap.add_argument("--x-metric", type=str, choices=["N_over_alpha", "alpha_over_N"], default="N_over_alpha")
    ap.add_argument("--friction", type=float, default=1.0, help="Friction to filter (tolerance 0.05)")
    ap.add_argument("--hue", type=str, choices=["alpha", "N", "friction"], default="alpha", help="Group by variable")
    ap.add_argument("--figsize", type=float, nargs=2, default=[3.0, 2.5], help="Figure width height")
    ap.add_argument("--fontsize", type=int, default=8, help="Font size")
    ap.add_argument("--no-legend", action="store_true", help="Remove legend")
    args = ap.parse_args()

    # Style
    plt.rcParams.update({'font.size': args.fontsize})

    points = load_data(args.root_dir, args.y_col)
    print(f"Loaded {len(points)} total points.")
    
    # Filter Friction
    if args.friction is not None:
        orig = len(points)
        points = [p for p in points if abs(p.friction - args.friction) < 0.05]
        print(f"Filtered {orig - len(points)} points not matching friction {args.friction}. Remaining: {len(points)}")
        
    if not points:
        print("No data remaining.")
        return

    # Grouping
    groups: Dict[float, List[DataPoint]] = {}
    for p in points:
        if args.hue == "alpha":
            key = p.ar
        elif args.hue == "N":
            key = float(p.n)
        else:
            key = p.friction
            
        # Tolerance matching for keys
        found_key = None
        for k in groups:
            if abs(k - key) < 1e-4:
                found_key = k
                break
        if found_key is not None:
            groups[found_key].append(p)
        else:
            groups[key] = [p]
            
    sorted_keys = sorted(groups.keys())
    
    fig, ax = plt.subplots(figsize=(args.figsize[0], args.figsize[1]))
    
    for key in sorted_keys:
        group = groups[key]
        
        # Prepare XY
        X_vals = []
        Y_vals = []
        for p in group:
            if args.x_metric == "alpha_over_N":
                val = p.ar / p.n if p.n != 0 else 0
            else:
                val = p.n / p.ar if p.ar != 0 else 0
            
            X_vals.append(val)
            Y_vals.append(p.y_val)
            
        # Average duplicate X
        by_x = {}
        for x, y in zip(X_vals, Y_vals):
            by_x.setdefault(x, []).append(y)
            
        XYE = []
        for x, ys in by_x.items():
            XYE.append((x, np.mean(ys), np.std(ys)))
        XYE.sort(key=lambda t: t[0])
        
        X = [t[0] for t in XYE]
        Y = [t[1] for t in XYE]
        Yerr = [t[2] for t in XYE]
        
        is_int = args.hue in ["alpha", "N"]
        lbl = f"{args.hue}={int(key) if is_int else key}"
        
        if X:
            line, = ax.plot(X, Y, 'o-', markersize=3, alpha=0.8, label=lbl, markeredgewidth=0.0)
            # Fill between
            X_np = np.array(X)
            Y_np = np.array(Y)
            Yerr_np = np.array(Yerr)
            ax.fill_between(X_np, Y_np - Yerr_np, Y_np + Yerr_np, color=line.get_color(), alpha=0.2, edgecolor="none")
            
    # Labels
    if args.x_metric == "alpha_over_N":
        ax.set_xlabel(r"$\alpha / N$")
    else:
        ax.set_xlabel(r"$N / \alpha$")
        
    y_name = "Stable Core Fraction" if "fraction" in args.y_col else "Stable Core Size"
    ax.set_ylabel(y_name)
    ax.set_title(f"Stable Core: vs {args.x_metric.replace('_','/')}", fontsize=args.fontsize)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    
    if not args.no_legend:
        if len(sorted_keys) > 8:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=args.fontsize-2)
        else:
            ax.legend(fontsize=args.fontsize-1)
            
    out_base = str(args.output)
    if out_base.endswith(".png"):
        out_base = out_base[:-4]
    
    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=300)
    # fig.savefig(f"{out_base}.svg")
    print(f"Saved {out_base}.png")
    
if __name__ == "__main__":
    main()
