#!/usr/bin/env python3
"""plot_collapsed.py

Plots Final Normalized Entanglement vs Alpha/N, overlaid by Friction Coefficient.
Supports both "Modern" summary.csv (with 'friction' col) and "Legacy" summary.csv (inferring from path).
"""

import argparse
import csv
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Regex to infer friction from directory path if missing in CSV
# Matches "mu0.4", "fric0.2", "friction_0.5", etc.
FRICTION_RE = re.compile(r"(?:mu|fric(?:tion)?)_?([0-9]*\.?[0-9]+)", re.IGNORECASE)

class DataPoint:
    def __init__(self, ar: float, n: int, friction: float, ent_norm: float):
        self.ar = ar
        self.n = n
        self.friction = friction
        self.ent_norm = ent_norm

def parse_friction_from_path(path: Path) -> Optional[float]:
    """Walk up the path to find a friction signature."""
    # Check parent directory names
    for part in path.parts:
        m = FRICTION_RE.search(part)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None

def load_data(root_dir: Path) -> List[DataPoint]:
    data_points = []
    summary_files = list(root_dir.rglob("summary.csv"))
    
    print(f"Found {len(summary_files)} summary files.")
    
    for csv_path in summary_files:
        # Try to infer friction from path as a fallback or default
        inferred_friction = parse_friction_from_path(csv_path)
        
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # 1. Parse AR (Alpha)
                    # Legacy: 'alpha', Modern: 'AR'
                    ar_str = row.get("AR") or row.get("alpha")
                    if not ar_str:
                        continue
                    ar = float(ar_str)
                    
                    # 2. Parse N
                    n_str = row.get("N")
                    if not n_str:
                        continue
                    n = int(float(n_str))
                    
                    # 3. Parse Friction
                    # Modern: 'friction', Legacy: Missing
                    fric_str = row.get("friction")
                    if fric_str:
                        friction = float(fric_str)
                    elif inferred_friction is not None:
                        friction = inferred_friction
                    else:
                        # Fallback default if absolutely no info found
                        friction = 0.4 
                        
                    # 4. Parse Entanglement
                    # Modern: 'ent_norm_end'
                    # Legacy: 'ent_sum_last' / 'ent_pairs_last' (maybe?) or just missing
                    ent_val = float("nan")
                    
                    if "ent_norm_end" in row:
                        ent_val = float(row["ent_norm_end"])
                    elif "ent_sum_last" in row and "ent_pairs_last" in row:
                        s = float(row["ent_sum_last"])
                        p = float(row["ent_pairs_last"])
                        if p > 0:
                            ent_val = s / p
                            
                    if math.isfinite(ent_val):
                        data_points.append(DataPoint(ar, n, friction, ent_val))
                        
                except ValueError:
                    continue
                    
    return data_points

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dirs", type=Path, nargs="+", help="Directories containing subfolders with analysis/summary.csv")
    ap.add_argument("--output", "-o", type=Path, default=None, help="Output file path (without extension). Defaults to scaling_collapsed_mu_overlay")
    ap.add_argument("--min-ar", type=float, default=None, help="Exclude data with Aspect Ratio < min-ar")
    ap.add_argument("--min-n", type=int, default=None, help="Exclude data with N < min-n")
    ap.add_argument("--export-csv", type=Path, default=None, help="Path to export data as CSV")
    ap.add_argument("--x-metric", type=str, choices=["alpha_over_N", "N_over_alpha"], default="alpha_over_N", help="Metric for X-axis")
    ap.add_argument("--bins", type=int, default=0, help="Number of bins for X-axis (0 to disable)")
    ap.add_argument("--bin-type", type=str, choices=["log", "linear"], default="log", help="Binning type")
    ap.add_argument("--friction", type=float, nargs="*", default=None, help="Filter for specific friction coefficients (space separated)")
    ap.add_argument("--hue", type=str, choices=["friction", "alpha", "N"], default="friction", help="Variable to group/color by")
    ap.add_argument("--z-factor", type=float, default=1.0, help="Scaling factor for X-axis (e.g., Z in N/(Z*alpha))")
    ap.add_argument("--figsize", type=float, nargs=2, default=[6, 4.5], help="Figure size (width height)")
    ap.add_argument("--fontsize", type=int, default=10, help="Base font size")
    ap.add_argument("--no-legend", action="store_true", help="Do not show legend")
    args = ap.parse_args()

    # Set global font size
    plt.rcParams.update({'font.size': args.fontsize})

    all_points = []
    print("Loading data...")
    for root in args.root_dirs:
        if not root.is_dir():
            print(f"Warning: {root} is not a directory. Skipping.")
            continue
        all_points.extend(load_data(root))
        
    points = all_points
    print(f"Loaded {len(points)} data points total.")

    if args.min_ar is not None:
        original_count = len(points)
        points = [p for p in points if p.ar >= args.min_ar]
        print(f"Filtered {original_count - len(points)} points with AR < {args.min_ar}. Remaining: {len(points)}")

    if args.min_n is not None:
        original_count = len(points)
        points = [p for p in points if p.n >= args.min_n]
        print(f"Filtered {original_count - len(points)} points with N < {args.min_n}. Remaining: {len(points)}")
        
    if args.friction is not None and len(args.friction) > 0:
        original_count = len(points)
        # Match any in the list
        keep = []
        for p in points:
            # Check if p.friction matches ANY in args.friction
            match = False
            for f_target in args.friction:
                if abs(p.friction - f_target) < 1e-4:
                    match = True
                    break
            if match:
                keep.append(p)
        points = keep
        print(f"Filtered {original_count - len(points)} points not in requested frictions. Remaining: {len(points)}")
    
    if not points:
        print("No valid data found.")
        return

    if args.export_csv:
        print(f"Exporting data to {args.export_csv}...")
        with args.export_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            # Match user request: alpha, N, alpha/N, final norm entanglement
            # Also adding friction as it's useful context
            writer.writerow(["alpha", "N", "alpha_over_N", "N_over_alpha", "final_norm_entanglement", "friction"])
            for p in points:
                writer.writerow([p.ar, p.n, p.ar / p.n, p.n / p.ar, p.ent_norm, p.friction])
        print("Export complete.")


    # Grouping Logic using --hue
    groups: Dict[float, List[DataPoint]] = {}
    
    for p in points:
        if args.hue == "friction":
            key = p.friction
        elif args.hue == "alpha":
            key = p.ar
        elif args.hue == "N":
            key = float(p.n)
        else:
            key = p.friction # Default
            
        # Float key matching with tolerance
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
    
    # Check if we should use integer formatting for legend if hue is N or alpha (usually ints)
    is_int_hue = (args.hue in ["alpha", "N"])

    # Determine global X range for consistent binning
    min_x, max_x = float("inf"), float("-inf")
    for p in points:
        if args.x_metric == "alpha_over_N":
            val = (p.ar / p.n) * args.z_factor
        else:
            val = p.n / (p.ar * args.z_factor)
        if val < min_x: min_x = val
        if val > max_x: max_x = val
        
    bin_edges = None
    if args.bins > 0:
        if args.bin_type == "log":
            # Handle 0 or negative if any? alpha/N should be positive.
            bin_edges = np.logspace(np.log10(min_x), np.log10(max_x), args.bins + 1)
        else:
            bin_edges = np.linspace(min_x, max_x, args.bins + 1)
    
    # User requested figsize
    fig, ax = plt.subplots(figsize=(args.figsize[0], args.figsize[1]))
    
    # Colormap for many lines?
    # If many groups, default cycle might repeat.
    if len(sorted_keys) > 10:
        # Use a colormap
        cm = plt.get_cmap('viridis')
        # normalize
        # c_norm = matplotlib.colors.Normalize(vmin=min(sorted_keys), vmax=max(sorted_keys))
        # But we iterate.
        pass

    for i, key in enumerate(sorted_keys):
        group = groups[key]
        
        if is_int_hue:
            label_str = f"{args.hue}={int(key)}"
        else:
            label_str = f"{args.hue}={key:g}"
            
        if args.hue == "friction":
            label_str = f"$\\mu={key:g}$" # Special pretty label for friction
        
        # Calculate X and Y
        # We might have duplicates for same X (different seeds) -> compute mean/std
        
        X_vals = []
        Y_vals = []
        for p in group:
            if args.x_metric == "alpha_over_N":
                X_vals.append((p.ar / p.n) * args.z_factor)
            else:
                X_vals.append(p.n / (p.ar * args.z_factor))
            Y_vals.append(p.ent_norm)
            
        X = []
        Y = []
        Yerr = []
        
        if args.bins > 0 and bin_edges is not None:
            # Binning
            X_vals = np.array(X_vals)
            Y_vals = np.array(Y_vals)
            
            # Digitize
            inds = np.digitize(X_vals, bin_edges)
            
            for j in range(1, len(bin_edges)):
                # Points in this bin
                in_bin = Y_vals[inds == j]
                if len(in_bin) > 0:
                    # X coord: Geometric mean of bin edges? Or mean of points?
                    # Usually geometric mean of edges for log plot
                    if args.bin_type == "log":
                        x_center = np.sqrt(bin_edges[j-1] * bin_edges[j])
                    else:
                        x_center = 0.5 * (bin_edges[j-1] + bin_edges[j])
                        
                    mean_y = np.mean(in_bin)
                    std_y = np.std(in_bin)
                    
                    X.append(x_center)
                    Y.append(mean_y)
                    Yerr.append(std_y)
        else:
            # No binning (or previous grouping logic)
            # Group by Exact X
            by_coord: Dict[float, List[float]] = {}
            for x_v, y_v in zip(X_vals, Y_vals):
                by_coord.setdefault(x_v, []).append(y_v)
                
            XYE = []
            for x_val, vals in by_coord.items():
                XYE.append((x_val, np.mean(vals), np.std(vals)))
            XYE.sort(key=lambda t: t[0])
            
            X = [t[0] for t in XYE]
            Y = [t[1] for t in XYE]
            Yerr = [t[2] for t in XYE]
        
        if X:
            # ax.errorbar(X, Y, yerr=Yerr, fmt='o-', label=f"$\\mu={fric:g}$", capsize=3, markersize=4, alpha=0.8)
            # User requested fill_between instead of errorbars
            p_line, = ax.plot(X, Y, 'o-', label=label_str, markersize=4, alpha=0.8)
            
            # Convert to numpy arrays for element-wise operations if not already
            X_np = np.array(X)
            Y_np = np.array(Y)
            Yerr_np = np.array(Yerr)
            
            ax.fill_between(X_np, Y_np - Yerr_np, Y_np + Yerr_np, color=p_line.get_color(), alpha=0.2, edgecolor="none")

    if args.x_metric == "alpha_over_N":
        label_base = r"$\alpha / N$" if args.z_factor == 1.0 else rf"$Z \alpha / N$"
        ax.set_xlabel(label_base)
        ax.set_title(f"Collapse: vs {label_base} ({args.hue})", fontsize=args.fontsize)
    else:
        label_base = r"$N / \alpha$" if args.z_factor == 1.0 else rf"$N / (Z \alpha)$"
        ax.set_xlabel(label_base)
        ax.set_title(f"Collapse: vs {label_base} (Z={args.z_factor:g}, {args.hue})", fontsize=args.fontsize)
        
    ax.set_ylabel("Final Norm. Ent.")
    ax.set_xscale("log")
    # ax.set_yscale("log") # Optional? Entanglement often 0-1 range roughly, linear might be ok for norm?
                           # Usually normalized entanglement is bounded, linear is prob fine.
                           
    ax.grid(True, alpha=0.3, which='both')
    
    # Legend might be big if many alphas
    if not args.no_legend:
        if len(sorted_keys) > 8:
             ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=args.fontsize-2)
        else:
             ax.legend(fontsize=args.fontsize-1)
    
    # Save
    out_name = args.output if args.output else Path("scaling_collapsed")
    # Ensure parent exists
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
    # Handle if user gave directory as output or filename without extension
    # Actually argparse type=Path gives full path.
    # If standard default usage:
    out_base = str(out_name)
    if out_base.endswith(".png") or out_base.endswith(".svg"):
        out_base = str(Path(out_base).with_suffix(""))
        
    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=300)
    fig.savefig(f"{out_base}.svg")
    print(f"Saved plots to:\n  {out_base}.png\n  {out_base}.svg")
    plt.close(fig)

if __name__ == "__main__":
    main()
