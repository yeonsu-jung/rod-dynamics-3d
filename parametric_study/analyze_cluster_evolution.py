#!/usr/bin/env python3
"""analyze_cluster_evolution.py

Analyzes the time-evolution of the largest connected component (cluster) size
from `network_metrics.json` files in a batch of runs.

Output:
  - Plots of Cluster Size (number of rods) vs Time.
  - Grouped by Friction (lines = ARs).
  - Grouped by AR (lines = Frictions).
"""

import argparse
import json
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Match ..._AR123...
AR_RE = re.compile(r"_AR(\d+)")

class RunData:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_name = run_dir.name
        self.n_rods = 200 # Default
        self.friction = 0.4 # Default
        self.ar = 0
        self.seed = "unknown"
        
        self.times: np.ndarray = np.array([])
        self.cluster_sizes: np.ndarray = np.array([]) # Absolute count (N * fraction)

    def load_metadata(self):
        # Parse AR from name
        m = AR_RE.search(self.run_name)
        if m:
            self.ar = int(m.group(1))
            
        # Parse scene.json
        scene_path = self.run_dir / "scene.json"
        if scene_path.exists():
            try:
                data = json.loads(scene_path.read_text())
                if "scene" in data and "populate" in data["scene"]:
                     self.n_rods = int(data["scene"]["populate"].get("count", self.n_rods))
                
                if "physics" in data and "soft_contact" in data["physics"]:
                    self.friction = float(data["physics"]["soft_contact"].get("mu", self.friction))
            except Exception:
                pass

    def load_metrics(self, dt: float = 0.0005):
        metrics_path = self.run_dir / "network_metrics.json"
        if not metrics_path.exists():
            return False
            
        try:
            data = json.loads(metrics_path.read_text())
            cluster_out = data.get("cluster_out", {})
            if not cluster_out:
                return False
                
            # Convert keys (frame str) to sorted lists
            frames = sorted([int(k) for k in cluster_out.keys()])
            
            t_list = []
            sz_list = []
            
            for f in frames:
                frac = float(cluster_out[str(f)])
                count = frac * self.n_rods
                t_list.append(f * dt)
                sz_list.append(count)
                
            self.times = np.array(t_list)
            self.cluster_sizes = np.array(sz_list)
            return True
            
        except Exception as e:
            print(f"Error loading {metrics_path}: {e}")
            return False

def plot_grouped(
    runs: List[RunData], 
    group_key_func, 
    line_key_func, 
    group_label: str, 
    line_label: str, 
    out_dir: Path,
    prefix: str,
    highlight_lines: Optional[List[float]] = None
):
    """
    Generic plotting function.
    group_key_func: lambda r: r.friction  (Creates one plot per friction)
    line_key_func:  lambda r: r.ar        (Creates lines for each AR)
    """
    
    # 1. Group runs
    groups = {}
    for r in runs:
        gk = group_key_func(r)
        groups.setdefault(gk, []).append(r)
        
    for gk in sorted(groups.keys()):
        run_group = groups[gk]
        
        # 2. Sub-group by line key (average over seeds)
        lines = {}
        for r in run_group:
            lk = line_key_func(r)
            lines.setdefault(lk, []).append(r)
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        has_data = False
        sorted_lks = sorted(lines.keys())
        
        for lk in sorted_lks:
            subset = lines[lk]
            if not subset:
                continue
                
            # Filter if we only want highlights (naive float check)
            if highlight_lines is not None:
                # check if lk matches any highlight with tolerance
                if not any(abs(lk - h) < 1e-4 for h in highlight_lines):
                    continue

            # Compute mean/std over time
            # We need to align time series. Simple approach: use the time points of the first run 
            # and interpolate others, or just assume they are consistent if from same sweep.
            # Assuming consistent frames here for simplicity, or truncating to min length.
            
            min_len = min(len(r.times) for r in subset)
            if min_len < 2:
                continue
                
            # Stack
            # Taking time from first
            t = subset[0].times[:min_len]
            Y = np.vstack([r.cluster_sizes[:min_len] for r in subset])
            
            mean = np.mean(Y, axis=0)
            std = np.std(Y, axis=0)
            
            lbl = f"{line_label}={lk}"
            p = ax.plot(t, mean, label=lbl)
            ax.fill_between(t, mean-std, mean+std, color=p[0].get_color(), alpha=0.2)
            has_data = True
            
        if not has_data:
            plt.close(fig)
            continue
            
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Largest Cluster Size (Rods)")
        ax.set_title(f"Cluster Size vs Time ({group_label}={gk})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Sanitize filename
        safe_gk = str(gk).replace(".", "p")
        out_name = f"{prefix}_{safe_gk}.png"
        fig.tight_layout()
        fig.savefig(out_dir / out_name, dpi=150)
        plt.close(fig)
        print(f"Generated {out_name}")

def main():
    parser = argparse.ArgumentParser(description="Analyze cluster size evolution.")
    parser.add_argument("batch_dir", type=Path, help="Directory containing run subfolders")
    parser.add_argument("--dt", type=float, default=0.0005, help="Timestep (default 0.0005)")
    args = parser.parse_args()
    
    if not args.batch_dir.exists():
        print("Batch directory not found.")
        return
        
    runs = []
    print(f"Scanning {args.batch_dir}...")
    
    for d in args.batch_dir.iterdir():
        if d.is_dir() and (d / "network_metrics.json").exists():
            r = RunData(d)
            r.load_metadata()
            if r.load_metrics(args.dt):
                runs.append(r)
                
    if not runs:
        print("No Valid runs found (need network_metrics.json).")
        return
        
    print(f"Loaded {len(runs)} runs.")
    
    out_dir = args.batch_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    
    # 1. Plot: Fixed Friction, Lines = AR
    print("Generating Cluster vs Time (grouped by Friction)...")
    plot_grouped(
        runs,
        group_key_func=lambda r: r.friction,
        line_key_func=lambda r: r.ar,
        group_label="Mu",
        line_label="AR",
        out_dir=out_dir,
        prefix="cluster_evol_mu"
    )
    
    # 2. Plot: Fixed AR, Lines = Friction
    print("Generating Cluster vs Time (grouped by AR)...")
    plot_grouped(
        runs,
        group_key_func=lambda r: r.ar,
        line_key_func=lambda r: r.friction,
        group_label="AR",
        line_label="Mu",
        out_dir=out_dir,
        prefix="cluster_evol_ar"
    )

if __name__ == "__main__":
    main()
