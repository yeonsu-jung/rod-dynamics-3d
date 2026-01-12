#!/usr/bin/env python3
"""analyze_reldisp_evolution.py

Analyzes the time-evolution of squared relative displacement (`reldisp_sq`)
from `output.csv` files in a batch of runs.

Output:
  - Plots of RelDispSq vs Time.
  - Grouped by Friction (lines = ARs).
  - Grouped by AR (lines = Frictions).
"""

import argparse
import csv
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
        self.reldisp_sq: np.ndarray = np.array([])

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
        output_path = self.run_dir / "output.csv"
        if not output_path.exists():
            return False
            
        try:
            # Using standard csv to avoid pandas dependency if simplistic
            times = []
            vals = []
            
            with output_path.open("r") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or "reldisp_sq" not in reader.fieldnames:
                    return False
                
                for row in reader:
                    try:
                        frame = int(float(row.get("frame", 0)))
                        val = float(row.get("reldisp_sq", 0.0))
                        times.append(frame * dt)
                        vals.append(val)
                    except ValueError:
                        continue
            
            if not times:
                return False
                
            self.times = np.array(times)
            self.reldisp_sq = np.array(vals)
            return True
            
        except Exception as e:
            print(f"Error loading {output_path}: {e}")
            return False

def plot_grouped(
    runs: List[RunData], 
    group_key_func, 
    line_key_func, 
    group_label: str, 
    line_label: str, 
    out_dir: Path,
    prefix: str,
    highlight_lines: Optional[List[float]] = None,
    log_y: bool = False
):
    """
    Generic plotting function.
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
                
            # Filter if we only want highlights
            if highlight_lines is not None:
                if not any(abs(lk - h) < 1e-4 for h in highlight_lines):
                    continue

            min_len = min(len(r.times) for r in subset)
            if min_len < 2:
                continue
                
            # Stack
            t = subset[0].times[:min_len]
            Y = np.vstack([r.reldisp_sq[:min_len] for r in subset])
            
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
        ax.set_ylabel("Squared Relative Displacement")
        ax.set_title(f"RelDispSq vs Time ({group_label}={gk})")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
        
        if log_y:
            ax.set_yscale("log")
        
        safe_gk = str(gk).replace(".", "p")
        out_name = f"{prefix}_{safe_gk}.png"
        fig.tight_layout()
        fig.savefig(out_dir / out_name, dpi=150)
        plt.close(fig)
        print(f"Generated {out_name}")

def main():
    parser = argparse.ArgumentParser(description="Analyze reldisp_sq evolution.")
    parser.add_argument("batch_dir", type=Path, help="Directory containing run subfolders")
    parser.add_argument("--dt", type=float, default=0.0005, help="Timestep (default 0.0005)")
    parser.add_argument("--log", action="store_true", help="Plot Y-axis in log scale")
    args = parser.parse_args()
    
    if not args.batch_dir.exists():
        print("Batch directory not found.")
        return
        
    runs = []
    print(f"Scanning {args.batch_dir}...")
    
    for d in args.batch_dir.iterdir():
        if d.is_dir() and (d / "output.csv").exists():
            r = RunData(d)
            r.load_metadata()
            if r.load_metrics(args.dt):
                runs.append(r)
                
    if not runs:
        print("No Valid runs found (need output.csv).")
        return
        
    print(f"Loaded {len(runs)} runs.")
    
    out_dir = args.batch_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    
    # 1. Plot: Fixed Friction, Lines = AR
    print("Generating RelDispSq vs Time (grouped by Friction)...")
    plot_grouped(
        runs,
        group_key_func=lambda r: r.friction,
        line_key_func=lambda r: r.ar,
        group_label="Mu",
        line_label="AR",
        out_dir=out_dir,
        prefix="reldisp_evol_mu",
        log_y=args.log
    )
    
    # 2. Plot: Fixed AR, Lines = Friction
    print("Generating RelDispSq vs Time (grouped by AR)...")
    plot_grouped(
        runs,
        group_key_func=lambda r: r.ar,
        line_key_func=lambda r: r.friction,
        group_label="AR",
        line_label="Mu",
        out_dir=out_dir,
        prefix="reldisp_evol_ar",
        log_y=args.log
    )

if __name__ == "__main__":
    main()
