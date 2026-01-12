#!/usr/bin/env python3
"""analyze_crossing_evolution.py

Analyzes the time-evolution of the minimum crossing number
from `endpoints.csv` files by invoking `compute_min_crossing`.

Output:
  - Plots of Min Crossing Number vs Time.
  - Grouped by Friction (lines = ARs).
  - Grouped by AR (lines = Frictions).
"""

import argparse
import csv
import json
import re
import subprocess
import time
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
        
        self.times: np.ndarray = np.array([])
        self.min_crossing: np.ndarray = np.array([])

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

    def ensure_metrics(self, binary_path: Path):
        endpoints_path = self.run_dir / "endpoints.csv"
        metrics_path = self.run_dir / "crossing_metrics.csv"
        
        if not metrics_path.exists():
            if not endpoints_path.exists():
                return False
            
            # Run compute tool
            print(f"Computing crossing metrics for {self.run_name}...")
            try:
                subprocess.run(
                    [str(binary_path), str(endpoints_path), str(metrics_path)],
                    check=True
                )
            except Exception as e:
                print(f"Failed to compute metrics for {self.run_name}: {e}")
                return False
                
        return metrics_path.exists()

    def load_metrics(self, dt: float = 0.0005):
        metrics_path = self.run_dir / "crossing_metrics.csv"
        if not metrics_path.exists():
            return False
            
        try:
            times = []
            vals = []
            
            with metrics_path.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        f_idx = int(row["frame"])
                        val = float(row["min_crossing"])
                        times.append(f_idx * dt)
                        vals.append(val)
                    except ValueError:
                        continue
            
            if not times:
                return False
                
            self.times = np.array(times)
            self.min_crossing = np.array(vals)
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
    groups = {}
    for r in runs:
        gk = group_key_func(r)
        groups.setdefault(gk, []).append(r)
        
    for gk in sorted(groups.keys()):
        run_group = groups[gk]
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
            
            if highlight_lines is not None:
                if not any(abs(lk - h) < 1e-4 for h in highlight_lines):
                    continue

            min_len = min(len(r.times) for r in subset)
            if min_len < 2:
                continue
                
            t = subset[0].times[:min_len]
            Y = np.vstack([r.min_crossing[:min_len] for r in subset])
            
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
        ax.set_ylabel("Min Crossing Number")
        ax.set_title(f"Min Crossing vs Time ({group_label}={gk})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        safe_gk = str(gk).replace(".", "p")
        out_name = f"{prefix}_{safe_gk}.png"
        fig.tight_layout()
        fig.savefig(out_dir / out_name, dpi=150)
        plt.close(fig)
        print(f"Generated {out_name}")

def compute_task(args):
    run_dir, binary_path = args
    endpoints_path = run_dir / "endpoints.csv"
    metrics_path = run_dir / "crossing_metrics.csv"
    
    try:
        # print(f"Computing {run_dir.name}...")
        subprocess.run(
            [str(binary_path), str(endpoints_path), str(metrics_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        print(f"Failed {run_dir.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze min crossing evolution.")
    parser.add_argument("batch_dir", type=Path, help="Directory containing run subfolders")
    parser.add_argument("--binary", type=Path, required=True, help="Path to compute_min_crossing binary")
    parser.add_argument("--dt", type=float, default=0.0005, help="Timestep (default 0.0005)")
    parser.add_argument("--jobs", type=int, default=0, help="Parallel jobs (0=auto)")
    args = parser.parse_args()
    
    if not args.batch_dir.exists():
        print("Batch directory not found.")
        return
        
    if not args.binary.exists():
        print("Binary not found. Please compile compute_min_crossing.")
        return
        
    runs = []
    print(f"Scanning {args.batch_dir}...")
    
    # Compute tasks
    tasks = []
    
    # Scan for candidates
    all_runs_potential = []
    for d in args.batch_dir.iterdir():
        if d.is_dir() and ((d / "endpoints.csv").exists() or (d / "crossing_metrics.csv").exists()):
            r = RunData(d)
            all_runs_potential.append(r)
            if not (r.run_dir / "crossing_metrics.csv").exists():
                 if (r.run_dir / "endpoints.csv").exists():
                     tasks.append((r.run_dir, args.binary))
    
    # Run parallel computation
    if tasks:
        import multiprocessing
        # threads = min(len(tasks), multiprocessing.cpu_count())
        # Use a reasonable number of threads e.g. 48 for seas_compute if requested
        # Or default to cpu_count
        threads = args.jobs if args.jobs > 0 else multiprocessing.cpu_count()
        print(f"Computing metrics for {len(tasks)} runs using {threads} workers...")
        
        with multiprocessing.Pool(threads) as pool:
            pool.map(compute_task, tasks)
            
    # Load all
    runs = []
    for r in all_runs_potential:
        r.load_metadata()
        if r.load_metrics(args.dt):
            runs.append(r)
                
    if not runs:
        print("No Valid runs found.")
        return
        
    print(f"Loaded {len(runs)} runs.")
    
    out_dir = args.batch_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    
    # 1. Plot: Fixed Friction, Lines = AR
    print("Generating Crossing vs Time (grouped by Friction)...")
    plot_grouped(
        runs,
        group_key_func=lambda r: r.friction,
        line_key_func=lambda r: r.ar,
        group_label="Mu",
        line_label="AR",
        out_dir=out_dir,
        prefix="crossing_evol_mu"
    )
    
    # 2. Plot: Fixed AR, Lines = Friction
    print("Generating Crossing vs Time (grouped by AR)...")
    plot_grouped(
        runs,
        group_key_func=lambda r: r.ar,
        line_key_func=lambda r: r.friction,
        group_label="AR",
        line_label="Mu",
        out_dir=out_dir,
        prefix="crossing_evol_ar"
    )

if __name__ == "__main__":
    main()
