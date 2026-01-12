#!/usr/bin/env python3
"""analyze_entanglement_mujoco.py

Analyzes entanglement metrics for MuJoCo runs using the logic from first_analysis(legacy).py.
Calculates:
  - Total Entanglement (sum of abs pairwise linking)
  - Sliding Ratio
  - Max Cluster Size
  
Outputs:
  - analysis/entanglement_summary.csv
  - Plots of metrics vs Time
"""

import argparse
import csv
import re
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import jax.numpy as jnp

# Add study directory to path to import util
sys.path.append(str(Path(__file__).parent.parent / "study"))
try:
    from util import analyze_pairwise_dist_entanglement, get_clusters
except ImportError as e:
    print(f"Error: Could not import util.py. Ensure it exists in 'study/' directory.\nDetails: {e}")
    sys.exit(1)

# Match folder name: ..._AR123_F0.5...
AR_RE = re.compile(r"_AR(\d+)")
F_RE = re.compile(r"_mu([0-9.]+)")

class RunData:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_name = run_dir.name
        self.ar = 0
        self.friction = 0.0
        self.n_rods = 0
        
        self.times = []
        self.entanglement = []
        self.sliding_ratio = []
        self.max_cluster_size = []
        self.valid = False

    def parse_metadata(self):
        m_ar = AR_RE.search(self.run_name)
        if m_ar:
            self.ar = int(m_ar.group(1))
        
        m_f = F_RE.search(self.run_name)
        if m_f:
            self.friction = float(m_f.group(1))

    def analyze(self, original_data_path_template):
        # 1. Load Data
        xipos_path = self.run_dir / "xipos_over_time.txt"
        xmat_path = self.run_dir / "xmat_over_time.txt"
        time_path = self.run_dir / "time_points.txt"
        
        if not xipos_path.exists() or not xmat_path.exists():
            print(f"Skipping {self.run_name}: missing trace files")
            return False

        try:
            # Load
            xipos_flat = np.loadtxt(xipos_path, delimiter=',')
            xmat_flat = np.loadtxt(xmat_path, delimiter=',')
            times_arr = np.loadtxt(time_path, delimiter=',')
            
            if xipos_flat.ndim == 1: xipos_flat = xipos_flat[None, :]
            if xmat_flat.ndim == 1: xmat_flat = xmat_flat[None, :]
            if times_arr.ndim == 0: times_arr = np.array([times_arr])
            
            n_steps = xipos_flat.shape[0]
            
            # Legacy Logic: Slice off first body (world)
            # xipos: (T, (N+1)*3) -> slice [:, 3:] -> (T, N*3)
            # xmat:  (T, (N+1)*9) -> slice [:, 9:] -> (T, N*9)
            
            # Heuristic check: if num cols > 3, assume first is world.
            if xipos_flat.shape[1] > 3:
                xipos_sliced = xipos_flat[:, 3:]
                xmat_sliced = xmat_flat[:, 9:]
            else:
                xipos_sliced = xipos_flat
                xmat_sliced = xmat_flat
                
            n_rods = xipos_sliced.shape[1] // 3
            self.n_rods = n_rods
            
            X = xipos_sliced.reshape(n_steps, n_rods, 3)
            M = xmat_sliced.reshape(n_steps, n_rods, 3, 3)
            
            local_z = np.array([0., 0., 1.])
            rod_len = 1.0
            h = rod_len / 2.0
            
            for t in range(n_steps):
                # Centers
                c = X[t] # (N, 3)
                
                # Axes = R @ z
                axes = np.einsum('nij,j->ni', M[t], local_z)
                
                # Endpoints
                p1 = c - h * axes
                p2 = c + h * axes
                
                nodes = np.hstack([p1, p2]) # (N, 6)
                
                # JAX compute
                dist, ent, pwd, pwa = analyze_pairwise_dist_entanglement(jnp.array(nodes), n_rods)
                self.entanglement.append(float(ent))
                self.times.append(float(times_arr[t]))
                
            self.valid = True
            print(f"Analyzed {self.run_name}: {n_steps} steps.")
            return True
            
        except Exception as e:
            print(f"Error analyzing {self.run_name}: {e}")
            return False

def plot_metric(runs, metric_name, ylabel, out_dir):
    # Group by Friction, plot vs Time (lines=AR)
    groups = {}
    for r in runs:
        groups.setdefault(r.friction, []).append(r)
        
    for f in sorted(groups.keys()):
        subset = groups[f]
        fig, ax = plt.subplots()
        for r in subset:
            if not r.valid: continue
            val = getattr(r, metric_name)
            if not val: continue
            ax.plot(r.times, val, label=f"AR={r.ar}")
            
        ax.set_xlabel("Time")
        ax.set_title(f"{ylabel} vs Time (Mu={f:.2f})")
        ax.legend()
        fig.savefig(out_dir / f"{metric_name}_mu{f:.2f}.png")
        plt.close(fig)

def plot_comparison_by_ar(runs, metric_name, ylabel, out_dir):
    # Group by AR, plot vs Time (lines=Friction)
    groups = {}
    for r in runs:
        groups.setdefault(r.ar, []).append(r)
        
    for ar in sorted(groups.keys()):
        subset = groups[ar]
        # Sort subset by friction for consistent legend order
        subset.sort(key=lambda x: x.friction)
        
        fig, ax = plt.subplots()
        for r in subset:
            if not r.valid: continue
            val = getattr(r, metric_name)
            if not val: continue
            ax.plot(r.times, val, marker='o', markersize=3, label=f"Mu={r.friction:.2f}")
            
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Time (AR={ar})")
        ax.legend()
        fig.savefig(out_dir / f"{metric_name}_combined_AR{ar}.png")
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--refresh", action="store_true", help="Force re-analysis, ignoring cache.")
    args = ap.parse_args()
    
    cache_file = args.runs_dir / "analysis_cache.pkl"
    results = []
    
    # Try loading from cache
    if cache_file.exists() and not args.refresh:
        try:
            print(f"Loading cached results from {cache_file}...")
            with open(cache_file, "rb") as f:
                results = pickle.load(f)
            print(f"Loaded {len(results)} runs from cache.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-analyzing...")
            results = []

    if not results:
        print(f"Scanning {args.runs_dir} for runs (recursive)...")
        
        # Recursively find all folders containing xmat_over_time.txt
        candidates = [p.parent for p in args.runs_dir.rglob("xmat_over_time.txt")]
        
        if not candidates:
            print("No run directories found (checked for xmat_over_time.txt).")
            return

        # Analyze
        for d in candidates:
            r = RunData(d)
            r.parse_metadata()
            print(f"Analyzing {r.run_name}...")
            if r.analyze(None):
                results.append(r)
        
        # Save to cache if we have results
        if results:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(results, f)
                print(f"Saved {len(results)} runs to {cache_file}")
            except Exception as e:
                print(f"Warning: Failed to save cache: {e}")
                
    if not results:
        print("No valid runs.")
        return
        
    out_dir = args.runs_dir / "analysis_entanglement"
    out_dir.mkdir(exist_ok=True)
    
    plot_metric(results, "entanglement", "Entanglement", out_dir)
    plot_comparison_by_ar(results, "entanglement", "Entanglement", out_dir)
    
    # Save Summary
    with open(out_dir / "entanglement_summary.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "ar", "friction", "entanglement_end"])
        for r in results:
            writer.writerow([r.run_name, r.ar, r.friction, r.entanglement[-1]])

if __name__ == "__main__":
    main()
