#!/usr/bin/env python3
"""analyze_crossing_mujoco.py

Analyzes the time-evolution of the minimum crossing number for MuJoCo runs.
Expects `endpoints.csv` to be present in each run folder.

Invocation:
    python3 parametric_study/analyze_crossing_mujoco.py <runs_root_dir> --binary build/compute_min_crossing
"""

import argparse
import csv
import re
import pickle
import subprocess
import multiprocessing
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional

# Match folder name: ..._AR123_F0.5...
# Match folder name: ..._AR123_F0.5...
AR_RE = re.compile(r"_AR(\d+)")
F_RE = re.compile(r"_mu([0-9.]+)")

class RunData:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_name = run_dir.name
        self.ar = 0
        self.friction = 0.0
        
        self.times: np.ndarray = np.array([])
        self.min_crossing: np.ndarray = np.array([])

    def parse_metadata(self):
        m_ar = AR_RE.search(self.run_name)
        if m_ar:
            self.ar = int(m_ar.group(1))
            
        m_f = F_RE.search(self.run_name)
        if m_f:
            self.friction = float(m_f.group(1))

    def ensure_metrics(self, binary_path: Path, refresh: bool = False):
        endpoints_path = self.run_dir / "endpoints_formatted.csv"
        metrics_path = self.run_dir / "crossing_metrics.csv"
        
        # Force cleanup if refresh requested
        if refresh:
            if metrics_path.exists(): metrics_path.unlink()
            if endpoints_path.exists(): endpoints_path.unlink()
            
        # Check for empty/invalid metrics file (header only is ~20 bytes)
        if metrics_path.exists():
            if metrics_path.stat().st_size < 30:
                print(f"Found incomplete metrics in {self.run_name}, regenerating...")
                metrics_path.unlink()
            else:
                return True
            
        if not endpoints_path.exists():
            # Try to generate from trajectory
            # Use separate function to write to endpoints_formatted.csv
            if not self.generate_endpoints_from_trajectory(endpoints_path):
                return False
        
        # Run C++ tool
        try:
            res = subprocess.run(
                [str(binary_path), str(endpoints_path), str(metrics_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return metrics_path.exists()
        except subprocess.CalledProcessError as e:
            print(f"Binary failed for {self.run_name}: {e.stderr}")
            return False
        except Exception as e:
            print(f"Error running binary for {self.run_name}: {e}")
            return False

    def generate_endpoints_from_trajectory(self, out_path: Path):
        # Use xipos and xmat as preferred source, containing N+1 bodies
        xipos_path = self.run_dir / "xipos_over_time.txt"
        xmat_path = self.run_dir / "xmat_over_time.txt"
        
        if not xipos_path.exists() or not xmat_path.exists():
            print(f"Missing xipos/xmat files in {self.run_name}")
            return False
            
        try:
            # Load
            X_flat = np.loadtxt(xipos_path, delimiter=',')
            M_flat = np.loadtxt(xmat_path, delimiter=',')
            
            if X_flat.ndim == 1: X_flat = X_flat[None, :]
            if M_flat.ndim == 1: M_flat = M_flat[None, :]
            
            n_steps = X_flat.shape[0]
            
            # Infer slicing based on n_rods
            # We expect cols = (n_rods + 1) * 3 for X, (n_rods + 1) * 9 for M
            # Or just n_rods if the file was saved differently.
            
            # Use regex metadata for n_rods check
            # self.ar is known. How to get n_rods? run name sometimes has it?
            # Or guess from shape.
            
            # If shape corresponds to N+1, slice.
            n_cols_x = X_flat.shape[1]
            n_cols_m = M_flat.shape[1]
            
            # Assume world body is index 0.
            # Check if dividing by 3 gives N+1
            n_bodies_x = n_cols_x // 3
            n_bodies_m = n_cols_m // 9
            
            # If we think there are 200 rods, and we see 201 bodies, slice.
            # If we iterate blindly:
            # Check consistency
            if n_bodies_x != n_bodies_m:
                print(f"Body count mismatch X({n_bodies_x}) vs M({n_bodies_m}) in {self.run_name}")
                return False
                
            # If we have exactly AR/F parsed, we might not know N from name.
            # But usually N is in the name "199_97_131" -> "N_N_N"? No that's seed.
            # Runs name: "mujoco_N200_sweep/..." -> N=200.
            # Assuming N=200 mostly.
            # Heuristic: slice off first body if > 1 body.
            
            if n_bodies_x > 1:
                # Slice [:, 3:] and [:, 9:]
                X_sliced = X_flat[:, 3:]
                M_sliced = M_flat[:, 9:]
                n_rods = n_bodies_x - 1
            else:
                X_sliced = X_flat
                M_sliced = M_flat
                n_rods = n_bodies_x
                
            # Reshape
            C = X_sliced.reshape(n_steps, n_rods, 3)
            R = M_sliced.reshape(n_steps, n_rods, 3, 3)
            
            rod_length = 1.0
            h = rod_length / 2.0
            local_z = np.array([0., 0., 1.])
            
            # Axis = R * z
            axes = np.einsum('tnij,j->tni', R, local_z)
            
            P1 = C - h * axes
            P2 = C + h * axes
            
            out_data = np.concatenate([P1, P2], axis=-1) # (T, N, 6)
            
            print(f"Generating endpoints for {self.run_name}: {n_steps} steps, {n_rods} rods.")
            
            with open(out_path, 'w') as f:
                # Write metadata comments
                rod_radius = 1.0 / self.ar if self.ar > 0 else 0.005
                f.write(f"#rod_radius={rod_radius}\n")
                f.write(f"#rod_length={rod_length}\n")
                # Write header required by C++ tool
                f.write("frame,id,x1,y1,z1,x2,y2,z2\n")
                
                for t in range(n_steps):
                    for i in range(n_rods):
                        # P1: out_data[t, i, 0:3]
                        # P2: out_data[t, i, 3:6]
                        row = out_data[t, i]
                        # frame, id, x1, y1, z1, x2, y2, z2
                        line = f"{t},{i},{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f}"
                        f.write(line + "\n")
            
            return True
            
        except Exception as e:
            print(f"Failed to generate endpoints for {self.run_name}: {e}")
            return False


    def load_metrics(self, dt: float = 0.0005, stride: int = 1000):
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
                        # Time = frame_index * stride * dt? 
                        # Wait, C++ tool frame index corresponds to line number in endpoints.csv
                        # In run_sims_with_mujoco.py, we write every stride.
                        # So line 0 is Step 0, line 1 is Step `stride`, etc.
                        # So actual time = frame * stride * dt.
                        # BUT the user usually passes just dt in previous scripts.
                        # Let's check compute_min_crossing logic. It just outputs frame number 0,1,2...
                        # So we need to scale manually if we want seconds.
                        # For consistency with existing plots, let's trust simple frame index or user provided scaling.
                        
                        # In run_sims_with_mujoco.py we write at (step-1)%stride==0. Step 1 is first?
                        # Simplification: time ~ f_idx * stride * dt
                        times.append(f_idx * stride * dt)
                        vals.append(val)
                    except ValueError:
                        continue
            
            if not times:
                return False
                
            self.times = np.array(times)
            self.min_crossing = np.array(vals)
            return True
        except Exception:
            return False

def compute_task(args):
    run_dir, binary_path, refresh = args
    r = RunData(run_dir)
    r.ensure_metrics(binary_path, refresh=refresh)

def plot_grouped(runs, group_key, line_key, group_label, line_label, out_dir, prefix):
    groups = {}
    for r in runs:
        gk = group_key(r)
        groups.setdefault(gk, []).append(r)
        
    for gk in sorted(groups.keys()):
        subset = groups[gk]
        
        # Further group by lines
        lines = {}
        for r in subset:
            lk = line_key(r)
            lines.setdefault(lk, []).append(r)
            
        fig, ax = plt.subplots(figsize=(6, 4))
        has_data = False
        
        for lk in sorted(lines.keys()):
            r_list = lines[lk]
            if not r_list: continue
            
            # Find min length
            lens = [len(r.times) for r in r_list]
            if not lens: continue
            min_len = min(lens)
            if min_len < 2: continue
            
            # Average
            # Assumes same sampling rate!
            t = r_list[0].times[:min_len]
            Y = np.vstack([r.min_crossing[:min_len] for r in r_list])
            mean = np.mean(Y, axis=0)
            std = np.std(Y, axis=0)
            
            p = ax.plot(t, mean, label=f"{line_label}={lk}")
            ax.fill_between(t, mean-std, mean+std, color=p[0].get_color(), alpha=0.2)
            has_data = True
            
        if has_data:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Min Crossing Number")
            ax.set_title(f"Crossing Number vs Time ({group_label}={gk})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            name = f"{prefix}_{group_label}{gk}.png"
            fig.tight_layout()
            fig.savefig(out_dir / name, dpi=200)
            print(f"Saved {name}")
            
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--binary", type=Path, default=Path("build/compute_min_crossing"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=0.0005)
    ap.add_argument("--refresh", action="store_true", help="Force re-analysis, ignoring cache.")
    args = ap.parse_args()
    
    if not args.binary.exists():
        print(f"Binary {args.binary} not found.")
        return

    cache_file = args.runs_dir / "analysis_crossing_cache.pkl"
    runs = []

    # Try loading from cache
    if cache_file.exists() and not args.refresh:
        try:
            print(f"Loading cached results from {cache_file}...")
            with open(cache_file, "rb") as f:
                runs = pickle.load(f)
            print(f"Loaded {len(runs)} runs from cache.")
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-analyzing...")
            runs = []

    if not runs:
        # 1. Identify runs
        print(f"Scanning {args.runs_dir} for runs (recursive)...")
        
        # Recursively find all folders containing xmat_over_time.txt
        candidates = [p.parent for p in args.runs_dir.rglob("xmat_over_time.txt")]
        if not candidates:
             candidates = [p.parent for p in args.runs_dir.rglob("endpoints.csv")]
            
        print(f"Found {len(candidates)} runs.")
        
        # 2. Compute metrics in parallel
        if candidates:
            with multiprocessing.Pool(args.jobs) as pool:
                tasks = [(d, args.binary, args.refresh) for d in candidates]
                pool.map(compute_task, tasks)
                
        # 3. Load
        for d in candidates:
            r = RunData(d)
            r.parse_metadata()
            if r.load_metrics(dt=args.dt, stride=args.stride):
                runs.append(r)
        
        if runs:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(runs, f)
                print(f"Saved {len(runs)} runs to {cache_file}")
            except Exception as e:
                print(f"Warning: Failed to save cache: {e}")
            
    if not runs:
        print("No valid metrics loaded.")
        return
        
    out_dir = args.runs_dir / "analysis_crossing"
    out_dir.mkdir(exist_ok=True)
    
    # Plot Mu fixed, AR lines
    plot_grouped(runs, lambda r: r.friction, lambda r: r.ar, "Friction", "AR", out_dir, "crossing_vs_time")
    
    # Plot AR fixed, Mu lines
    plot_grouped(runs, lambda r: r.ar, lambda r: r.friction, "AR", "Friction", out_dir, "crossing_vs_time")

if __name__ == "__main__":
    main()
