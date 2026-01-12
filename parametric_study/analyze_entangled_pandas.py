#!/usr/bin/env python3
"""analyze_entangled_pandas.py

Optimized version of analyze_entangled_n200.py using Pandas for fast CSV loading.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Regex to parse directory names
# Matches: ..._SEED123_N200..._AR100...
# We focus on extracting seed and AR.
# Updated to match names like: ..._AR100... or ..._AR100_Friction0.4...
RUN_RE = re.compile(r"_([0-9]+_[0-9]+_[0-9]+)_AR(\d+)")

def get_run_n(run_dir: Path) -> int:
    """Extract N (rod count) from scene.json, default 200."""
    scene_path = run_dir / "scene.json"
    if scene_path.exists():
        try:
            data = json.loads(scene_path.read_text())
            # Look for populate count
            # scene -> populate -> count
            return int(data.get("scene", {}).get("populate", {}).get("count", 200))
        except Exception:
            pass
    return 200

def get_run_friction(run_dir: Path) -> float:
    """Extract friction from scene.json, default 0.4."""
    scene_path = run_dir / "scene.json"
    if scene_path.exists():
        try:
            data = json.loads(scene_path.read_text())
            # submit_entangled.py writes to physics.soft_contact.mu
            # Fallback to older location just in case
            phy = data.get("physics", {})
            if "soft_contact" in phy and "mu" in phy["soft_contact"]:
                return float(phy["soft_contact"]["mu"])
            return float(phy.get("friction", 0.4))
        except Exception:
            pass
    return 0.4

class RunData:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_name = run_dir.name
        
        # Parse metadata
        m = RUN_RE.search(self.run_name)
        if m:
            self.seed = m.group(1)
            self.ar = int(m.group(2))
        else:
            self.seed = "unknown"
            self.ar = 0
            
        self.n_rods = get_run_n(run_dir)
        self.friction = get_run_friction(run_dir)
            
        # Placeholders for data
        self.frame: np.ndarray = np.array([])
        self.ke: np.ndarray = np.array([])
        self.ent_sum: np.ndarray = np.array([])
        self.ent_pairs: np.ndarray = np.array([])
        self.reldisp_sq: np.ndarray = np.array([])
        self.gyration_sq: np.ndarray = np.array([])
        
        # New metrics
        self.rms_contact_distance: Optional[np.ndarray] = None
        self.max_cluster_fraction: Optional[np.ndarray] = None

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size
        self.max_size = 1

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Merge smaller into larger
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            self.max_size = max(self.max_size, self.size[root_i])
            return True
        return False

def load_network_metrics(path: Path, n_rods: int, force: bool = False, burst_gap: int = 50) -> Tuple[Optional[Dict[int, float]], Optional[Dict[int, float]]]:
    """Load network.csv using Pandas."""
    run_dir = path.parent
    cache_path = run_dir / "network_metrics.json"

    # Try loading cache
    if not force and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            rms_out = {int(k): v for k, v in data.get("rms_out", {}).items()}
            cluster_out = {int(k): v for k, v in data.get("cluster_out", {}).items()}
            return (rms_out if rms_out else None), (cluster_out if cluster_out else None)
        except Exception:
            pass

    if not path.exists():
        return None, None

    print(f"Reading {path.name} with pandas... (N={n_rods})")
    try:
        # Load CSV with fast engine
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None, None

    if df.empty:
        return None, None

    # Ensure required columns
    # Ensure required columns (u,v OR rod_i,rod_j)
    if 'rod_i' in df.columns and 'rod_j' in df.columns:
        df = df.rename(columns={'rod_i': 'u', 'rod_j': 'v'})
    
    required_cols = {'frame', 'u', 'v'}
    if not required_cols.issubset(df.columns):
        print(f"Missing columns in {path}: Found {list(df.columns)}")
        return None, None
        
    has_dist = 'distance' in df.columns

    # 1. Compute RMS Contact Distance per frame
    rms_out: Dict[int, float] = {}
    
    if has_dist:
        # Group by frame and compute RMS
        # sqrt(mean(distance^2))
        # Pandas optimization: pre-calculate dist sq
        df['dist_sq'] = df['distance'] ** 2
        grouped = df.groupby('frame')['dist_sq'].mean()
        # Take sqrt of the means
        rms_series = np.sqrt(grouped)
        rms_out = rms_series.to_dict()
    
    # 2. Cluster Analysis (Union-Find on Bursts)
    cluster_out: Dict[int, float] = {}
    
    # Get sorted unique frames
    unique_frames = np.sort(df['frame'].unique())
    if len(unique_frames) == 0:
        return rms_out, cluster_out

    # Define bursts logic
    # Group frames: if f[i] - f[i-1] > gap, new burst
    # Vectorized burst detection
    if len(unique_frames) > 1:
        diffs = np.diff(unique_frames)
        # Identify indices where gap > burst_gap
        # Using 0-referenced indexing for unique_frames
        split_indices = np.where(diffs > burst_gap)[0] + 1
        bursts = np.split(unique_frames, split_indices)
    else:
        bursts = [unique_frames]

    print(f"Processing {len(bursts)} bursts...")

    # For each burst, we need all edges (u, v) from all frames in the burst
    # We can do this efficiently by filtering the DF
    
    # Sort DF by frame to make slicing faster?
    df = df.sort_values('frame')
    # Set frame as index for fast lookups if needed. 
    # Actually, strictly slicing might be faster if we know start/end indices.
    # But usually just `isin` or range query is okay.
    
    # Optimization: Iterate bursts, define min/max frame for query to reduce boolean mask search space
    
    for burst in bursts:
        if len(burst) == 0:
            continue
            
        f_start = burst[0]
        f_end = burst[-1]
        
        # Get edges for this burst
        # Only select rows where frame is in burst (trivial if contiguous range)
        # Since bursts are by definition clusters of frames, usually range [f_start, f_end] covers it 
        # IF there are no sparse frames inside the burst. Our definition of burst allows gaps <= burst_gap.
        # But logically we want all rows with frame >= f_start and frame <= f_end that are IN the df.
        
        # Filter DF
        mask = (df['frame'] >= f_start) & (df['frame'] <= f_end)
        burst_rows = df.loc[mask]
        
        # Perform Union-Find on these edges
        uf = UnionFind(n_rods)
        
        # Iterate over numpy array values for speed (faster than iterrows)
        # Columns u and v
        # Assuming 0-indexed rods? Check data. Typically 0 to N-1.
        # Use vectorized operations if possible? UnionFind is inherently sequential/iterative.
        # But we can iterate loops in python over numpy array which is faster than DictReader.
        
        edges = burst_rows[['u', 'v']].to_numpy()
        
        for u, v in edges:
            if 0 <= u < n_rods and 0 <= v < n_rods:
                uf.union(int(u), int(v))
        
        # Record result for the LAST frame of the burst (representative)
        rep_frame = int(burst[-1])
        cluster_out[rep_frame] = float(uf.max_size) / float(n_rods)

    # Save cache
    try:
        with cache_path.open("w") as f:
            json.dump({
                "rms_out": rms_out, 
                "cluster_out": cluster_out,
                "burst_gap": burst_gap
            }, f)
    except Exception as e:
        print(f"Failed to write cache: {e}")

    return rms_out, cluster_out

def _positive_floor(y: np.ndarray) -> float:
    finite_pos = y[np.isfinite(y) & (y > 0)]
    if finite_pos.size == 0:
        return 1e-12
    return float(np.nanmin(finite_pos))

def apply_scale(ax: plt.Axes, scale: str) -> None:
    if scale == "linear":
        ax.set_xscale("linear")
        ax.set_yscale("linear")
    elif scale == "semilogy":
        ax.set_xscale("linear")
        ax.set_yscale("log")
    elif scale == "loglog":
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        raise ValueError(f"Unknown scale: {scale}")

def safe_mean_std_band(mean: np.ndarray, std: np.ndarray, floor: float):
    lower = mean - std
    upper = mean + std
    lower = np.where(np.isfinite(lower), np.maximum(lower, floor), np.nan)
    upper = np.where(np.isfinite(upper), np.maximum(upper, floor), np.nan)
    return lower, upper

def plot_run_timeseries(run: RunData, out_dir: Path, scale: str, dt: float, xlabel: str = "Time (s)") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Time axis
    t = run.frame * dt
    # For log-x plots, avoid t=0 problem
    x = t
    if scale == "loglog":
        x = np.where(t == 0, t[1] if len(t) > 1 else 1e-6, t)

    rms_gyr = np.sqrt(np.maximum(run.gyration_sq, 0.0))
    rms_reldisp = np.sqrt(np.maximum(run.reldisp_sq, 0.0))
    
    # Normalized entanglement
    ent_norm = np.full_like(run.ent_sum, np.nan)
    valid_pairs = run.ent_pairs > 0
    ent_norm[valid_pairs] = run.ent_sum[valid_pairs] / run.ent_pairs[valid_pairs]

    have_net = run.rms_contact_distance is not None
    have_clust = run.max_cluster_fraction is not None
    
    # Determine number of rows
    base_rows = 5
    extra_rows = (1 if have_net else 0) + (1 if have_clust else 0)
    nrows = base_rows + extra_rows
    
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3*nrows), sharex=True)
    if nrows == 1: axes = [axes] # robust check

    idx = 0
    axes[idx].plot(x, run.ke)
    axes[idx].set_ylabel("KE")
    idx += 1

    axes[idx].plot(x, run.ent_sum)
    axes[idx].set_ylabel("Entanglement sum")
    idx += 1

    axes[idx].plot(x, ent_norm)
    axes[idx].set_ylabel("Norm Ent (sum/pairs)")
    idx += 1

    axes[idx].plot(x, rms_reldisp)
    axes[idx].set_ylabel("RMS rel. disp")
    idx += 1

    axes[idx].plot(x, rms_gyr)
    axes[idx].set_ylabel("RMS gyration")
    idx += 1

    if have_net:
        axes[idx].plot(x, run.rms_contact_distance)
        axes[idx].set_ylabel("RMS contact dist")
        idx += 1
        
    if have_clust:
        axes[idx].plot(x, run.max_cluster_fraction)
        axes[idx].set_ylabel("Max Cluster Frac")
        idx += 1

    axes[nrows-1].set_xlabel(xlabel)

    for ax in axes:
        apply_scale(ax, scale)
        ax.grid(True, alpha=0.25, which="both")

    title = f"{run.run_name} (seed={run.seed}, AR={run.ar}) [{scale}]"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"timeseries_{run.run_name}_{scale}.png", dpi=160)
    plt.close(fig)

def plot_overlaid_by_ar(runs: List[RunData], out_dir: Path, scale: str, dt: float, xlabel: str = "Time (s)") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by AR
    by_ar: Dict[int, List[RunData]] = {}
    for r in runs:
        by_ar.setdefault(r.ar, []).append(r)

    def get_ent_norm(r: RunData):
        val = np.full_like(r.ent_sum, np.nan)
        msk = r.ent_pairs > 0
        val[msk] = r.ent_sum[msk] / r.ent_pairs[msk]
        return val

    metrics = [
        ("KE", lambda r: r.ke, "KE"),
        ("ent_sum", lambda r: r.ent_sum, "Entanglement sum"),
        ("ent_norm", get_ent_norm, "Norm Ent (sum/pairs)"),
        ("rms_reldisp", lambda r: np.sqrt(np.maximum(r.reldisp_sq, 0.0)), "RMS rel. disp"),
        ("rms_gyration", lambda r: np.sqrt(np.maximum(r.gyration_sq, 0.0)), "RMS gyration"),
    ]

    # Optional network-based metric
    if any(r.rms_contact_distance is not None for r in runs):
        metrics.append(
            (
                "rms_contact_distance",
                lambda r: (r.rms_contact_distance if r.rms_contact_distance is not None else np.full_like(r.frame, np.nan, dtype=float)),
                "RMS contact distance",
            )
        )
    if any(r.max_cluster_fraction is not None for r in runs):
        metrics.append(
            (
                "max_cluster_fraction",
                lambda r: (r.max_cluster_fraction if r.max_cluster_fraction is not None else np.full_like(r.frame, np.nan, dtype=float)),
                "Max Cluster Fraction (size/N)",
            )
        )

    for key, get_series, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for ar in sorted(by_ar.keys()):
            group = by_ar[ar]
            # Align by truncating to min length
            min_len = min(len(r.frame) for r in group)
            if min_len < 2:
                continue

            x0 = group[0].frame[:min_len] * dt
            # If log, mask 0
            x = x0
            if scale == "loglog":
                 x = np.where(x0 == 0, x0[1] if len(x0) > 1 else 1e-6, x0)

            Y = np.vstack([get_series(r)[:min_len] for r in group])
            mean = np.nanmean(Y, axis=0)
            std = np.nanstd(Y, axis=0)

            if scale in ("semilogy", "loglog"):
                floor = _positive_floor(mean)
                mean_plot = np.where(np.isfinite(mean), np.maximum(mean, floor), np.nan)
                ax.plot(x, mean_plot, label=f"AR={ar} (n={len(group)})")
                lower, upper = safe_mean_std_band(mean, std, floor)
                ax.fill_between(x, lower, upper, alpha=0.15)
            else:
                ax.plot(x, mean, label=f"AR={ar} (n={len(group)})")
                ax.fill_between(x, mean - std, mean + std, alpha=0.15)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs time (mean±std over seeds) [{scale}]")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        apply_scale(ax, scale)

        fig.tight_layout()
        fig.savefig(out_dir / f"overlaid_by_ar_{key}_{scale}.png", dpi=160)
        plt.close(fig)

def plot_metrics_vs_ar(runs: List[RunData], out_dir: Path):
    """Plot final metrics vs AR, grouping by Friction."""
    # Data: friction -> AR -> list of values
    # Extract end values
    
    # Metrics to plot
    metrics = {
        "final_ent_sum": lambda r: r.ent_sum[-1] if r.ent_sum.size > 0 else np.nan,
        "final_ent_norm": lambda r: (r.ent_sum[-1] / r.ent_pairs[-1]) if (r.ent_sum.size > 0 and r.ent_pairs[-1] > 0) else np.nan,
        "final_rms_gyration": lambda r: math.sqrt(r.gyration_sq[-1]) if r.gyration_sq.size > 0 else np.nan,
        "final_max_cluster_fraction": lambda r: r.max_cluster_fraction[-1] if (r.max_cluster_fraction is not None and r.max_cluster_fraction.size > 0) else np.nan,
        "final_contacts": lambda r: r.rms_contact_distance[-1] if (r.rms_contact_distance is not None and r.rms_contact_distance.size > 0) else np.nan
    }
    
    # Gather unique frictions
    all_frictions = sorted(list(set(r.friction for r in runs)))
    
    for m_name, m_func in metrics.items():
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        # For each friction, plot a series
        for fric in all_frictions:
            # Gather (AR, value) pairs for this friction
            xy = []
            for r in runs:
                if abs(r.friction - fric) < 1e-6:
                    val = m_func(r)
                    if math.isfinite(val):
                        xy.append((r.ar, val))
            
            if not xy:
                continue
                
            # Sort by AR
            xy.sort(key=lambda t: t[0])
            # Group by AR to handle multiples (err bars)
            from itertools import groupby
            X, Y, Yerr = [], [], []
            for ar, group in groupby(xy, key=lambda t: t[0]):
                vals = [t[1] for t in group]
                X.append(ar)
                Y.append(np.mean(vals))
                Yerr.append(np.std(vals))
            
            if X:
                ax.errorbar(X, Y, yerr=Yerr, fmt='o-', label=f"mu={fric}")

        ax.set_xlabel("Aspect Ratio (AR)")
        ax.set_ylabel(m_name.replace("_", " "))
        ax.set_xscale("log")
        ax.set_title(f"{m_name} vs AR")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        out_path = out_dir / f"{m_name}_vs_ar.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

def plot_metrics_vs_friction(runs: List[RunData], out_dir: Path):
    """Plot final metrics vs Friction, grouping by AR."""
    metrics = {
        "final_ent_sum": lambda r: r.ent_sum[-1] if r.ent_sum.size > 0 else np.nan,
        "final_ent_norm": lambda r: (r.ent_sum[-1] / r.ent_pairs[-1]) if (r.ent_sum.size > 0 and r.ent_pairs[-1] > 0) else np.nan,
        "final_rms_gyration": lambda r: math.sqrt(r.gyration_sq[-1]) if r.gyration_sq.size > 0 else np.nan,
        "final_max_cluster_fraction": lambda r: r.max_cluster_fraction[-1] if (r.max_cluster_fraction is not None and r.max_cluster_fraction.size > 0) else np.nan,
        "final_contacts": lambda r: r.rms_contact_distance[-1] if (r.rms_contact_distance is not None and r.rms_contact_distance.size > 0) else np.nan
    }
    
    # Gather unique ARs
    all_ars = sorted(list(set(r.ar for r in runs)))
    
    for m_name, m_func in metrics.items():
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        for ar in all_ars:
            xy = []
            for r in runs:
                if r.ar == ar:
                    val = m_func(r)
                    if math.isfinite(val):
                        xy.append((r.friction, val))
            
            if not xy:
                continue
                
            xy.sort(key=lambda t: t[0]) # sort by friction
            
            from itertools import groupby
            X, Y, Yerr = [], [], []
            for f, group in groupby(xy, key=lambda t: t[0]):
                vals = [t[1] for t in group]
                X.append(f)
                Y.append(np.mean(vals))
                Yerr.append(np.std(vals))
                
            if X:
                ax.errorbar(X, Y, yerr=Yerr, fmt='o-', label=f"AR={ar}")
                
        ax.set_xlabel("Friction Coefficient")
        ax.set_ylabel(m_name.replace("_", " "))
        ax.set_title(f"{m_name} vs Friction")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        out_path = out_dir / f"{m_name}_vs_friction.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="Directory containing run subfolders (e.g. runs/batch_name)")
    ap.add_argument("--force", action="store_true", help="Force re-calculation of network metrics")
    ap.add_argument("--burst-gap", type=int, default=50, help="Max gap between frames to merge into one burst")
    ap.add_argument("--use-network", action="store_true", help="Analyzes network.csv if present")
    ap.add_argument("--dt", type=float, default=1.0, help="Time step size")
    ap.add_argument("--timescale", type=float, default=None, help="Optional characteristic time scale")
    ap.add_argument("--per-run", action="store_true", help="Generate per-run plots")
    ap.add_argument("--scales", type=str, default="linear,semilogy,loglog", help="Plot scales")
    ap.add_argument("--out", type=Path, default=None, help="Output directory")
    args = ap.parse_args()

    if not args.run_dir.exists():
        raise SystemExit(f"Error: {args.run_dir} does not exist")

    print(f"Scanning {args.run_dir}...")
    run_subdirs = sorted([d for d in args.run_dir.iterdir() if d.is_dir() and (d / "output.csv").exists()])
    
    if not run_subdirs:
        print("No valid run subdirectories found (must contain output.csv).")
        return

    runs: List[RunData] = []
    
    for d in run_subdirs:
        rd = RunData(d)
        
        # Load output.csv
        try:
            # Standard python read is fine for output.csv as it's small (strided)
            # Or use pandas too? output.csv is usually small. Stick to numpy loadtxt/pandas.
            # Using pandas is easier.
            out_df = pd.read_csv(d / "output.csv")
            # Columns: frame,contacts,KE,max_overlap,gyration_sq,reldisp_sq,ent_sum,ent_pairs
            
            rd.frame = out_df['frame'].to_numpy()
            rd.ke = out_df['KE'].to_numpy()
            rd.ent_sum = out_df['ent_sum'].to_numpy()
            rd.ent_pairs = out_df['ent_pairs'].to_numpy()
            
            # Direct read (assuming CSV has pre-calculated values matching original script's expectations)
            rd.reldisp_sq = out_df['reldisp_sq'].to_numpy()
            rd.contacts = out_df['contacts'].to_numpy() # Original script used contacts
            rd.max_overlap = out_df['max_overlap'].to_numpy()
            rd.gyration_sq = out_df['gyration_sq'].to_numpy()
            
        except Exception as e:
            print(f"Error reading output.csv in {d.name}: {e}")
            continue

        # Load Network Metrics (Cached or Computed)
        rms_map, clust_map = load_network_metrics(d / "network.csv", rd.n_rods, force=args.force, burst_gap=args.burst_gap)
        
        # Map these back to output frames
        # Create array of same size as rd.frame
        # If a frame isn't in network metrics, user NaN
        
        if rms_map:
            # Map dictionary values to the time series frames
            # Use list comp or numpy map
            rd.rms_contact_distance = np.array([rms_map.get(f, np.nan) for f in rd.frame])
        else:
            rd.rms_contact_distance = np.full(rd.frame.shape, np.nan)
            
        if clust_map:
            rd.max_cluster_fraction = np.array([clust_map.get(f, np.nan) for f in rd.frame])
        else:
            rd.max_cluster_fraction = np.full(rd.frame.shape, np.nan)
            
        runs.append(rd)

    if not runs:
        print("No valid runs loaded.")
        return

    # Create Analysis Dir
    out_dir = args.run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    
    # Save Summary CSV
    summary_path = out_dir / "summary.csv"
    print(f"Writing summary to {summary_path}...")
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run", "seed", "AR", "N", "friction", 
            "frames", "time_end", "KE0", "KE_end", 
            "ent_sum_end", "ent_norm_end", 
            "rms_reldisp_end", "rms_gyr_end", 
            "rms_contact_distance_end", "max_cluster_frac_end"
        ])
        
        for r in runs:
            # Extract end values
            if len(r.frame) == 0: continue
            
            idx = -1
            idx = -1
            time_end = float(r.frame[idx]) * args.dt
            
            ent_sum_end = float(r.ent_sum[idx])
            pairs_end = float(r.ent_pairs[idx])
            ent_norm_end = ent_sum_end / pairs_end if pairs_end > 0 else float("nan")
            
            rms_reldisp = math.sqrt(max(float(r.reldisp_sq[idx]), 0.0))
            rms_gyr = math.sqrt(max(float(r.gyration_sq[idx]), 0.0))
            
            rms_contact = float(r.rms_contact_distance[idx]) if r.rms_contact_distance is not None else float("nan")
            clust_frac = float(r.max_cluster_fraction[idx]) if r.max_cluster_fraction is not None else float("nan")
            
            w.writerow([
                r.run_name, r.seed, r.ar, r.n_rods, r.friction,
                len(r.frame), time_end,
                float(r.ke[0]), float(r.ke[idx]),
                ent_sum_end, ent_norm_end,
                rms_reldisp, rms_gyr,
                rms_contact, clust_frac
            ])

    # Generate Timeseries Plots
    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    for s in scales:
        if s not in ["linear", "semilogy", "loglog"]:
            continue
            
        print(f"Generating overlaid plots ({s})...")
        plot_overlaid_by_ar(runs, out_dir, s, args.dt)
        
    if args.per_run:
        print("Generating per-run plots...")
        for r in runs:
            for s in scales:
                plot_run_timeseries(r, out_dir, s, args.dt)

    # Generate Summary Plots
    print("Generating summary plots...")
    plot_metrics_vs_ar(runs, out_dir)
    plot_metrics_vs_friction(runs, out_dir)
    
    print("Done.")

if __name__ == "__main__":
    main()
