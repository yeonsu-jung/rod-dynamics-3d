#!/usr/bin/env python3
"""
Analyze connectivity and minimum pair distance for MuJoCo simulation data.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re
from pathlib import Path
from multiprocessing import Pool
import sys

# Regex for parsing folder names
AR_RE = re.compile(r"_AR(\d+)")
F_RE = re.compile(r"[_ ](?:mu|Friction|F)([\d\.]+)")
N_RE = re.compile(r"(?:^|_)N(\d+)(?:_|$)")

def parse_metadata(path: Path):
    """Extract N, AR, mu from folder name."""
    name = path.name
    parent = path.parent.name
    
    ar = None
    mu = None
    n = None
    
    m_ar = AR_RE.search(name)
    if m_ar: ar = int(m_ar.group(1))
    
    m_f = F_RE.search(name)
    if m_f: mu = float(m_f.group(1))
    
    m_n = N_RE.search(name)
    if m_n: n = int(m_n.group(1))
    
    if n is None:
        m_n = N_RE.search(parent)
        if m_n: n = int(m_n.group(1))
        
    return n, ar, mu

def load_time_points(time_file: Path):
    """Load time points from time_points.txt."""
    if not time_file.exists():
        return None
    try:
        times = np.loadtxt(time_file)
        # Ensure it's always an array, even if single value
        if times.ndim == 0:
            times = np.array([times])
        return times
    except:
        return None

def run_connectivity_extraction(args):
    """Run C++ connectivity extraction tool."""
    endpoints_path, binary_path, output_path, overwrite = args
    if output_path.exists() and not overwrite:
        return True
        
    try:
        cmd = [str(binary_path), str(endpoints_path), str(output_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting {endpoints_path}: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Error executing {binary_path}: {e}")
        return False

def compute_min_pair_distance(endpoints_path: Path):
    """
    Compute minimum pairwise distance between rod endpoints over time.
    Returns: dict {frame: min_distance}
    """
    try:
        df = pd.read_csv(endpoints_path, comment='#')
        
        # Group by frame
        grouped = df.groupby('frame')
        min_distances = {}
        
        for frame_id, group in grouped:
            # Extract endpoints
            p1 = group[['x1', 'y1', 'z1']].values
            p2 = group[['x2', 'y2', 'z2']].values
            
            n_rods = len(group)
            if n_rods < 2:
                min_distances[frame_id] = np.nan
                continue
            
            # Compute all pairwise distances
            # For each rod i, compute distance to all other rods j
            # Distance between two line segments: min over 4 endpoint pairs
            
            min_dist = np.inf
            
            for i in range(n_rods):
                for j in range(i+1, n_rods):
                    # 4 distances: p1i-p1j, p1i-p2j, p2i-p1j, p2i-p2j
                    d1 = np.linalg.norm(p1[i] - p1[j])
                    d2 = np.linalg.norm(p1[i] - p2[j])
                    d3 = np.linalg.norm(p2[i] - p1[j])
                    d4 = np.linalg.norm(p2[i] - p2[j])
                    
                    pair_min = min(d1, d2, d3, d4)
                    min_dist = min(min_dist, pair_min)
            
            min_distances[frame_id] = min_dist
            
        return min_distances
        
    except Exception as e:
        print(f"Error computing min distance for {endpoints_path}: {e}")
        return None

def analyze_connectivity_jaccard(connectivity_path: Path, time_points: np.ndarray = None):
    """
    Compute Jaccard similarity vs t-1 and vs t0.
    Returns: (times, jaccard_prev, jaccard_t0)
    """
    if not connectivity_path.exists():
        return None
        
    try:
        df = pd.read_csv(connectivity_path)
        grouped = df.groupby('frame')
        
        frames_dict = {}
        for frame_id, group in grouped:
            s = group['source'].values
            t = group['target'].values
            edges = np.sort(np.vstack((s, t)).T, axis=1)
            frames_dict[frame_id] = set(map(tuple, edges))
            
        sorted_frames = sorted(frames_dict.keys())
        if len(sorted_frames) < 2:
            return None
            
        times = []
        jaccards_prev = []
        jaccards_t0 = []
        
        e0 = frames_dict[sorted_frames[0]]
        
        for i, f_curr in enumerate(sorted_frames):
            e_curr = frames_dict[f_curr]
            
            # vs prev
            if i > 0:
                f_prev = sorted_frames[i-1]
                e_prev = frames_dict[f_prev]
                u_p = len(e_prev.union(e_curr))
                i_p = len(e_prev.intersection(e_curr))
                j_p = i_p / u_p if u_p > 0 else 1.0
            else:
                j_p = 1.0
            
            # vs t0
            u_0 = len(e0.union(e_curr))
            i_0 = len(e0.intersection(e_curr))
            j_0 = i_0 / u_0 if u_0 > 0 else 1.0
            
            # Map frame to time
            if time_points is not None and f_curr < len(time_points):
                t = time_points[f_curr]
            else:
                t = float(f_curr)
            
            times.append(t)
            jaccards_prev.append(j_p)
            jaccards_t0.append(j_0)
            
        return np.array(times), np.array(jaccards_prev), np.array(jaccards_t0)
        
    except Exception as e:
        print(f"Error analyzing connectivity {connectivity_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="MuJoCo Connectivity Analysis")
    parser.add_argument("directories", nargs='+', type=Path, help="Directories to scan")
    parser.add_argument("--binary", type=Path, default=Path("build/extract_connectivity"), 
                       help="Path to connectivity extraction tool")
    parser.add_argument("--jobs", "-j", type=int, default=8, help="Parallel jobs")
    parser.add_argument("--refresh", action="store_true", help="Re-run extraction")
    parser.add_argument("--output-dir", type=Path, default=Path("mujoco_connectivity_analysis"), 
                       help="Output directory for plots")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.binary.exists():
        print(f"Binary not found at {args.binary}")
        print("Please compile: g++ -std=c++17 -O3 -fopenmp src/tools/extract_connectivity_perrod.cpp -o build/extract_connectivity_perrod")
        return

    # 1. Scan for endpoints_formatted.csv
    print("Scanning directories...")
    tasks = []
    run_meta = []
    
    for d in args.directories:
        for p in d.rglob("endpoints_formatted.csv"):
            n, ar, mu = parse_metadata(p.parent)
            if n is None or ar is None or mu is None:
                continue
                
            connectivity_csv = p.parent / "connectivity_mujoco.csv"
            tasks.append((p, args.binary, connectivity_csv, args.refresh))
            
            # Load time points
            time_file = p.parent / "time_points.txt"
            time_points = load_time_points(time_file)
            
            run_meta.append({
                'endpoints_path': p,
                'connectivity_path': connectivity_csv,
                'time_points': time_points,
                'n': n,
                'ar': ar,
                'mu': mu,
                'id': p.parent.name
            })
            
    print(f"Found {len(tasks)} runs.")
    
    # 2. Extract connectivity (parallel)
    if tasks:
        print("Extracting connectivity...")
        with Pool(args.jobs) as pool:
            results = pool.map(run_connectivity_extraction, tasks)
    
    # 3. Compute metrics
    print("Computing metrics...")
    data = []
    
    for meta in run_meta:
        # Connectivity Jaccard
        conn_result = analyze_connectivity_jaccard(
            meta['connectivity_path'], 
            meta['time_points']
        )
        
        # Min pair distance
        min_dist_dict = compute_min_pair_distance(meta['endpoints_path'])
        
        if conn_result is None and min_dist_dict is None:
            continue
            
        entry = {
            'n': meta['n'],
            'ar': meta['ar'],
            'mu': meta['mu'],
            'id': meta['id']
        }
        
        if conn_result is not None:
            t, j_prev, j_t0 = conn_result
            entry['time'] = t
            entry['jaccard_prev'] = j_prev
            entry['jaccard_t0'] = j_t0
        
        if min_dist_dict is not None:
            # Map to time array
            frames = sorted(min_dist_dict.keys())
            if meta['time_points'] is not None and len(meta['time_points']) > 0:
                times_dist = np.array([meta['time_points'][f] if f < len(meta['time_points']) else f 
                                      for f in frames])
            else:
                times_dist = np.array(frames, dtype=float)
            
            distances = np.array([min_dist_dict[f] for f in frames])
            entry['time_dist'] = times_dist
            entry['min_distance'] = distances
        
        data.append(entry)
    
    print(f"Successfully analyzed {len(data)} runs.")
    
    # 4. Plot
    def plot_metric(data_list, metric_key, time_key, ylabel, filename_suffix, ylim=None):
        """Generic plotting function."""
        # Group by (N, Mu)
        groups_n_mu = {}
        for d in data_list:
            if metric_key not in d or time_key not in d:
                continue
            key = (d['n'], d['mu'])
            groups_n_mu.setdefault(key, []).append(d)
            
        for (n, mu), runs in groups_n_mu.items():
            plt.figure(figsize=(8, 5))
            ars = sorted(list(set(r['ar'] for r in runs)))
            cmap = plt.cm.viridis
            
            for r in runs:
                ar = r['ar']
                c_val = (ars.index(ar) / (len(ars)-1)) if len(ars) > 1 else 0.5
                color = cmap(c_val)
                plt.plot(r[time_key], r[metric_key], color=color, alpha=0.6, linewidth=1)
                
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=cmap(i/(len(ars)-1) if len(ars)>1 else 0.5), 
                                     label=f"AR={a}") for i, a in enumerate(ars)]
            if len(legend_elements) > 10: 
                legend_elements = [legend_elements[0], legend_elements[-1]]
            
            plt.legend(handles=legend_elements, title="Aspect Ratio")
            plt.xlabel("Time (s)")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} (N={n}, $\\mu$={mu})")
            if ylim: plt.ylim(ylim)
            plt.grid(True, alpha=0.3)
            
            out_name = args.output_dir / f"{filename_suffix}_N{n}_mu{mu}.png"
            plt.savefig(out_name, dpi=150)
            plt.close()
            print(f"Saved {out_name}")

        # Group by (AR, Mu)
        groups_ar_mu = {}
        for d in data_list:
            if metric_key not in d or time_key not in d:
                continue
            key = (d['ar'], d['mu'])
            groups_ar_mu.setdefault(key, []).append(d)
            
        for (ar, mu), runs in groups_ar_mu.items():
            plt.figure(figsize=(8, 5))
            ns = sorted(list(set(r['n'] for r in runs)))
            cmap = plt.cm.plasma
            
            for r in runs:
                n = r['n']
                c_val = (ns.index(n) / (len(ns)-1)) if len(ns) > 1 else 0.5
                color = cmap(c_val)
                plt.plot(r[time_key], r[metric_key], color=color, alpha=0.6, linewidth=1)
                
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=cmap(i/(len(ns)-1) if len(ns)>1 else 0.5), 
                                     label=f"N={n_val}") for i, n_val in enumerate(ns)]
            ncol = 2 if len(legend_elements) > 10 else 1
            
            plt.legend(handles=legend_elements, title="Rod Count (N)", ncol=ncol)
            plt.xlabel("Time (s)")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} (AR={ar}, $\\mu$={mu})")
            if ylim: plt.ylim(ylim)
            plt.grid(True, alpha=0.3)
            
            out_name = args.output_dir / f"{filename_suffix}_AR{ar}_mu{mu}.png"
            plt.savefig(out_name, dpi=150)
            plt.close()
            print(f"Saved {out_name}")
    
    # Plot connectivity metrics
    print("Plotting connectivity stability (t vs t-1)...")
    plot_metric(data, 'jaccard_prev', 'time', "Jaccard Index (t vs t-1)", 
                "connectivity_prev", ylim=(0, 1.05))
    
    print("Plotting connectivity stability (t vs t0)...")
    plot_metric(data, 'jaccard_t0', 'time', "Jaccard Index (t vs t0)", 
                "connectivity_t0", ylim=(0, 1.05))
    
    # Plot min distance
    print("Plotting minimum pair distance...")
    plot_metric(data, 'min_distance', 'time_dist', "Minimum Pair Distance", 
                "min_pair_distance")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
