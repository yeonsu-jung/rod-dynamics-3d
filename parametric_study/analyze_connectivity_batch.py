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
# Check specific patterns based on user's examples
# "20260101-135327_180_271_742_AR300_Friction1.0_Kick0.1"
# "20260103-1625_RUN_keys199,97,131_N200_mu0.4000_AR1000_A0.100"
AR_RE = re.compile(r"_AR(\d+)")
F_RE = re.compile(r"[_ ](?:mu|Friction|F)([\d\.]+)")
N_RE = re.compile(r"(?:^|_)N(\d+)(?:_|$)")

def parse_metadata(path: Path):
    name = path.name
    parent = path.parent.name
    
    ar = None
    mu = None
    n = None
    
    # Check current folder name
    m_ar = AR_RE.search(name)
    if m_ar: ar = int(m_ar.group(1))
    
    m_f = F_RE.search(name)
    if m_f: mu = float(m_f.group(1))
    
    m_n = N_RE.search(name)
    if m_n: n = int(m_n.group(1))
    
    # Fallback to parent folder if missing (e.g. relax3rd_N100_sweep)
    if n is None:
        m_n = N_RE.search(parent)
        if m_n: n = int(m_n.group(1))
        
    return n, ar, mu

def run_extraction(args):
    perrod_path, binary_path, output_path, overwrite = args
    if output_path.exists() and not overwrite:
        return True
        
    try:
        cmd = [str(binary_path), str(perrod_path), str(output_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting {perrod_path}: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Error executing {binary_path}: {e}")
        return False

def load_and_compute_metric(args):
    connectivity_path, dt = args
    if not connectivity_path.exists():
        return None
        
    try:
        # Load edges: frame -> set of edges
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
        jaccards_prev = [] # vs t-1
        jaccards_t0 = []   # vs t=0
        
        e0 = frames_dict[sorted_frames[0]]
        
        for i in range(len(sorted_frames)):
            f_curr = sorted_frames[i]
            e_curr = frames_dict[f_curr]
            
            # 1. vs prev (for i>0)
            if i > 0:
                f_prev = sorted_frames[i-1]
                e_prev = frames_dict[f_prev]
                
                u_p = len(e_prev.union(e_curr))
                i_p = len(e_prev.intersection(e_curr))
                
                j_p = i_p / u_p if u_p > 0 else 1.0
                if len(e_prev) == 0 and len(e_curr) == 0: j_p = 1.0
            else:
                j_p = 1.0 # t=0 vs t=0? or undefined? Let's say 1.0
            
            # 2. vs t=0
            u_0 = len(e0.union(e_curr))
            i_0 = len(e0.intersection(e_curr))
            j_0 = i_0 / u_0 if u_0 > 0 else 1.0
            if len(e0) == 0 and len(e_curr) == 0: j_0 = 1.0
            
            t = f_curr * dt
            times.append(t)
            jaccards_prev.append(j_p)
            jaccards_t0.append(j_0)
            
        return np.array(times), np.array(jaccards_prev), np.array(jaccards_t0)
        
    except Exception as e:
        print(f"Analysis error {connectivity_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Batch Connectivity Analysis")
    parser.add_argument("directories", nargs='+', type=Path, help="Directories to scan")
    parser.add_argument("--binary", type=Path, default=Path("build/extract_connectivity_perrod"), help="Path to C++ tool")
    parser.add_argument("--dt", type=float, default=0.0005, help="Time step")
    parser.add_argument("--jobs", "-j", type=int, default=8, help="Parallel jobs")
    parser.add_argument("--refresh", action="store_true", help="Re-run extraction")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results/connectivity"), help="Output directory for plots")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.binary.exists():
        print(f"Binary not found at {args.binary}")
        return

    # 1. Scan
    print("Scanning directories...")
    tasks = []
    run_meta = [] # (path, n, ar, mu)
    
    for d in args.directories:
        for p in d.rglob("perrod.csv"):
            n, ar, mu = parse_metadata(p.parent)
            if n is None or ar is None or mu is None:
                continue
                
            out_csv = p.parent / "connectivity.csv"
            tasks.append((p, args.binary, out_csv, args.refresh))
            run_meta.append({
                'path': out_csv,
                'n': n,
                'ar': ar,
                'mu': mu,
                'id': p.parent.name
            })
            
    print(f"Found {len(tasks)} runs.")
    
    # 2. Extract (Parallel)
    if tasks:
        with Pool(args.jobs) as pool:
            results = pool.map(run_extraction, tasks)
            
    # 3. Analyze (Parallel)
    print("Analyzing connectivity stability...")
    analysis_tasks = [(m['path'], args.dt) for m in run_meta]
    
    with Pool(args.jobs) as pool:
        metrics = pool.map(load_and_compute_metric, analysis_tasks)
        
    # Aggegate
    data = []
    for meta, res in zip(run_meta, metrics):
        if res is None: continue
        t, j_prev, j_t0 = res
        meta['time'] = t
        meta['jaccard_prev'] = j_prev
        meta['jaccard_t0'] = j_t0
        data.append(meta)
        
    print(f"Successfully analyzed {len(data)} runs.")
    
    # 4. Filter and Plot
    
    def plot_metric(data_list, metric_key, metric_label, filename_suffix):
        # Group by (N, Mu)
        groups_n_mu = {}
        for d in data_list:
            key = (d['n'], d['mu'])
            groups_n_mu.setdefault(key, []).append(d)
            
        for (n, mu), runs in groups_n_mu.items():
            plt.figure(figsize=(8, 5))
            ars = sorted(list(set(r['ar'] for r in runs)))
            cmap = plt.cm.viridis
            
            for r in runs:
                ar = r['ar']
                if len(ars) > 1:
                    c_val = (ars.index(ar) / (len(ars)-1))
                else:
                    c_val = 0.5
                color = cmap(c_val)
                plt.plot(r['time'], r[metric_key], color=color, alpha=0.6, linewidth=1)
                
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=cmap(i/(len(ars)-1) if len(ars)>1 else 0.5), label=f"AR={a}") for i, a in enumerate(ars)]
            if len(legend_elements) > 10: legend_elements = [legend_elements[0], legend_elements[-1]]
            
            plt.legend(handles=legend_elements, title="Aspect Ratio")
            plt.xlabel("Time (s)")
            plt.ylabel(metric_label)
            plt.title(f"{metric_label} (N={n}, $\mu$={mu})")
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            
            out_name = args.output_dir / f"{filename_suffix}_vs_time_N{n}_mu{mu}.png"
            plt.savefig(out_name, dpi=150)
            plt.close()

        # Group by (AR, Mu)
        groups_ar_mu = {}
        for d in data_list:
            key = (d['ar'], d['mu'])
            groups_ar_mu.setdefault(key, []).append(d)
            
        for (ar, mu), runs in groups_ar_mu.items():
            plt.figure(figsize=(8, 5))
            ns = sorted(list(set(r['n'] for r in runs)))
            cmap = plt.cm.plasma
            
            for r in runs:
                n = r['n']
                if len(ns) > 1:
                    c_val = (ns.index(n) / (len(ns)-1))
                else:
                    c_val = 0.5
                color = cmap(c_val)
                plt.plot(r['time'], r[metric_key], color=color, alpha=0.6, linewidth=1)
                
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=cmap(i/(len(ns)-1) if len(ns)>1 else 0.5), label=f"N={n_val}") for i, n_val in enumerate(ns)]
            ncol = 1
            if len(legend_elements) > 10: ncol=2
            
            plt.legend(handles=legend_elements, title="Rod Count (N)", ncol=ncol)
            plt.xlabel("Time (s)")
            plt.ylabel(metric_label)
            plt.title(f"{metric_label} (AR={ar}, $\mu$={mu})")
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            
            out_name = args.output_dir / f"{filename_suffix}_vs_time_AR{ar}_mu{mu}.png"
            plt.savefig(out_name, dpi=150)
            plt.close()

    # Plot both metrics
    print("Plotting Stability (t vs t-1)...")
    plot_metric(data, 'jaccard_prev', "Jaccard Index (t vs t-1)", "stability_prev")
    
    print("Plotting Stability (t vs t0)...")
    plot_metric(data, 'jaccard_t0', "Jaccard Index (t vs t0)", "stability_t0")

if __name__ == "__main__":
    main()
