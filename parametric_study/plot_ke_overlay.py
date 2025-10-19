#!/usr/bin/env python3
"""
plot_ke_overlay.py

Generate an overlaid plot of KE vs time (frame) for multiple runs in the parametric study.

Usage:
  python3 plot_ke_overlay.py --runs-root /path/to/runs --job-name my_job --outdir ./plots
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import re

RUN_RE = re.compile(r"RUN_rods.*_AR(?P<alpha>[-+0-9.]+).*_N(?P<N>[-+0-9.]+).*_C(?P<C>[-+0-9.]+).*_L(?P<L>[-+0-9.]+)")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', type=Path, default=Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs"))
    ap.add_argument('--job-name', type=str, default='', help='Job name subdirectory under runs-root')
    ap.add_argument('--glob', default='*RUN_rods*', help='glob to select run folders')
    ap.add_argument('--outdir', type=Path, default=Path('plots'))
    ap.add_argument('--max-runs', type=int, default=10, help='Max number of runs to plot to avoid clutter')
    return ap.parse_args()

def discover_runs(runs_root: Path, patt: str):
    if not runs_root.exists():
        raise SystemExit(f"Runs root not found: {runs_root}")
    for d in sorted(runs_root.glob(patt)):
        if d.is_dir():
            yield d

def parse_run_name(name: str):
    m = RUN_RE.search(name)
    if m:
        return {k: float(v) if '.' in v else int(v) for k, v in m.groupdict().items()}
    return {}

def main():
    args = parse_args()
    runs_root = args.runs_root
    if args.job_name:
        runs_root = runs_root / args.job_name
    runs = list(discover_runs(runs_root, args.glob))[:args.max_runs]

    plt.figure(figsize=(10, 6))
    for rd in runs:
        prof = rd / 'profile.csv'
        if not prof.exists():
            continue
        try:
            df = pd.read_csv(prof)
            alpha = parse_run_name(rd.name).get('alpha', 'unknown')
            plt.plot(df['frame'], df['KE'], label=f'AR={alpha}')
        except Exception as e:
            print(f"Skipping {rd.name}: {e}")

    plt.xlabel('Frame (time)')
    plt.ylabel('Kinetic Energy (KE)')
    plt.title('Overlaid KE vs Time for Parametric Runs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / 'ke_overlay.png'
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == '__main__':
    main()
