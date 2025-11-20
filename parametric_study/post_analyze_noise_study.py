#!/usr/bin/env python3
"""
post_analyze_noise_study.py

Analyze results from parametric noise study: Effects of noise amplitude (fSigma) and friction on KE evolution.

Scans run directories, extracts KE data from profile.csv, fits exponential decay, generates plots.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import argparse

def _as_float(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return None
    return None

def find_friction_heuristic(obj):
    """Best-effort extraction of friction coefficient from a scene dict.
    Tries common locations/keys first, then falls back to DFS search.
    """
    if not isinstance(obj, (dict, list)):
        return float('nan')

    # 1) Prefer explicit path: scene -> bodies[0] -> friction|friction_s|friction_d
    try:
        scene = obj.get('scene', {}) if isinstance(obj, dict) else {}
        bodies = scene.get('bodies', []) if isinstance(scene, dict) else []
        if isinstance(bodies, list) and bodies:
            b0 = bodies[0]
            if isinstance(b0, dict):
                for k in ('friction', 'friction_s', 'friction_d', 'mu', 'MU'):
                    if k in b0:
                        v = _as_float(b0[k])
                        if v is not None:
                            return v
    except Exception:
        pass

    # 2) Generic DFS for likely keys
    def dfs(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (int, float, str)) and ('fric' in k.lower() or k.lower() == 'mu'):
                    fv = _as_float(v)
                    if fv is not None:
                        return fv
            for v in o.values():
                r = dfs(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for it in o:
                r = dfs(it)
                if r is not None:
                    return r
        return None

    r = dfs(obj)
    return r if r is not None else float('nan')

def find_fSigma_heuristic(obj):
    """Best-effort extraction of noise amplitude fSigma from a scene dict.
    Tries common locations/keys first, then falls back to DFS search.
    """
    if not isinstance(obj, (dict, list)):
        return float('nan')

    # 1) Prefer explicit path: scene -> randomForce -> fSigma|tauMag
    try:
        scene = obj.get('scene', {}) if isinstance(obj, dict) else {}
        rf = scene.get('randomForce', {}) if isinstance(scene, dict) else {}
        if isinstance(rf, dict):
            for k in ('fSigma', 'tauMag', 'sigma', 'noiseSigma'):
                if k in rf:
                    v = _as_float(rf[k])
                    if v is not None:
                        return v
    except Exception:
        pass

    # 2) Generic DFS for likely keys
    def dfs(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (int, float, str)) and (k.lower() in ('fsigma', 'tausigma', 'taumag', 'sigma') or 'sigma' in k.lower()):
                    fv = _as_float(v)
                    if fv is not None:
                        return fv
            for v in o.values():
                r = dfs(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for it in o:
                r = dfs(it)
                if r is not None:
                    return r
        return None

    r = dfs(obj)
    return r if r is not None else float('nan')

def analyze_run(run_dir: Path):
    csv_path = run_dir / "profile.csv"
    scene_path = run_dir / "scene.json"
    if not csv_path.exists() or not scene_path.exists():
        return None

    # Load scene to extract params
    with open(scene_path, 'r') as f:
        scene = json.load(f)
    friction = find_friction_heuristic(scene)
    fSigma = find_fSigma_heuristic(scene)

    if np.isnan(friction) or np.isnan(fSigma):
        print(f"Warning: Could not extract params from {scene_path}")
        # Try to infer from directory name as a fallback: *_fSigmaX_frictionY_*
        try:
            name = run_dir.name
            # crude parse
            if 'fSigma' in name and 'friction' in name:
                import re
                m1 = re.search(r"fSigma([0-9\.eE+-]+)", name)
                m2 = re.search(r"friction([0-9\.eE+-]+)", name)
                if m1 and np.isnan(fSigma):
                    fSigma = float(m1.group(1))
                if m2 and np.isnan(friction):
                    friction = float(m2.group(1))
        except Exception:
            pass
        if np.isnan(friction) or np.isnan(fSigma):
            return None

    # Load KE data
    df = pd.read_csv(csv_path)
    if 'frame' not in df.columns or 'KE' not in df.columns:
        return None
    frames = df['frame'].values.astype(float)
    ke = df['KE'].values.astype(float)
    return {
        'frames': frames,
        'ke': ke,
        'friction': friction,
        'fSigma': fSigma,
    }

def exp_decay(t, a, b):
    return a * np.exp(b * t)

def fit_exponential(ke, frames):
    try:
        # Fit last 2/3 of data to capture decay
        n = len(ke)
        start = n // 3
        popt, _ = curve_fit(exp_decay, frames[start:], ke[start:], p0=[ke[start], -0.001])
        return popt[1]  # decay rate b
    except:
        return np.nan

def make_plots(data, exponents, outdir: Path):
    outdir.mkdir(exist_ok=True)

    # Group by friction
    frictions = sorted({d['friction'] for d in data.values()})
    fSigmas = sorted({d['fSigma'] for d in data.values()})

    # Plot linear scale
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = ['blue', 'green', 'red', 'purple']
    for i, friction in enumerate(frictions):
        ax = axes[i]
        ax.set_title(f"Friction = {friction} (Linear)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("KE")
        for j, fSigma in enumerate(fSigmas):
            key = (fSigma, friction)
            if key in data:
                frames, ke = data[key]['frames'], data[key]['ke']
                ax.plot(frames, ke, label=f"fSigma={fSigma}", color=colors[j])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(outdir / "noise_friction_study_linear.png", dpi=150)
    plt.close()

    # Plot log scale
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for i, friction in enumerate(frictions):
        ax = axes[i]
        ax.set_title(f"Friction = {friction} (Log Y)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("KE (log scale)")
        for j, fSigma in enumerate(fSigmas):
            key = (fSigma, friction)
            if key in data:
                frames, ke = data[key]['frames'], data[key]['ke']
                ax.semilogy(frames, ke, label=f"fSigma={fSigma}", color=colors[j])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(outdir / "noise_friction_study_log.png", dpi=150)
    plt.close()

    # Plot exponents vs fSigma for each friction
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, friction in enumerate(frictions):
        fs = []
        exps = []
        for fSigma in fSigmas:
            key = (fSigma, friction)
            if key in exponents and not np.isnan(exponents[key]):
                fs.append(fSigma)
                exps.append(exponents[key])
        if fs:
            ax.plot(fs, exps, 'o-', label=f"Friction={friction}")
    ax.set_xlabel("fSigma")
    ax.set_ylabel("Decay Exponent b")
    ax.set_title("Decay Exponents vs Noise Amplitude")
    ax.legend()
    ax.grid(True)
    plt.savefig(outdir / "noise_decay_exponents.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze parametric noise study results.")
    parser.add_argument('--job-name', type=str, required=True, help='Job name used in submission.')
    parser.add_argument('--runs-root', type=str, default='/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs', help='Root directory for runs.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for plots.')
    parser.add_argument('--make-plots', action='store_true', help='Generate plots.')
    args = parser.parse_args()

    runs_root = Path(args.runs_root) / args.job_name
    if not runs_root.exists():
        raise SystemExit(f"Runs root not found: {runs_root}")

    run_dirs_file = runs_root / "run_dirs.txt"
    if run_dirs_file.exists():
        with open(run_dirs_file, 'r') as f:
            run_dirs = [Path(line.strip()) for line in f if line.strip()]
    else:
        # Scan for run directories
        run_dirs = [d for d in runs_root.iterdir() if d.is_dir() and d.name.startswith('20')]

    data = {}
    exponents = {}

    for run_dir in run_dirs:
        result = analyze_run(run_dir)
        if result is None:
            continue
        key = (result['fSigma'], result['friction'])
        data[key] = result
        exponents[key] = fit_exponential(result['ke'], result['frames'])

    if not data:
        print("No data found.")
        return

    outdir = Path(args.outdir)
    if args.make_plots:
        make_plots(data, exponents, outdir)
        print(f"Plots saved to {outdir}")

if __name__ == "__main__":
    main()