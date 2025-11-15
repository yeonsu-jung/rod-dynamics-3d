#!/usr/bin/env python3
"""
Parametric study: Effects of noise amplitude (fSigma) and friction on KE evolution.

Sweeps fSigma and friction, runs headless simulations, plots KE vs time.
"""

import os
import subprocess
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Base scene template
BASE_SCENE = {
    "scene": {
        "periodic": {
            "enabled": True,
            "min": [-0.6, -0.6, -0.6],
            "max": [ 0.6,  0.6,  0.6],
            "cellSize": 2.0
        },
        "populate": {
            "count": 100,
            "mode": "nonoverlap",
            "spacingMul": 2.0,
            "seed": 12345,
            "maxAttempts": 50000
        },
        # Initial random velocity kick disabled to study pure noise-driven energization
        "randomInit": {
            "enabled": False,
            "vSigma": 0.0,
            "wSpeed": 0.0,
            "seed": 42
        },
        "randomForce": {
            "enabled": True,
            "fSigma": 0.01,  # will vary
            "tauMag": 0.01,  # set to fSigma
            "seed": 98765
        },
        "bodies": [
            {
                "length": 1,
                "diameter": 0.05,
                "density": 1000.0,
                "restitution": 1.0,
                "friction": 0.0,  # will vary
                "friction_s": 1.0,
                "friction_d": 1.0
            }
        ]
    },
    "physics": {
        "dt": 0.0016667,
        "gravity": [0.0, 0.0, 0.0],
        "lin_damp": 0.0,
        "ang_damp": 0.0,
        "substeps": 1,
        "solver": {
            "velIters": 120,
            "baumgarte": 0.0,
            "allowedPen": 0.003,
            "splitImpulse": True,
            "splitOrient": True,
            "ngsNormalSweeps": 1,
            "ngsHighVThresh": 0.4
        }
    }
}

def generate_scene(fSigma, friction, scenes_dir):
    scene = BASE_SCENE.copy()
    scene["scene"]["randomForce"]["fSigma"] = fSigma
    scene["scene"]["randomForce"]["tauMag"] = fSigma  # match translational
    scene["scene"]["bodies"][0]["friction"] = friction
    scene["scene"]["bodies"][0]["friction_s"] = friction
    scene["scene"]["bodies"][0]["friction_d"] = friction

    filename = f"noise_f{fSigma}_fric{friction}.json"
    path = scenes_dir / filename
    with open(path, 'w') as f:
        json.dump(scene, f, indent=2)
    return path

def run_simulation(scene_path, csv_path, exe_path, steps=1000):
    # Prefer per-rod CSV since it's supported broadly in our runs; aggregate later.
    cmd = [
        str(exe_path),
        "--scene", str(scene_path),
        "--headless",
        "--steps", str(steps),
        "--perrod", str(csv_path),
        "--perrod-max", str(steps)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=exe_path.parent)
    if result.returncode != 0:
        print(f"Error running {scene_path}: {result.stderr}")
        return False
    return True

def exp_decay(t, a, b):
    return a * np.exp(b * t)

def fit_exponential(ke, frames):
    try:
        # Fit last 2/3 of data to capture decay
        n = len(ke)
        start = n // 3
        if _HAS_SCIPY:
            popt, _ = curve_fit(exp_decay, frames[start:], ke[start:], p0=[max(ke[start], 1e-8), -0.001])
            return popt[1]  # decay rate b
        # Fallback: simple log-linear fit using numpy (approximate)
        y = np.array(ke[start:], dtype=float)
        y[y <= 1e-12] = 1e-12
        x = np.array(frames[start:], dtype=float)
        b, a = np.polyfit(x, np.log(y), 1)  # log(y) ~ a + b x
        return b
    except:
        return np.nan

def main():
    # Paths
    script_dir = Path(__file__).parent
    scenes_dir = script_dir / "scenes"
    csvs_dir = script_dir / "csvs"
    exe_path = script_dir.parent / "build" / "rigidbody_viewer_3d"

    scenes_dir.mkdir(exist_ok=True)
    csvs_dir.mkdir(exist_ok=True)

    # Parameter sweeps
    fSigmas = [0.01, 0.1, 1.0, 10.0]
    frictions = [0.0, 0.1, 1.0, 3.0]
    steps = 5000

    # Collect data
    data = {}
    exponents = {}

    for fSigma in fSigmas:
        for friction in frictions:
            key = (fSigma, friction)
            scene_path = generate_scene(fSigma, friction, scenes_dir)
            csv_path = csvs_dir / f"noise_f{fSigma}_fric{friction}.csv"

            if csv_path.exists():
                print(f"Skipping {key}, CSV exists")
            else:
                print(f"Running {key}")
                if not run_simulation(scene_path, csv_path, exe_path, steps):
                    continue

            # Load KE data: support either global KE or per-rod KE_total aggregated
            df = pd.read_csv(csv_path)
            if {"frame","KE"}.issubset(df.columns):
                frames = df["frame"].values
                ke = df["KE"].values
            elif {"frame","rod","KE_total"}.issubset(df.columns):
                agg = df.groupby('frame')["KE_total"].sum().reset_index()
                frames = agg["frame"].values
                ke = agg["KE_total"].values
            else:
                print(f"Unrecognized CSV schema for {csv_path}, columns={df.columns.tolist()}")
                continue
            data[key] = (frames, ke)
            exponents[key] = fit_exponential(ke, frames)

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
                frames, ke = data[key]
                ax.plot(frames, ke, label=f"fSigma={fSigma}", color=colors[j])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(script_dir / "noise_friction_study_linear.png", dpi=150)
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
                frames, ke = data[key]
                ax.semilogy(frames, ke, label=f"fSigma={fSigma}", color=colors[j])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(script_dir / "noise_friction_study_log.png", dpi=150)
    plt.close()

    # Plot exponents vs fSigma for each friction
    try:
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
        ax.set_ylabel("Decay Exponent b" + (" (curve_fit)" if _HAS_SCIPY else " (log-linear approx)"))
        ax.set_title("Decay Exponents vs Noise Amplitude")
        ax.legend()
        ax.grid(True)
        plt.savefig(script_dir / "noise_decay_exponents.png", dpi=150)
        plt.close()
    except Exception as e:
        print("Exponent plotting skipped:", e)

    print("Study complete. Plots saved: linear, log, exponents.")

if __name__ == "__main__":
    main()