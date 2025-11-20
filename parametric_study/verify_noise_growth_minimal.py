#!/usr/bin/env python3
"""
verify_noise_growth_minimal.py

Goal: Empirically verify that with zero friction/damping and nonzero stochastic forcing,
kinetic energy (KE) grows over time in the rod-dynamics-3d simulator.

This script:
- Generates a minimal scene (periodic box, single body) with friction=0, restitution=1,
  lin_damp=ang_damp=0, and randomForce enabled with configurable fSigma.
- Runs the headless simulator for given steps and writes profile.csv.
- Loads KE vs frame, converts to time (t = frame * dt), fits a line to KE(t) and reports slope.
- Saves a KE plot with the linear fit overlay.

Usage:
  python3 verify_noise_growth_minimal.py --fSigma 0.5 --steps 50000 --outdir analysis_minimal_noise

Notes:
- The exact theoretical slope depends on how the engine applies noise (force vs impulse, mass, etc.).
  This script focuses on demonstrating monotonic growth and approximate linear trend.
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, subprocess, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_root_dir(start: Path | None = None, target_name: str = "rod-dynamics-3d") -> Path:
    p = Path.cwd() if start is None else Path(start).resolve()
    for ancestor in [p, *p.parents]:
        if ancestor.name == target_name:
            return ancestor
    raise SystemExit(f"Could not find repository root named '{target_name}' starting from {p}")


def build_minimal_scene(fSigma: float, friction: float, count: int, box_half: float, dt: float) -> dict:
    return {
        "scene": {
            "periodic": {
                "enabled": True,
                "min": [-box_half, -box_half, -box_half],
                "max": [ box_half,  box_half,  box_half],
                "cellSize": 2.0*box_half
            },
            "populate": {
                "count": int(count),
                "mode": "random",
                "spacingMul": 3.0,
                "seed": 1234,
                "maxAttempts": 10000
            },
            "randomInit": {
                "enabled": True,
                "vSigma": 0.0,  # start at rest
                "wSpeed": 0.0,
                "seed": 42
            },
            "randomForce": {
                "enabled": True,
                "fSigma": float(fSigma),
                "tauMag": float(fSigma),
                "seed": 98765
            },
            "bodies": [
                {
                    "length": 1.0,
                    "diameter": 0.02,
                    "density": 1000.0,
                    "restitution": 1.0,
                    "friction": float(friction),
                    "friction_s": float(friction),
                    "friction_d": float(friction)
                }
            ]
        },
        "physics": {
            "dt": float(dt),
            "gravity": [0.0, 0.0, 0.0],
            "lin_damp": 0.0,
            "ang_damp": 0.0,
            "substeps": 1,
            "solver": {
                "velIters": 60,
                "baumgarte": 0.0,
                "allowedPen": 0.001,
                "splitImpulse": False,
                "splitOrient": False,
                "ngsNormalSweeps": 1,
                "ngsHighVThresh": 1e9  # avoid high-velocity clamping if any
            }
        }
    }


def run_sim(scene: dict, steps: int, outdir: Path, exe_path: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    scene_path = (outdir / "scene.json").resolve()
    with open(scene_path, 'w') as f:
        json.dump(scene, f, indent=2)

    csv_path = (outdir / "profile.csv").resolve()
    cmd = [
        str(exe_path),
        "--scene", str(scene_path),
        "--headless",
        "--steps", str(int(steps)),
        "--csv", str(csv_path)
    ]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=exe_path.parent)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise SystemExit(f"Simulation failed with code {res.returncode}")
    return csv_path


def analyze(csv_path: Path, dt: float, plot_path: Path):
    df = pd.read_csv(csv_path)
    if 'frame' not in df.columns:
        raise SystemExit("profile.csv missing required column 'frame'")

    # Prefer 'KE', but some builds may only populate other KE_* snapshots; pick the first non-all-zero.
    ke_candidates = [
        'KE',
        'KE_after_pbcWrap',
        'KE_after_posCorrect',
        'KE_after_solve',
        'KE_after_warmstart',
        'KE_after_integrate'
    ]
    ke_series = None
    chosen = None
    for col in ke_candidates:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
            if np.any(np.isfinite(arr)) and np.any(arr != 0.0):
                ke_series = arr
                chosen = col
                break
    if ke_series is None:
        # Fall back to zeros if literally no KE data present to avoid crash
        if 'KE' in df.columns:
            ke_series = pd.to_numeric(df['KE'], errors='coerce').to_numpy(dtype=float)
            chosen = 'KE'
        else:
            raise SystemExit("profile.csv does not contain any usable KE columns")

    t = df['frame'].to_numpy(dtype=float) * dt
    ke = ke_series

    # Skip initial segment where KE is exactly zero (engine may delay KE computation)
    nz = np.flatnonzero(ke > 0)
    start_idx = int(nz[0]) if nz.size else 0

    # Linear fit KE ~ a + b t using final 2/3 of the nonzero portion
    n = len(ke)
    i0 = max(start_idx, (start_idx + n) // 3)
    if i0 >= n - 2:
        i0 = max(0, n - 2)
    A = np.vstack([np.ones(n - i0), t[i0:]]).T
    y = ke[i0:]
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeff  # slope b is growth rate [J/s]

    plt.figure(figsize=(8,5))
    plt.plot(t, ke, label=f'{chosen} (sim)')
    plt.plot(t[i0:], a + b * t[i0:], 'r--', label=f'Linear fit (slope={b:.3e} J/s)')
    plt.xlabel('Time [s]')
    plt.ylabel('KE [J]')
    plt.title('KE growth with noise (no friction/damping)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return float(b)


def main():
    ap = argparse.ArgumentParser(description='Verify KE growth with noise and zero friction/damping')
    ap.add_argument('--fSigma', type=float, default=0.5, help='Noise amplitude for randomForce')
    ap.add_argument('--friction', type=float, default=0.0, help='Friction coefficient')
    ap.add_argument('--count', type=int, default=1, help='Number of rods (use 1 to avoid contacts)')
    ap.add_argument('--box-half', type=float, default=3.0, help='Half-size of periodic box')
    ap.add_argument('--steps', type=int, default=30000, help='Simulation steps')
    ap.add_argument('--dt', type=float, default=0.0016667, help='Timestep')
    ap.add_argument('--outdir', type=str, default='analysis_minimal_noise', help='Output directory')
    args = ap.parse_args()

    root = find_root_dir()
    exe_path = root / 'build' / 'rigidbody_viewer_3d'
    if not exe_path.exists():
        raise SystemExit(f"Binary not found: {exe_path}, build with -DBUILD_HEADLESS=ON")

    outdir = Path(args.outdir)
    scene = build_minimal_scene(args.fSigma, args.friction, args.count, args.box_half, args.dt)
    csv_path = run_sim(scene, args.steps, outdir, exe_path)

    slope = analyze(csv_path, args.dt, outdir / 'ke_growth.png')
    with open(outdir / 'summary.txt', 'w') as f:
        f.write(f"fSigma={args.fSigma}\nfriction={args.friction}\nsteps={args.steps}\n")
        f.write(f"Estimated KE slope (dE/dt) = {slope:.6e} J/s\n")
    print(f"Done. Plot: {outdir/'ke_growth.png'}  Slope: {slope:.3e} J/s")


if __name__ == '__main__':
    main()
