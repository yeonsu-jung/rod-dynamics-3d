#!/usr/bin/env python3
"""
submit_noise_study.py

Submit SLURM jobs for parametric noise study: Effects of noise amplitude (fSigma) and friction on KE evolution.

- Sweeps fSigma and friction, submits headless simulations as separate jobs.
- For each parameter set, creates a run directory under /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/{job_name} with a generated scene.json.
- Copies the built executable (build/rigidbody_viewer_3d) into each run directory.
- Writes an sbatch script that runs the simulation and a lightweight post-analysis.
- Submits the job via sbatch.

Usage: python submit_noise_study.py --job-name noise_test

Then, for analysis: python post_analyze_noise_study.py --job-name noise_test --make-plots --outdir analysis_noise_test

Prereq: Build headless binary first
  module load cmake
  mkdir -p build && cd build && cmake .. -DBUILD_HEADLESS=ON && cmake --build . -j
"""

from __future__ import annotations
from pathlib import Path
<<<<<<< HEAD
from datetime import datetime
import shutil, subprocess, os, stat, sys, json
import argparse

# ------------------- USER CONFIG -------------------

fSigmas = [0.01, 0.1, 1.0, 10.0]
frictions = [0.0, 0.1, 1.0, 3.0]
STEPS = 5000

# SLURM defaults
SLURM = {
    "partition":  "seas_compute",
    "time":       "0-12:00",
    "mem":        "500",
    "ntasks":     1,
    "cpus":       4,
    "nodes":      1,
    "mail_user":  os.environ.get("USER_EMAIL", ""),
    "mail_type":  "END",
    "module_line":"module load python",
    "conda_env":  "simdata-analysis",
}
=======
try:
    from scipy.optimize import curve_fit
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
>>>>>>> refs/remotes/origin/main

BASE_SCENE = {
    "scene": {
        "periodic": {
            "enabled": True,
<<<<<<< HEAD
            "min": [-1.0, -1.0, -1.0],
            "max": [ 1.0,  1.0,  1.0],
=======
            "min": [-0.6, -0.6, -0.6],
            "max": [ 0.6,  0.6,  0.6],
>>>>>>> refs/remotes/origin/main
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

# ------------------- HELPERS -------------------

def find_root_dir(start: Path | None = None, target_name: str = "rod-dynamics-3d") -> Path:
    p = Path.cwd() if start is None else Path(start).resolve()
    for ancestor in [p, *p.parents]:
        if ancestor.name == target_name:
            return ancestor
    raise SystemExit(f"Could not find repository root named '{target_name}' starting from {p}")

<<<<<<< HEAD
def now_ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_executable(path: Path):
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if not os.access(path, os.X_OK):
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
=======
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
>>>>>>> refs/remotes/origin/main

def compute_param_sets():
    ps = []
    for fSigma in fSigmas:
        for friction in frictions:
            ps.append({
                "fSigma": float(fSigma),
                "friction": float(friction),
                "steps": STEPS,
            })
    return ps

def make_run_dir(runs_root: Path, params: dict, job_name: str) -> Path:
    name = (
        f"{now_ts()}_RUN_noise"
        f"_fSigma{params['fSigma']}"
        f"_friction{params['friction']}"
        f"_{job_name}"
    )
    run_dir = runs_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

<<<<<<< HEAD
def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)
=======
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
>>>>>>> refs/remotes/origin/main

def save_json(data, path: Path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def generate_scene(run_dir: Path, params: dict) -> Path:
    scene = BASE_SCENE.copy()
    scene["scene"]["randomForce"]["fSigma"] = params["fSigma"]
    scene["scene"]["randomForce"]["tauMag"] = params["fSigma"]
    scene["scene"]["bodies"][0]["friction"] = params["friction"]
    scene["scene"]["bodies"][0]["friction_s"] = params["friction"]
    scene["scene"]["bodies"][0]["friction_d"] = params["friction"]

    scene_path = run_dir / "scene.json"
    save_json(scene, scene_path)
    return scene_path

def write_readme_and_options(run_dir: Path, params: dict, sim_cmd: str):
    (run_dir / "README.txt").write_text(
        "Parameters:\n" + "\n".join(f"{k}: {v}" for k, v in params.items()) + f"\n\nCMD:\n{sim_cmd}\n"
    )
    with open(run_dir / "options.txt", "w") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

def write_sbatch(run_dir: Path, sim_cmd: str):
    post_py = r"""
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
csv = 'profile.csv'
if os.path.exists(csv):
    df = pd.read_csv(csv)
    if 'frame' in df.columns and 'KE' in df.columns:
        os.makedirs('figs', exist_ok=True)
        plt.figure(figsize=(8,5))
        plt.plot(df['frame'], df['KE'])
        plt.xlabel('Frame'); plt.ylabel('Kinetic Energy (J)'); plt.title('KE vs Frame')
        plt.tight_layout(); plt.savefig('figs/ke.png')
        # Simple exponential fit
        try:
            from scipy.optimize import curve_fit
            t = df['frame'].to_numpy(dtype=float)
            y = df['KE'].to_numpy(dtype=float)
            def model(t,a,b): return a*np.exp(b*t)
            popt,_ = curve_fit(model, t, y, p0=[y[0], -1e-3], maxfev=20000)
            a,b = popt
            with open('analysis.txt','w') as f:
                f.write(f'Exp fit: a={a:.6g}, b={b:.6g}\n')
        except Exception as e:
            with open('analysis.txt','w') as f:
                f.write(f'Exp fit failed: {e}\n')
"""
    sb = f"""#!/bin/bash
#SBATCH -n {SLURM['ntasks']}
#SBATCH -c {SLURM['cpus']}
#SBATCH -N {SLURM['nodes']}
#SBATCH -t {SLURM['time']}
#SBATCH -p {SLURM['partition']}
#SBATCH --mem={SLURM['mem']}
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err
#SBATCH --mail-type={SLURM['mail_type']}
{f"#SBATCH --mail-user={SLURM['mail_user']}" if SLURM['mail_user'] else ""}

set -euo pipefail
{SLURM['module_line']}
{('mamba activate '+SLURM['conda_env']) if SLURM['conda_env'] else "# (no conda activation)"}

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

# --- run simulation ---
{sim_cmd}

# --- post-analysis (inline) ---
echo "Running post-analysis..."
python3 - <<'PY'
{post_py}
PY

echo "Post-analysis complete."
"""
    sbatch_path = run_dir / "Sbatch.sh"
    sbatch_path.write_text(sb)
    os.chmod(sbatch_path, 0o755)
    return sbatch_path

def submit(run_dir: Path):
    print(f"Submitting: {run_dir}")
    subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir, check=True)

# ------------------- MAIN -------------------

def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for parametric noise study.")
    parser.add_argument('--job-name', type=str, default='noname', help='Job name for organizing runs.')
    args = parser.parse_args()

    root_dir = find_root_dir()
    runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs") / args.job_name
    runs_root.mkdir(parents=True, exist_ok=True)

    binary_src = root_dir / "build/rigidbody_viewer_3d"
    ensure_executable(binary_src)

    param_sets = compute_param_sets()

    run_dirs = []
    for params in param_sets:
        run_dir = make_run_dir(runs_root, params, args.job_name)
        run_dirs.append(run_dir)

        # copy binary
        binary_dst = run_dir / "rigidbody_viewer_3d"
        shutil.copy2(binary_src, binary_dst)
        os.chmod(binary_dst, 0o755)

        # copy this file
        shutil.copy2(Path(__file__), run_dir / Path(__file__).name)

        # generate scene.json in run dir
        scene_path = generate_scene(run_dir, params)

        # build simulation command
        sim_cmd = (
            f"./rigidbody_viewer_3d "
            f"--headless "
            f"--scene scene.json "
            f"--steps {int(params['steps'])} "
            f"--csv profile.csv "
            f"--threads ${{SLURM_CPUS_PER_TASK:-{SLURM['cpus']}}}"
        )

        # write docs and sbatch
        write_readme_and_options(run_dir, params, sim_cmd)
        write_sbatch(run_dir, sim_cmd)

        # submit
        submit(run_dir)

    # Save list of run directories
    run_dirs_file = runs_root / "run_dirs.txt"
    with open(run_dirs_file, 'w') as f:
        for rd in run_dirs:
            f.write(str(rd) + '\n')
    print(f"Saved run directories to: {run_dirs_file}")

    # Save combined analysis command
    combined_cmd = f"python3 post_analyze_noise_study.py --job-name {args.job_name} --make-plots --outdir analysis_noise_{args.job_name}"
    combined_cmd_file = runs_root / "combined_analysis_command.txt"
    with open(combined_cmd_file, 'w') as f:
        f.write(combined_cmd + '\n')
    print(f"Saved combined analysis command to: {combined_cmd_file}")
    print(f"Command: {combined_cmd}")

if __name__ == "__main__":
    main()