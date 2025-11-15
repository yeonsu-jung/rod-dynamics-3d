#!/usr/bin/env python3
"""
submit_parametric_runs.py

Submit SLURM jobs to run headless rod dynamics simulations for various aspect ratios.

- Discovers repo root by walking up until a folder named 'rod-dynamics-3d' is found.
- For each parameter set, creates a run directory under /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/{job_name} with a generated scene.json.
- Copies the built executable (build/rigidbody_viewer_3d) into each run directory.
- Writes an sbatch script that runs the simulation and a lightweight post-analysis.
- Submits the job via sbatch.

Usage: python submit_parametric_runs.py --job-name TEST
       python submit_parametric_runs.py --sweep friction --alpha 100 --frictions "0.05,0.1,0.2,0.4,1" --job-name friction2

       python post_analyze_friction_effects.py --job-name friction2 --alpha 100 --make-plots --outdir analysis_friction2

Prereq: Build headless binary first
  module load cmake
  mkdir -p build && cd build && cmake .. -DBUILD_HEADLESS=ON && cmake --build . -j
"""



from __future__ import annotations
from pathlib import Path
from datetime import datetime
import shutil, subprocess, os, stat, sys, json
import argparse

# ------------------- USER CONFIG -------------------

alpha_values = [25, 50, 100, 200, 500]
cpus_values = [4, 4, 12, 24, 48]  # corresponding to alpha_values
rod_length = 1.0
C = 1.5
# count formula matches prior study: N = (2*C)^3 / (d * L^2)
# Allow an extra multiplicative factor per template
factor = 5.
STEPS = 2000
# Friction sweep defaults
friction_values = [0.05, 0.1, 0.2, 0.4]

# SLURM defaults (override by editing here)
SLURM = {
    "partition":  "seas_compute",
    "time":       "0-12:00",
    "mem":        "500",
    "ntasks":     1,
    "cpus":        4,  # will be overridden per run
    "nodes":      1,
    "mail_user":  os.environ.get("USER_EMAIL", ""),  # set if desired
    "mail_type":  "END",
    "module_line":"module load python",
    "conda_env":  "simdata-analysis",   # e.g., "simdata-analysis" or ""
}

BASE_SCENE_REL = "assets/scenes/dissipation_study_sample.json"

# ------------------- HELPERS -------------------

def find_root_dir(start: Path | None = None, target_name: str = "rod-dynamics-3d") -> Path:
    p = Path.cwd() if start is None else Path(start).resolve()
    for ancestor in [p, *p.parents]:
        if ancestor.name == target_name:
            return ancestor
    raise SystemExit(f"Could not find repository root named '{target_name}' starting from {p}")

def now_ts():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def ensure_executable(path: Path):
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if not os.access(path, os.X_OK):
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


def compute_param_sets(mode: str, alpha_single: int | float | None, frictions: list[float] | None):
    ps = []
    if mode == 'friction':
        a = int(alpha_single) if alpha_single is not None else 100
        rod_diameter = rod_length / a
        N_base = (2 * C) ** 3 / (rod_diameter * rod_length ** 2)
        N = int(N_base * factor)
        for mu in (frictions or friction_values):
            ps.append({
                "alpha": a,
                "seed": 111,
                "N": N,
                "C": C,
                "rod_length": rod_length,
                "rod_diameter": rod_diameter,
                "factor": factor,
                "steps": STEPS,
                "mu": float(mu),
            })
        return ps
    # default: AR sweep
    for alpha in alpha_values:
        rod_diameter = rod_length / alpha
        N_base = (2 * C) ** 3 / (rod_diameter * rod_length ** 2)
        N = int(N_base * factor)
        ps.append({
            "alpha": alpha,
            "seed": 111,
            "N": N,
            "C": C,
            "rod_length": rod_length,
            "rod_diameter": rod_diameter,
            "factor": factor,
            "steps": STEPS,
        })
    return ps


def make_run_dir(runs_root: Path, params: dict, job_name: str) -> Path:
    name = (
        f"{now_ts()}_RUN_rods"
        f"_AR{params['alpha']}"
        f"_SEED{params['seed']}"
        f"_N{params['N']}"
        f"_C{params['C']}"
        f"_L{params['rod_length']}"
    )
    if 'mu' in params:
        name += f"_MU{params['mu']}"
    name += f"_{job_name}"
    run_dir = runs_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path: Path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def generate_scene(base_scene_path: Path, run_dir: Path, params: dict) -> Path:
    base = load_json(base_scene_path)
    # Be robust to structure: set populate.count and first body's diameter if present
    # Scene populate
    scene = base.get('scene', {})
    populate = scene.get('populate', {})
    populate['count'] = int(params['N'])
    scene['populate'] = populate
    # Ensure bodies list exists
    bodies = scene.get('bodies', base.get('scene', {}).get('bodies', []))
    if not isinstance(bodies, list) or len(bodies) == 0:
        bodies = [{}]
    # diameter
    bodies[0]['diameter'] = float(params['rod_diameter'])
    # friction if provided
    if 'mu' in params:
        mu = float(params['mu'])
        bodies[0]['friction'] = mu
        # set static/dynamic if present in template, otherwise also set both
        bodies[0]['friction_s'] = mu if 'friction_s' in bodies[0] or True else bodies[0].get('friction_s', mu)
        bodies[0]['friction_d'] = mu if 'friction_d' in bodies[0] or True else bodies[0].get('friction_d', mu)
    scene['bodies'] = bodies
    base['scene'] = scene

    scene_path = run_dir / "scene.json"
    save_json(base, scene_path)
    return scene_path


def write_readme_and_options(run_dir: Path, params: dict, sim_cmd: str):
    (run_dir / "README.txt").write_text(
        "Parameters:\n" + "\n".join(f"{k}: {v}" for k, v in params.items()) + f"\n\nCMD:\n{sim_cmd}\n"
    )
    with open(run_dir / "options.txt", "w") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")


def write_sbatch(run_dir: Path, sim_cmd: str):
    # Quick inline post-analysis: save a KE plot if matplotlib available
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

echo "PWD: $(pwd)"
echo "CMD: {sim_cmd}"

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
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for parametric rod dynamics simulations.")
    parser.add_argument('--job-name', type=str, default='noname', help='Job name for organizing runs.')
    parser.add_argument('--sweep', choices=['ar','friction'], default='ar', help='Sweep aspect-ratio (ar) or friction coefficients (friction).')
    parser.add_argument('--alpha', type=int, default=100, help='Alpha to use for friction sweep.')
    parser.add_argument('--frictions', type=str, default='', help='Comma-separated list of friction coefficients for friction sweep.')
    args = parser.parse_args()

    root_dir = find_root_dir()
    runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs") / args.job_name
    runs_root.mkdir(parents=True, exist_ok=True)

    binary_src = root_dir / "build/rigidbody_viewer_3d"
    ensure_executable(binary_src)

    base_scene_path = root_dir / BASE_SCENE_REL
    if not base_scene_path.exists():
        raise SystemExit(f"Base scene not found: {base_scene_path}")

    frics = None
    if args.sweep == 'friction':
        if args.frictions.strip():
            try:
                frics = [float(x) for x in args.frictions.split(',')]
            except Exception:
                raise SystemExit("Invalid --frictions value. Use comma-separated floats, e.g., 0.05,0.1,0.2,0.4")

    param_sets = compute_param_sets(args.sweep, args.alpha, frics)

    run_dirs = []
    for params in param_sets:
        # Set CPUs based on alpha
        try:
            idx = alpha_values.index(params['alpha'])
            SLURM["cpus"] = cpus_values[idx]
        except ValueError:
            # default to middle value if alpha not in alpha_values
            SLURM["cpus"] = 4

        run_dir = make_run_dir(runs_root, params, args.job_name)
        run_dirs.append(run_dir)

        # copy binary
        binary_dst = run_dir / "rigidbody_viewer_3d"
        shutil.copy2(binary_src, binary_dst)
        os.chmod(binary_dst, 0o755)

        # copy this file
        shutil.copy2(Path(__file__), run_dir / Path(__file__).name)

        # generate scene.json in run dir
        scene_path = generate_scene(base_scene_path, run_dir, params)

        # build simulation command
        cutoff = 2.0 * float(params['rod_diameter'])  # conservative local cutoff
        ent_period = 100
        sim_cmd = (
            f"./rigidbody_viewer_3d "
            f"--headless "
            f"--scene {scene_path.name} "
            f"--steps {int(params['steps'])} "
            f"--csv profile.csv "
            f"--threads ${{SLURM_CPUS_PER_TASK:-{SLURM['cpus']}}} "
            f"--entanglement --entanglement-cutoff {cutoff:.6g} "
            f"--entanglement-period {ent_period} "
            f"--entanglement-threads ${{SLURM_CPUS_PER_TASK:-{SLURM['cpus']}}}"
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
    if args.sweep == 'friction':
        combined_cmd = f"python3 post_analyze_friction_effects.py --job-name {args.job_name} --alpha {args.alpha} --make-plots --outdir analysis_friction_{args.job_name}"
    else:
        combined_cmd = f"python3 post_analyze_parametric_runs.py --job-name {args.job_name} --make-plots --outdir analysis_{args.job_name}"
    combined_cmd_file = runs_root / "combined_analysis_command.txt"
    with open(combined_cmd_file, 'w') as f:
        f.write(combined_cmd + '\n')
    print(f"Saved combined analysis command to: {combined_cmd_file}")
    print(f"Command: {combined_cmd}")

if __name__ == "__main__":
    main()
