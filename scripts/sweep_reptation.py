#!/usr/bin/env python3
"""sweep_reptation.py — Parameter sweep for frictional reptation.

Launches headless simulations over combinations of (mu, R_cyl, v0, w0)
and collects per-run summary CSVs into a single combined file.

Usage:
    python scripts/sweep_reptation.py [--exe PATH] [--out-dir DIR] [--dry-run]
"""

import argparse
import itertools
import os
import subprocess
import sys
import json

import numpy as np


def make_scene(base: dict, R: float, friction: float,
               use_cuda: bool,
               use_nsc: bool = False, sigma_v: float = None,
               thermal: bool = False) -> dict:
    d = json.loads(json.dumps(base))

    # Apply tube radius
    d.setdefault("scene", {}).setdefault("cylinder", {})["radius"] = R

    # Thermal mode: compute kBT from sigma_v and rod mass
    if thermal and sigma_v is not None:
        import math
        rod_length = 1.0
        rod_diameter = 0.01  # from reptation.json base
        rod_density = 1000.0  # from reptation.json base
        if "scene" in d and "bodies" in d["scene"] and d["scene"]["bodies"]:
            b0 = d["scene"]["bodies"][0]
            rod_length = b0.get("length", rod_length)
            rod_diameter = b0.get("diameter", rod_diameter)
            rod_density = b0.get("density", rod_density)
        
        rod_radius = rod_diameter / 2.0
        rod_mass = rod_density * math.pi * rod_radius**2 * rod_length
        kBT = rod_mass * sigma_v**2
        
        d["scene"]["randomInit"] = {
            "enabled": True,
            "mode": "thermal",
            "kBT": kBT,
            "seed": 42,
            "projectParallelSpin": True,
        }
    else:
        # Disable randomInit if not using thermal (rely on CLI --set-velocity)
        if "randomInit" in d.get("scene", {}):
            d["scene"]["randomInit"]["enabled"] = False 

    # Apply friction
    if use_nsc:
        nsc = d.setdefault("physics", {}).setdefault("nsc", {})
        nsc["enabled"] = True
        nsc["mu"] = friction
    else:
        sc = d.setdefault("physics", {}).setdefault("soft_contact", {})
        sc["enabled"] = True
        sc["mu"] = friction
        sc["mu_static"] = friction
        
    if use_cuda:
        d.setdefault("physics", {}).setdefault("soft_contact", {})["use_cuda"] = True
        
    return d


def main():
    parser = argparse.ArgumentParser(description="Reptation parameter sweep")
    parser.add_argument("--exe", default="./build_head/rigidbody_viewer_3d",
                        help="Path to headless binary")
    parser.add_argument("--scene", default="assets/scenes/reptation.json",
                        help="Base scene JSON")
    parser.add_argument("--out-dir", default="results/reptation",
                        help="Directory for output CSVs")
    parser.add_argument("--steps", type=int, default=2_000_000,
                        help="Max simulation steps")
    parser.add_argument("--stop-ke", type=float, default=1e-10,
                        help="KE threshold for early stop")
    parser.add_argument("--stop-ke-avg-window", type=int, default=5,
                        help="Rolling average window for stop-KE")
    parser.add_argument("--mus", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
                        help="Friction coefficients to sweep")
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[0.2, 0.3, 0.5],
                        help="Cylinder radii to sweep")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of random trials per (mu, R)")
    parser.add_argument("--sigma-v", type=float, default=1.0,
                        help="Std-dev of initial axial velocity (MB) (Used for kBT in thermal init)")
    parser.add_argument("--sigma-w", type=float, default=0.5,
                        help="Std-dev of initial angular velocity components (Ignored in thermal init mode)")
    parser.add_argument("--thermal", action="store_true",
                        help="Use native C++ thermal (Maxwell-Boltzmann) initialization")
    parser.add_argument("--use-cuda", action="store_true",
                        help="Set use_cuda in soft contact scene settings")
                        
    # NSC arguments for alignment
    parser.add_argument("--nsc", action="store_true",
                        help="Use NSC (impulse-based) contact solver instead of soft contact.")
    parser.add_argument("--nsc-iters", type=int, default=40)
    parser.add_argument("--nsc-beta", type=float, default=0.2)
    parser.add_argument("--nsc-cfm", type=float, default=0.0)
    parser.add_argument("--nsc-omega", type=float, default=1.0)
    parser.add_argument("--nsc-pos-iters", type=int, default=5)
    parser.add_argument("--nsc-pos-psor", type=int, default=50)

    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--sbatch", action="store_true",
                        help="Submit as a Slurm Job Array instead of sequential local execution")
    parser.add_argument("--cpus", type=int, default=1,
                        help="Number of CPUs per Slurm task")
    parser.add_argument("--combined-csv", default=None,
                        help="Path for combined summary CSV (default: <out-dir>/combined.csv)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    combined_path = args.combined_csv or os.path.join(args.out_dir, "combined.csv")

    with open(args.scene, "r") as f:
        base_scene = json.load(f)

    total = len(args.mus) * len(args.radii) * args.trials
    run_idx = 0
    array_commands = []

    for mu, R in itertools.product(args.mus, args.radii):
        for trial in range(args.trials):
            run_idx += 1
            rng = np.random.default_rng(seed=trial)
            
            tag = f"mu{mu}_R{R}_t{trial}"
            summary_path = os.path.join(args.out_dir, f"rept_{tag}.csv")
            scene_path = os.path.join(args.out_dir, f"scene_{tag}.json")
            
            scene_data = make_scene(base_scene, R=R, friction=mu, use_cuda=args.use_cuda,
                                    use_nsc=args.nsc, sigma_v=args.sigma_v, thermal=args.thermal)
            
            # Explicit seed override via JSON
            if args.thermal:
                scene_data["scene"]["randomInit"]["seed"] = trial

            with open(scene_path, "w") as f:
                json.dump(scene_data, f, indent=2)

            cmd = [
                args.exe,
                "--headless", "--steps", str(args.steps),
                "--scene", scene_path,
                "--stop-ke-threshold", str(args.stop_ke),
                "--stop-ke-avg-window", str(args.stop_ke_avg_window),
                "--reptation-summary", summary_path,
                "--quiet",
            ]

            if args.nsc:
                cmd.extend([
                    "--nsc",
                    "--nsc-iters", str(args.nsc_iters),
                    "--nsc-beta", str(args.nsc_beta),
                    "--nsc-pos-iters", str(args.nsc_pos_iters),
                    "--nsc-pos-psor", str(args.nsc_pos_psor)
                ])
                if args.nsc_cfm != 0.0:
                    cmd.extend(["--nsc-cfm", str(args.nsc_cfm)])
                if args.nsc_omega != 1.0:
                    cmd.extend(["--nsc-omega", str(args.nsc_omega)])
            
            if not args.thermal:
                v0 = rng.normal(0, args.sigma_v)  # axial velocity (Y-axis)
                w0 = rng.normal(0, args.sigma_w, size=2)  # tumbling (X, Z)
                cmd.extend([
                    "--set-velocity", "0", "0", f"{v0:.6f}", "0",
                    "--set-ang-velocity", "0", f"{w0[0]:.6f}", "0", f"{w0[1]:.6f}"
                ])
                print(f"[{run_idx}/{total}] mu={mu} R={R} trial={trial} v0={v0:.4f}")
            else:
                print(f"[{run_idx}/{total}] mu={mu} R={R} trial={trial} (thermal kBT)")

            if args.dry_run:
                print("  " + " ".join(cmd))
                continue

            if args.sbatch:
                import shlex
                # Ensure the path contains absolute structure or resolve via cd
                array_commands.append(" ".join(shlex.quote(c) for c in cmd))
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  WARNING: non-zero exit code {result.returncode}",
                          file=sys.stderr)
                    if result.stderr:
                        print(f"  stderr: {result.stderr[:200]}", file=sys.stderr)

    if args.sbatch and array_commands:
        cmds_file = os.path.join(args.out_dir, "array_commands.txt")
        with open(cmds_file, "w") as f:
            f.write("\n".join(array_commands) + "\n")
            
        master_sb = f"""#!/bin/bash
#SBATCH -n 1
#SBATCH -c {args.cpus}
#SBATCH -t 0-12:00:00
#SBATCH -p seas_compute
#SBATCH --mem=8G
#SBATCH --array=1-{len(array_commands)}
#SBATCH -o {os.path.abspath(args.out_dir)}/output_%A_%a.out
#SBATCH -e {os.path.abspath(args.out_dir)}/errors_%A_%a.err
#SBATCH --job-name=sweep_rept_array

set -euo pipefail

CMD=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {os.path.abspath(cmds_file)})
echo "Running Array task $SLURM_ARRAY_TASK_ID"
echo "Command: $CMD"

eval "$CMD"
echo "Job complete."
"""
        master_sbatch_path = os.path.join(args.out_dir, "Master_Sbatch.sh")
        with open(master_sbatch_path, "w") as f:
            f.write(master_sb)
            
        print(f"Submitting SLURM Array Job with {len(array_commands)} tasks...")
        subprocess.run(["sbatch", "Master_Sbatch.sh"], cwd=args.out_dir)

    elif not args.dry_run and not args.sbatch:
        # Combine all individual summary CSVs into one (Local execution)
        combine_summaries(args.out_dir, combined_path)
        print(f"\nDone. Combined summary: {combined_path}")


def combine_summaries(out_dir, combined_path):
    """Concatenate all rept_*.csv files into a single CSV."""
    import glob
    files = sorted(glob.glob(os.path.join(out_dir, "rept_*.csv")))
    if not files:
        print("No summary files found to combine.")
        return

    header_written = False
    with open(combined_path, "w") as out:
        for f in files:
            with open(f) as inp:
                lines = inp.readlines()
                for line in lines:
                    if line.startswith("mu,"):
                        if not header_written:
                            out.write(line)
                            header_written = True
                    else:
                        out.write(line)


if __name__ == "__main__":
    main()
