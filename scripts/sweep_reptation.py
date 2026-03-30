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

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Reptation parameter sweep")
    parser.add_argument("--exe", default="./build-headless/rigidbody_viewer_3d",
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
                        help="Std-dev of initial axial velocity (MB)")
    parser.add_argument("--sigma-w", type=float, default=0.5,
                        help="Std-dev of initial angular velocity components")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--combined-csv", default=None,
                        help="Path for combined summary CSV (default: <out-dir>/combined.csv)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    combined_path = args.combined_csv or os.path.join(args.out_dir, "combined.csv")

    total = len(args.mus) * len(args.radii) * args.trials
    run_idx = 0

    for mu, R in itertools.product(args.mus, args.radii):
        for trial in range(args.trials):
            run_idx += 1
            rng = np.random.default_rng(seed=trial)
            v0 = rng.normal(0, args.sigma_v)  # axial velocity (Y-axis)
            w0 = rng.normal(0, args.sigma_w, size=2)  # tumbling (X, Z)

            tag = f"mu{mu}_R{R}_t{trial}"
            summary_path = os.path.join(args.out_dir, f"rept_{tag}.csv")

            cmd = [
                args.exe,
                "--headless", "--steps", str(args.steps),
                "--scene", args.scene,
                "--nsc",
                "--nsc-mu", str(mu),
                "--set-velocity", "0", "0", str(v0), "0",
                "--set-ang-velocity", "0", str(w0[0]), "0", str(w0[1]),
                "--stop-ke-threshold", str(args.stop_ke),
                "--stop-ke-avg-window", str(args.stop_ke_avg_window),
                "--reptation-summary", summary_path,
                "--quiet",
            ]

            print(f"[{run_idx}/{total}] mu={mu} R={R} trial={trial} v0={v0:.4f}")
            if args.dry_run:
                print("  " + " ".join(cmd))
                continue

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  WARNING: non-zero exit code {result.returncode}",
                      file=sys.stderr)
                if result.stderr:
                    print(f"  stderr: {result.stderr[:200]}", file=sys.stderr)

    if not args.dry_run:
        # Combine all individual summary CSVs into one
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
