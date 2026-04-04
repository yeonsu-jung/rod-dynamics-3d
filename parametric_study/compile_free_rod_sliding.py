#!/usr/bin/env python3
"""compile_free_rod_sliding.py

Scan free-rod run directories and compute final sliding length per trajectory.

Sliding length = cumulative arc-length of the free rod's centre-of-mass:
    L = sum_i |CoM(i) - CoM(i-1)|
where CoM = midpoint of the two rod endpoints.

Output CSV columns:
    N, AR, seed, metric, rod, mu, sliding_length, metric_value,
    total_frames, final_time, final_displacement

Usage:
    python parametric_study/compile_free_rod_sliding.py \
        --runs-dir /path/to/runs/free_rod_nsc_gaussian \
        --extreme-rods-csv assets/extreme_rods.csv \
        --output sliding_length_summary.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Pattern: <timestamp>_N{N}_{seed_id}_AR{ar}_{Metric}_rod{rod}
_DIR_RE = re.compile(
    r"^\d{8}-\d{6}_N(\d+)_(.+?)_AR(\d+)_(Min|Max)(FSA|FTA)_rod(\d+)$"
)

# Pattern: free_rod_endpoints_mu{tag}.csv  where tag encodes mu value
_MU_RE = re.compile(r"^free_rod_endpoints_mu(.+)\.csv$")

# Pattern: endpoints_N{N}_AR{AR}_{seed}_{Metric}_rod{rod}_mu{tag}.csv  (bundle layout)
_BUNDLE_RE = re.compile(
    r"^endpoints_N(\d+)_AR(\d+)_(.+?)_(Min|Max)(FSA|FTA)_rod(\d+)_mu(.+)\.csv$"
)


def parse_mu_tag(tag: str) -> float:
    """Convert filename mu tag back to float: 0p05 -> 0.05, 1p0 -> 1.0, m0p5 -> -0.5"""
    s = tag.replace("m", "-").replace("p", ".")
    return float(s)


def compute_sliding_length(df: pd.DataFrame) -> dict:
    """From endpoint CSV (frame,time,rod,x0,y0,z0,x1,y1,z1), compute sliding metrics."""
    if df.empty or len(df) < 2:
        return None

    # Centre of mass = midpoint of endpoints
    px = (df["x0"].values + df["x1"].values) / 2.0
    py = (df["y0"].values + df["y1"].values) / 2.0
    pz = (df["z0"].values + df["z1"].values) / 2.0

    # Cumulative arc-length
    dx = np.diff(px)
    dy = np.diff(py)
    dz = np.diff(pz)
    steps = np.sqrt(dx**2 + dy**2 + dz**2)
    sliding_length = float(np.sum(steps))

    # Net displacement (start to end)
    disp = np.sqrt(
        (px[-1] - px[0])**2 + (py[-1] - py[0])**2 + (pz[-1] - pz[0])**2
    )

    return {
        "sliding_length": sliding_length,
        "final_displacement": float(disp),
        "total_frames": len(df),
        "final_time": float(df["time"].iloc[-1]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compile sliding length from free-rod endpoint trajectories."
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=None,
        help="Path to the free-rod runs directory (e.g. runs/free_rod_nsc_gaussian)."
    )
    parser.add_argument(
        "--bundle-dir", type=Path, nargs="+", default=None,
        help="Path(s) to bundle output directories containing endpoints_*.csv files."
    )
    parser.add_argument(
        "--extreme-rods-csv", type=Path, default=None,
        help="Optional extreme_rods.csv to join metric values."
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output CSV path."
    )
    args = parser.parse_args()

    if args.runs_dir is None and args.bundle_dir is None:
        raise SystemExit("Must specify --runs-dir and/or --bundle-dir.")
    if args.runs_dir is not None and not args.runs_dir.is_dir():
        raise SystemExit(f"Runs directory not found: {args.runs_dir}")

    # Load extreme rods lookup for metric values
    metric_value_lookup = {}
    if args.extreme_rods_csv is not None and args.extreme_rods_csv.exists():
        edf = pd.read_csv(args.extreme_rods_csv)
        for _, row in edf.iterrows():
            key = (int(row["N"]), int(row["AR"]), str(row["ID"]),
                   str(row["Metric"]), int(row["RodIndex"]))
            metric_value_lookup[key] = float(row["Value"])
        print(f"Loaded {len(metric_value_lookup)} metric values from {args.extreme_rods_csv}")

    rows = []
    processed = 0
    skipped = 0
    errors = 0

    # ---- Bundle directory scanning ----
    if args.bundle_dir:
        for bdir in args.bundle_dir:
            if not bdir.is_dir():
                print(f"WARNING: bundle dir not found: {bdir}")
                continue
            csv_files = sorted(bdir.glob("endpoints_*.csv"))
            print(f"Scanning bundle dir {bdir.name}: {len(csv_files)} CSV files")
            for csv_path in csv_files:
                bm = _BUNDLE_RE.match(csv_path.name)
                if not bm:
                    skipped += 1
                    continue
                N_str, AR_str, seed_id, minmax, fsa_fta, rod_str, mu_tag = bm.groups()
                N, AR, rod = int(N_str), int(AR_str), int(rod_str)
                metric = minmax + fsa_fta
                try:
                    mu = parse_mu_tag(mu_tag)
                except ValueError:
                    errors += 1
                    continue
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    errors += 1
                    continue
                result = compute_sliding_length(df)
                if result is None:
                    errors += 1
                    continue
                key = (N, AR, seed_id, metric, rod)
                metric_value = metric_value_lookup.get(key, np.nan)
                rows.append({
                    "N": N, "AR": AR, "seed": seed_id,
                    "metric": metric, "rod": rod, "mu": mu,
                    "sliding_length": result["sliding_length"],
                    "final_displacement": result["final_displacement"],
                    "total_frames": result["total_frames"],
                    "final_time": result["final_time"],
                    "metric_value": metric_value,
                })
                processed += 1
                if processed % 500 == 0:
                    print(f"  ... processed {processed} trajectories")

    # ---- Per-run-directory scanning ----
    if args.runs_dir is None:
        run_dirs = []
    else:
        run_dirs = sorted(d for d in args.runs_dir.iterdir() if d.is_dir())
    if run_dirs:
        print(f"Scanning {len(run_dirs)} run directories...")

    # When multiple timestamps exist for the same (N,seed,AR,metric,rod),
    # keep only the latest (sorted order = chronological for YYYYMMDD-HHMMSS).
    unique_runs = {}
    for run_dir in run_dirs:
        m = _DIR_RE.match(run_dir.name)
        if not m:
            skipped += 1
            continue
        N_str, seed_id, AR_str, minmax, fsa_fta, rod_str = m.groups()
        key = (N_str, seed_id, AR_str, minmax + fsa_fta, rod_str)
        unique_runs[key] = run_dir  # last (= latest timestamp) wins

    print(f"  {len(unique_runs)} unique runs after dedup ({len(run_dirs) - len(unique_runs) - skipped} duplicates)")

    for (N_str, seed_id, AR_str, metric, rod_str), run_dir in sorted(unique_runs.items()):
        N = int(N_str)
        AR = int(AR_str)
        rod = int(rod_str)

        # Find all endpoint CSVs in this dir
        for csv_path in sorted(run_dir.glob("free_rod_endpoints_mu*.csv")):
            mu_match = _MU_RE.match(csv_path.name)
            if not mu_match:
                continue

            try:
                mu = parse_mu_tag(mu_match.group(1))
            except ValueError:
                errors += 1
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception:
                errors += 1
                continue

            result = compute_sliding_length(df)
            if result is None:
                errors += 1
                continue

            # Look up the metric value
            key = (N, AR, seed_id, metric, rod)
            metric_value = metric_value_lookup.get(key, np.nan)

            rows.append({
                "N": N,
                "AR": AR,
                "seed": seed_id,
                "metric": metric,
                "rod": rod,
                "mu": mu,
                "sliding_length": result["sliding_length"],
                "final_displacement": result["final_displacement"],
                "total_frames": result["total_frames"],
                "final_time": result["final_time"],
                "metric_value": metric_value,
            })
            processed += 1

        if processed % 500 == 0 and processed > 0:
            print(f"  ... processed {processed} trajectories")

    if not rows:
        raise SystemExit("No valid trajectories found.")

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["N", "AR", "seed", "metric", "mu"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"\nDone: {processed} trajectories compiled, {skipped} dirs skipped, {errors} errors")
    print(f"Output: {args.output}")
    print(f"Shape: {out_df.shape}")
    print(f"\nSample:\n{out_df.head(10).to_string()}")


if __name__ == "__main__":
    main()
