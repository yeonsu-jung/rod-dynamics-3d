#!/usr/bin/env python3
"""Build analysis CSV for cohort 5 (free_rod_sweep4_bundled runs).

Endpoint files live directly in each bundle directory:
  <RUNS_DIR>/<bundle_dir>/endpoints_N{N}_AR{AR}_{id1}_{id2}_{id3}_{Metric}_rod{rod}_mu{mu}.csv

Computes final_sl as cumulative path length of the rod midpoint.
Looks up Value from extreme_rods_summary.csv.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RUNS_DIR = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/free_rod_sweep4_bundled")
DEFAULT_EXTREME_CSV = Path("/n/home01/yjung/Github/rod-dynamics-3d/extreme_rods_summary.csv")
DEFAULT_OUTPUT_DIR = Path("/n/home01/yjung/Github/rod-dynamics-3d/analysis_outputs_cohort5")
DEFAULT_COHORT_TAG = "sweep5"

# endpoints_N10_AR1000_278_868_121_MaxFSA_rod1_mu0p0.csv
FILE_RE = re.compile(
    r"^endpoints_N(?P<N>\d+)_AR(?P<AR>\d+)_(?P<id1>\d+)_(?P<id2>\d+)_(?P<id3>\d+)_"
    r"(?P<Metric>MinFSA|MaxFSA|MinFTA|MaxFTA)_rod(?P<rod>\d+)_mu(?P<mu_tag>[0-9pm]+)\.csv$"
)


def mu_tag_to_float(tag: str) -> float:
    """Convert mu tag like '0p1' or '1p0' to float."""
    return float(tag.replace("p", "."))


def compute_final_sl(path: Path) -> float:
    """Cumulative midpoint path length from an endpoints CSV."""
    try:
        df = pd.read_csv(path)
        if df.empty or len(df) < 2:
            return np.nan
        mx = (df["x0"].values + df["x1"].values) / 2.0
        my = (df["y0"].values + df["y1"].values) / 2.0
        mz = (df["z0"].values + df["z1"].values) / 2.0
        dx = np.diff(mx)
        dy = np.diff(my)
        dz = np.diff(mz)
        return float(np.sum(np.sqrt(dx**2 + dy**2 + dz**2)))
    except Exception as e:
        print(f"  WARNING: could not process {path.name}: {e}")
        return np.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final_sliding_vs_extreme_value CSV from bundled endpoint runs")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--extreme-csv", type=Path, default=DEFAULT_EXTREME_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cohort-tag", type=str, default=DEFAULT_COHORT_TAG)
    parser.add_argument(
        "--bundle-prefix",
        type=str,
        default="",
        help="Optional prefix to select a specific submission batch (e.g., 20260326-034)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runs_dir = args.runs_dir
    extreme_csv = args.extreme_csv
    output_dir = args.output_dir
    output_csv = output_dir / f"final_sliding_vs_extreme_value_{args.cohort_tag}.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load extreme_rods_summary to look up Values
    # Columns: N, AR, ID, Metric, RodIndex, Value, FilePath
    print(f"Loading extreme CSV: {extreme_csv}")
    extreme_df = pd.read_csv(extreme_csv)
    print(f"  Columns: {extreme_df.columns.tolist()}")
    print(f"  Shape:   {extreme_df.shape}")

    # Build a lookup dict: (N, AR, ID, Metric, RodIndex) -> Value
    extreme_lookup: dict[tuple, float] = {}
    for _, row in extreme_df.iterrows():
        key = (int(row["N"]), int(row["AR"]), str(row["ID"]), str(row["Metric"]), int(row["RodIndex"]))
        extreme_lookup[key] = float(row["Value"])
    print(f"  Lookup entries: {len(extreme_lookup)}")

    # Collect all bundle directories (skip non-directories)
    bundle_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if args.bundle_prefix:
        bundle_dirs = [p for p in bundle_dirs if p.name.startswith(args.bundle_prefix)]
    print(f"\nFound {len(bundle_dirs)} bundle directories")

    rows = []
    missing_extreme = 0
    n_files = 0

    for bdir in sorted(bundle_dirs):
        ep_files = sorted(bdir.glob("endpoints_*.csv"))
        for ep_path in ep_files:
            m = FILE_RE.match(ep_path.name)
            if not m:
                print(f"  SKIP (no match): {ep_path.name}")
                continue

            N = int(m.group("N"))
            AR = int(m.group("AR"))
            id_str = f"{m.group('id1')}_{m.group('id2')}_{m.group('id3')}"
            metric = m.group("Metric")
            rod = int(m.group("rod"))
            mu = mu_tag_to_float(m.group("mu_tag"))
            n_files += 1

            # Compute sliding length from trajectory
            final_sl = compute_final_sl(ep_path)

            # Look up entanglement metric value
            key = (N, AR, id_str, metric, rod)
            value = extreme_lookup.get(key)
            if value is None:
                missing_extreme += 1
                value = np.nan

            rows.append({
                "N": N,
                "AR": AR,
                "ID": id_str,
                "Metric": metric,
                "RodIndex": rod,
                "mu": mu,
                "final_sl": final_sl,
                "Value": value,
            })

    print(f"Processed {n_files} endpoint files")
    print(f"Missing extreme lookup: {missing_extreme}")

    if not rows:
        print("ERROR: No rows collected!")
        return

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["N", "AR", "ID", "Metric", "mu"]).reset_index(drop=True)

    print(f"Result shape: {result_df.shape}")
    print(f"final_sl range: {result_df['final_sl'].min():.4f} – {result_df['final_sl'].max():.4f}")
    print(f"Value range:    {result_df['Value'].min():.4f} – {result_df['Value'].max():.4f}")
    print(f"NaN final_sl:   {result_df['final_sl'].isna().sum()}")
    print(f"NaN Value:      {result_df['Value'].isna().sum()}")
    print(result_df.head(5).to_string())

    result_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()
