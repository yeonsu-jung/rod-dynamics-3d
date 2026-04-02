#!/usr/bin/env python3
"""Analyze reptation sweep outputs.

This script expects per-run reptation summary CSVs and, optionally, matching
per-run per-rod CSVs. For signed sliding length, it extracts the last recorded
`py` value for rod 0 from each per-rod file and aggregates results by `gap` and
`mu`.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def load_summary_rows(out_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(out_dir / "rept_*.csv")))
    rows = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        row = df.iloc[-1].copy()
        row["summary_file"] = os.path.basename(path)
        row["tag"] = os.path.basename(path)[len("rept_"):-len(".csv")]
        rows.append(row)

    if not rows:
        raise SystemExit(f"No reptation summary CSVs found in {out_dir}")

    summary = pd.DataFrame(rows)
    if "gap" not in summary.columns and {"R_cyl", "d_rod"}.issubset(summary.columns):
        summary["gap"] = summary["R_cyl"] - 0.5 * summary["d_rod"]
    return summary


def extract_final_y(perrod_path: Path, rod_id: int) -> float | None:
    try:
        df = pd.read_csv(perrod_path, comment="#")
    except Exception:
        return None
    if df.empty or "rod" not in df.columns or "py" not in df.columns:
        return None
    rod_df = df[df["rod"] == rod_id]
    if rod_df.empty:
        return None
    return float(rod_df.iloc[-1]["py"])


def attach_final_y(summary: pd.DataFrame, out_dir: Path, rod_id: int) -> pd.DataFrame:
    final_y = []
    perrod_file = []
    for tag in summary["tag"]:
        path = out_dir / f"perrod_{tag}.csv"
        perrod_file.append(path.name if path.exists() else "")
        final_y.append(extract_final_y(path, rod_id) if path.exists() else None)
    summary = summary.copy()
    summary["perrod_file"] = perrod_file
    summary["final_y"] = final_y
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reptation sweep outputs")
    parser.add_argument("--out-dir", default="results/reptation",
                        help="Directory containing rept_*.csv and perrod_*.csv")
    parser.add_argument("--rod-id", type=int, default=0,
                        help="Rod id to analyze in per-rod outputs")
    parser.add_argument("--combined-out", default=None,
                        help="Optional path to write per-run combined CSV")
    parser.add_argument("--summary-out", default=None,
                        help="Optional path to write aggregated CSV")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summary = load_summary_rows(out_dir)
    summary = attach_final_y(summary, out_dir, args.rod_id)

    preferred_cols = [
        "mu", "gap", "R_cyl", "L_rod", "d_rod", "net_displacement",
        "total_path_length", "wall_hits", "sim_time", "final_KE",
        "final_y", "stop_slide_vel_threshold", "stop_slide_vel_min_steps",
        "summary_file", "perrod_file",
    ]
    present_cols = [col for col in preferred_cols if col in summary.columns]
    combined = summary[present_cols].sort_values(["gap", "mu"]).reset_index(drop=True)

    agg_map = {
        "final_y": ["mean", "std", "min", "max"],
        "net_displacement": ["mean", "std"],
        "total_path_length": ["mean", "std"],
        "wall_hits": ["mean"],
        "sim_time": ["mean"],
        "final_KE": ["mean"],
    }
    agg_map = {k: v for k, v in agg_map.items() if k in combined.columns}
    grouped = combined.groupby(["gap", "mu"], dropna=False).agg(agg_map).reset_index()
    grouped.columns = [
        col if isinstance(col, str) else "_".join(str(x) for x in col if x)
        for col in grouped.columns.to_flat_index()
    ]
    grouped = grouped.sort_values(["gap", "mu"]).reset_index(drop=True)

    print("=== Per-run results ===")
    print(combined.to_string(index=False))
    print("\n=== Aggregated by gap and mu ===")
    print(grouped.to_string(index=False))

    if args.combined_out:
        Path(args.combined_out).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(args.combined_out, index=False)
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(args.summary_out, index=False)


if __name__ == "__main__":
    main()
