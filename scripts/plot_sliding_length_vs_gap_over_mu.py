#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot sliding length versus gap/mu from tangent-stop summaries"
    )
    parser.add_argument(
        "--input",
        default="results/reptation_ar200_thermal_sv0p1_sw0p2_dt1e4_tangent_fullsweep/tangent_stop_summary.csv",
        help="Input summary CSV",
    )
    parser.add_argument(
        "--scatter-output",
        default="results/reptation_ar200_thermal_sv0p1_sw0p2_dt1e4_tangent_fullsweep/sliding_length_vs_gap_over_mu_scatter.png",
        help="Scatter plot output path",
    )
    parser.add_argument(
        "--summary-output",
        default="results/reptation_ar200_thermal_sv0p1_sw0p2_dt1e4_tangent_fullsweep/sliding_length_vs_gap_over_mu_summary.png",
        help="Summary plot output path",
    )
    parser.add_argument(
        "--csv-output",
        default="results/reptation_ar200_thermal_sv0p1_sw0p2_dt1e4_tangent_fullsweep/sliding_length_vs_gap_over_mu_summary.csv",
        help="Summary table output path",
    )
    parser.add_argument(
        "--title-suffix",
        default="",
        help="Optional suffix appended to plot titles",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    scatter_path = Path(args.scatter_output)
    summary_path = Path(args.summary_output)
    csv_path = Path(args.csv_output)

    df = pd.read_csv(input_path)
    parsed = df["tag"].map(lambda tag: tag.split("_"))
    df["gap"] = parsed.map(lambda parts: float(parts[1][3:]))
    df["mu"] = parsed.map(lambda parts: float(parts[2][2:]))
    df["trial"] = parsed.map(lambda parts: int(parts[3][1:]))
    df = df[df["mu"] > 0.0].copy()
    df["gap_over_mu"] = df["gap"] / df["mu"]
    df["sliding_length"] = df["stop_py"].abs()
    if "resolved" not in df.columns:
        df["resolved"] = df["stop_time"] < (df["final_time"] - 1e-9)

    title_suffix = f" {args.title_suffix.strip()}" if args.title_suffix.strip() else ""
    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    colors = {0.001: "#1f77b4", 0.01: "#ff7f0e", 0.1: "#2ca02c"}
    for gap in sorted(df["gap"].unique()):
        sub = df[df["gap"] == gap]
        ax.scatter(
            sub["gap_over_mu"],
            sub["sliding_length"],
            s=28,
            alpha=0.75,
            color=colors.get(gap, None),
            label=f"gap={gap:g}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gap / mu")
    ax.set_ylabel("Sliding length at detected stop |stop_py|")
    ax.set_title(f"Sliding length vs gap / mu{title_suffix}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=200)

    summary = (
        df.groupby(["gap", "mu", "gap_over_mu"], as_index=False)
        .agg(
            n=("sliding_length", "size"),
            resolved_count=("resolved", "sum"),
            median_sliding_length=("sliding_length", "median"),
            mean_sliding_length=("sliding_length", "mean"),
            min_sliding_length=("sliding_length", "min"),
            max_sliding_length=("sliding_length", "max"),
        )
        .sort_values("gap_over_mu")
    )
    summary.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for gap in sorted(summary["gap"].unique()):
        sub = summary[summary["gap"] == gap].sort_values("gap_over_mu")
        ax.plot(
            sub["gap_over_mu"],
            sub["median_sliding_length"],
            marker="o",
            linewidth=1.8,
            color=colors.get(gap, None),
            label=f"gap={gap:g}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gap / mu")
    ax.set_ylabel("Median sliding length |stop_py|")
    ax.set_title(f"Median sliding length vs gap / mu{title_suffix}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(summary_path, dpi=200)

    print(f"Wrote {scatter_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()