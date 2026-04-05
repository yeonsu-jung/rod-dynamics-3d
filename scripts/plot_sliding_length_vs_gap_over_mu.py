#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_reptation_tag(tag: str) -> dict[str, float | int | str]:
    parts = tag.split("_")
    parsed: dict[str, float | int | str] = {"tag": tag}
    for part in parts:
        if part.startswith("gap"):
            parsed["gap"] = float(part[3:])
        elif part.startswith("mu"):
            parsed["mu"] = float(part[2:])
        elif part.startswith("t") and part[1:].isdigit():
            parsed["trial"] = int(part[1:])
        elif part.startswith("init"):
            parsed["init"] = part
    return parsed


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
        "--mean-output",
        default="results/reptation_ar200_thermal_sv0p1_sw0p2_dt1e4_tangent_fullsweep/sliding_length_vs_gap_over_mu_mean.png",
        help="Mean plot output path",
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
    mean_path = Path(args.mean_output)
    csv_path = Path(args.csv_output)

    df = pd.read_csv(input_path)
    parsed = pd.DataFrame(df["tag"].map(parse_reptation_tag).tolist())
    df = pd.concat([df, parsed[["gap", "mu", "trial"]]], axis=1)
    df = df[df["mu"] > 0.0].copy()
    df["gap_over_mu"] = df["gap"] / df["mu"]
    df["sliding_length"] = df["stop_py"].abs()
    if "resolved" not in df.columns:
        df["resolved"] = df["stop_time"] < (df["final_time"] - 1e-9)

    title_suffix = f" {args.title_suffix.strip()}" if args.title_suffix.strip() else ""
    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    mean_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    single_color = "#1f77b4"

    def make_scatter(figsize):
        fig, ax = plt.subplots(figsize=figsize)
        for gap in sorted(df["gap"].unique()):
            sub = df[df["gap"] == gap]
            ax.scatter(sub["gap_over_mu"], sub["sliding_length"], s=28, alpha=0.75, color=single_color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$g / \mu$")
        ax.set_ylabel(r"$L / l$")
        ax.set_title(f"Sliding length vs gap / mu{title_suffix}")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        return fig

    def make_summary_plot(col, title_prefix, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        for gap in sorted(summary["gap"].unique()):
            sub = summary[summary["gap"] == gap].sort_values("gap_over_mu")
            ax.plot(sub["gap_over_mu"], sub[col], marker="o", linestyle="none", color=single_color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$g / \mu$")
        ax.set_ylabel(r"$L / l$")
        ax.set_title(f"{title_prefix} sliding length vs gap / mu{title_suffix}")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        return fig

    make_scatter((7.6, 4.8)).savefig(scatter_path, dpi=200)
    make_scatter((4, 3)).savefig(scatter_path.with_stem(scatter_path.stem + "_small"), dpi=200)

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

    make_summary_plot("median_sliding_length", "Median", (7.6, 4.8)).savefig(summary_path, dpi=200)
    make_summary_plot("median_sliding_length", "Median", (4, 3)).savefig(summary_path.with_stem(summary_path.stem + "_small"), dpi=200)

    make_summary_plot("mean_sliding_length", "Mean", (7.6, 4.8)).savefig(mean_path, dpi=200)
    make_summary_plot("mean_sliding_length", "Mean", (4, 3)).savefig(mean_path.with_stem(mean_path.stem + "_small"), dpi=200)

    print(f"Wrote {scatter_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {mean_path}")
    print(f"Wrote {csv_path}")
    print("Also wrote _small variants at 4x3 inches.")


if __name__ == "__main__":
    main()