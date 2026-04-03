#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path(
    "/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/final_entanglement_summary.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "final_entanglement_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact CSV summaries and plots from final_entanglement_summary.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Aggregate CSV produced by collect_final_entanglement.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where reduced CSVs and plots will be written.",
    )
    return parser.parse_args()


def load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_columns = [
        "n_rods",
        "ar",
        "friction",
        "sigma_v",
        "sigma_w",
        "final_frame",
        "final_contacts",
        "final_ke",
        "final_max_overlap",
        "final_gyration_sq",
        "final_reldisp_sq",
        "final_ent_sum",
        "final_ent_pairs",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def write_compact_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    compact_columns = [
        "submission_mode",
        "n_rods",
        "seed",
        "ar",
        "friction",
        "sigma_v",
        "sigma_w",
        "final_frame",
        "final_contacts",
        "final_ke",
        "final_max_overlap",
        "final_reldisp_sq",
        "final_ent_sum",
        "final_ent_pairs",
        "run_dir",
    ]
    compact = df.loc[:, [column for column in compact_columns if column in df.columns]].copy()
    compact = compact.sort_values(["n_rods", "seed", "ar", "friction"]).reset_index(drop=True)
    output_path = output_dir / "final_entanglement_compact.csv"
    compact.to_csv(output_path, index=False)
    return output_path


def write_grouped_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    grouped = (
        df.groupby(["n_rods", "ar", "friction"], dropna=False)
        .agg(
            runs=("final_ent_sum", "size"),
            ent_sum_mean=("final_ent_sum", "mean"),
            ent_sum_std=("final_ent_sum", "std"),
            ent_pairs_mean=("final_ent_pairs", "mean"),
            ent_pairs_std=("final_ent_pairs", "std"),
            final_ke_mean=("final_ke", "mean"),
            final_ke_std=("final_ke", "std"),
            reldisp_mean=("final_reldisp_sq", "mean"),
            reldisp_std=("final_reldisp_sq", "std"),
        )
        .reset_index()
        .sort_values(["n_rods", "ar", "friction"])
    )
    output_path = output_dir / "final_entanglement_grouped.csv"
    grouped.to_csv(output_path, index=False)
    return output_path


def add_series(ax: plt.Axes, subset: pd.DataFrame, x: str, y: str, yerr: str, label: str) -> None:
    ordered = subset.sort_values(x)
    if ordered.empty:
        return
    ax.errorbar(
        ordered[x],
        ordered[y],
        yerr=ordered[yerr],
        fmt="o-",
        ms=3,
        lw=1,
        capsize=2,
        label=label,
    )


def plot_ent_vs_ar(grouped: pd.DataFrame, output_dir: Path) -> Path:
    n_values = sorted(grouped["n_rods"].dropna().unique())
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), squeeze=False)
    for idx, n_rods in enumerate(n_values):
        ax = axes[idx // 3][idx % 3]
        n_df = grouped[grouped["n_rods"] == n_rods]
        for friction in sorted(n_df["friction"].dropna().unique()):
            subset = n_df[n_df["friction"] == friction]
            add_series(ax, subset, "ar", "ent_sum_mean", "ent_sum_std", f"mu={friction:g}")
        ax.set_xscale("log")
        ax.set_title(f"N={int(n_rods)}")
        ax.set_xlabel("AR")
        ax.set_ylabel("Final entanglement sum")
        ax.grid(True, alpha=0.25)
    for idx in range(len(n_values), 9):
        axes[idx // 3][idx % 3].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / "final_ent_sum_vs_ar_by_n.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_ent_vs_friction(grouped: pd.DataFrame, output_dir: Path) -> Path:
    n_values = sorted(grouped["n_rods"].dropna().unique())
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), squeeze=False)
    for idx, n_rods in enumerate(n_values):
        ax = axes[idx // 3][idx % 3]
        n_df = grouped[grouped["n_rods"] == n_rods]
        for ar in sorted(n_df["ar"].dropna().unique()):
            subset = n_df[n_df["ar"] == ar]
            add_series(ax, subset, "friction", "ent_sum_mean", "ent_sum_std", f"AR={int(ar)}")
        ax.set_title(f"N={int(n_rods)}")
        ax.set_xlabel("mu")
        ax.set_ylabel("Final entanglement sum")
        ax.grid(True, alpha=0.25)
    for idx in range(len(n_values), 9):
        axes[idx // 3][idx % 3].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / "final_ent_sum_vs_friction_by_n.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_pairs_vs_ar(grouped: pd.DataFrame, output_dir: Path) -> Path:
    n_values = sorted(grouped["n_rods"].dropna().unique())
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), squeeze=False)
    for idx, n_rods in enumerate(n_values):
        ax = axes[idx // 3][idx % 3]
        n_df = grouped[grouped["n_rods"] == n_rods]
        for friction in sorted(n_df["friction"].dropna().unique()):
            subset = n_df[n_df["friction"] == friction]
            add_series(ax, subset, "ar", "ent_pairs_mean", "ent_pairs_std", f"mu={friction:g}")
        ax.set_xscale("log")
        ax.set_title(f"N={int(n_rods)}")
        ax.set_xlabel("AR")
        ax.set_ylabel("Final entangled pairs")
        ax.grid(True, alpha=0.25)
    for idx in range(len(n_values), 9):
        axes[idx // 3][idx % 3].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = output_dir / "final_ent_pairs_vs_ar_by_n.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.input)
    compact_path = write_compact_csv(df, args.output_dir)
    grouped_path = write_grouped_csv(df, args.output_dir)

    grouped = pd.read_csv(grouped_path)
    ent_ar_plot = plot_ent_vs_ar(grouped, args.output_dir)
    ent_mu_plot = plot_ent_vs_friction(grouped, args.output_dir)
    pairs_plot = plot_pairs_vs_ar(grouped, args.output_dir)

    print(f"Wrote compact CSV: {compact_path}")
    print(f"Wrote grouped CSV: {grouped_path}")
    print(f"Wrote plot: {ent_ar_plot}")
    print(f"Wrote plot: {ent_mu_plot}")
    print(f"Wrote plot: {pairs_plot}")


if __name__ == "__main__":
    main()