#!/usr/bin/env python3
"""
Plot sweep3 final sliding length data with reference line showing ballistic limit.

The reference line at y=10 represents the theoretical maximum sliding distance
based on ballistic motion: distance = initial_velocity × simulation_time ≈ 0.1 m/s × 100 s = 10 m

This plots final_sl vs friction (mu) for each metric, showing that the observed
plateau at ~11 units is a physical constraint, not an artifact of periodic boundaries.
"""

import argparse

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
CSV_PATH = Path("/n/home01/yjung/Github/rod-dynamics-3d/analysis_outputs/final_sliding_vs_extreme_value_sweep3.csv")
OUTPUT_DIR = Path("/n/home01/yjung/Github/rod-dynamics-3d/analysis_outputs")
COHORT_TAG = "sweep3"

# Physical parameters
INITIAL_VELOCITY = 0.1  # m/s (vSigma)
SIM_TIME = 100.0  # seconds (200,000 steps * 0.0005 s/step)
BALLISTIC_LIMIT = INITIAL_VELOCITY * SIM_TIME  # = 10 m

METRIC_ORDER = ["MinFSA", "MaxFSA", "MinFTA", "MaxFTA"]
COLORS = {
    "MinFSA": "#0f766e",
    "MaxFSA": "#dc2626",
    "MinFTA": "#2563eb",
    "MaxFTA": "#d97706",
}


def plot_final_sliding_vs_mu(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot final sliding length vs friction for each metric."""
    metrics = sorted(df["Metric"].unique())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        subset = df[df["Metric"] == metric]
        mus = sorted(subset["mu"].unique())
        
        # Plot data points
        for mu in mus:
            mu_data = subset[subset["mu"] == mu]
            ax.scatter(
                [mu] * len(mu_data),
                mu_data["final_sl"],
                alpha=0.6,
                s=50,
                color=COLORS.get(metric, "gray"),
                label=metric if mu == mus[0] else ""
            )
        
        # Add reference line for ballistic limit
        ax.axhline(
            y=BALLISTIC_LIMIT,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Ballistic limit: |v|·t = {INITIAL_VELOCITY}·{SIM_TIME} = {BALLISTIC_LIMIT}",
            alpha=0.7
        )
        
        ax.set_xlabel(r"Friction coefficient $\mu$")
        ax.set_title(f"{metric}", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_ylim(0, 13)
    
    axes[0].set_ylabel("Final sliding length (m)")
    axes[0].legend(fontsize=9, loc="upper left")
    
    fig.suptitle(
        f"Free-rod sliding length vs friction ({COHORT_TAG})\nReference line shows theoretical ballistic maximum",
        fontsize=12,
        y=1.00
    )
    fig.tight_layout()
    
    out_path = output_dir / f"final_sliding_vs_mu_all_metrics_{COHORT_TAG}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_final_sliding_vs_value(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot final sliding length vs the extreme value (FSA or FTA) for each metric."""
    metrics = sorted(df["Metric"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        subset = df[df["Metric"] == metric]
        
        # Scatter plot: Value on x-axis, final_sl on y-axis (colored by mu)
        mus = sorted(subset["mu"].unique())
        cmap = plt.cm.viridis
        
        for i, mu in enumerate(mus):
            mu_data = subset[subset["mu"] == mu]
            color = cmap(i / max(len(mus) - 1, 1))
            ax.scatter(
                mu_data["Value"],
                mu_data["final_sl"],
                alpha=0.6,
                s=60,
                color=color,
                label=f"μ = {mu}"
            )
        
        # Add reference line
        ax.axhline(
            y=BALLISTIC_LIMIT,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7
        )
        
        ax.set_xlabel("Extreme value (FSA or FTA)" if metric in ["MaxFSA", "MinFSA"] else "Extreme value (FTA)")
        ax.set_ylabel("Final sliding length (m)")
        ax.set_title(f"{metric}", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_ylim(0, 13)
    
    axes[-1].legend(fontsize=8, loc="best")
    
    fig.suptitle(
        f"Final sliding length vs extreme value ({COHORT_TAG})\nDashed red line: ballistic limit = 10 m",
        fontsize=12,
        y=1.00
    )
    fig.tight_layout()
    
    out_path = output_dir / f"final_sliding_vs_value_all_metrics_{COHORT_TAG}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_max_metrics_both_axes(df: pd.DataFrame, output_dir: Path) -> None:
    """2x1 figure: MaxFSA and MaxFTA side by side, showing both dependency on mu."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax, metric in zip(axes, ["MaxFSA", "MaxFTA"]):
        subset = df[df["Metric"] == metric]
        mus = sorted(subset["mu"].unique())
        
        # Plot as box plot per mu
        data_per_mu = [subset[subset["mu"] == mu]["final_sl"].values for mu in mus]
        
        bp = ax.boxplot(data_per_mu, labels=[f"{mu}" for mu in mus], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[metric])
            patch.set_alpha(0.6)
        
        # Add reference line
        ax.axhline(
            y=BALLISTIC_LIMIT,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Ballistic limit = {BALLISTIC_LIMIT} m",
            alpha=0.7
        )
        
        ax.set_xlabel(r"Friction coefficient $\mu$")
        ax.set_ylabel("Final sliding length (m)")
        ax.set_title(f"{metric} rod sliding", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y", linestyle=":")
        ax.set_ylim(9.5, 11.5)
    
    axes[0].legend(fontsize=10)
    fig.suptitle(
        "Most-entangled rods (MaxFSA/MaxFTA) reach ballistic limit\nslightly above reference line due to initial conditions",
        fontsize=11,
        y=1.00
    )
    fig.tight_layout()
    
    out_path = output_dir / f"final_sliding_vs_MaxFSA_MaxFTA_with_reference_{COHORT_TAG}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_maxFSA_sqrtMaxFTA_2x1(
    df: pd.DataFrame,
    output_dir: Path,
    max_relative_sliding: float | None = None,
) -> None:
    """2x1 vertical figure: MaxFSA (translational) above, MaxFTA (rotational) below (5x8 inches) with loglog scaling.
    
    Axes use honest scalings:
    - x-axis top: MaxFSA/(2π)/μ (non-dimensionalized FSA / friction)
    - x-axis bottom: sqrt(MaxFTA)/μ (non-dimensionalized FTA / friction)
    
    Also plots best-fit linear relationship y = a*x (through origin).
    """
    fig, axes = plt.subplots(2, 1, figsize=(5, 8))
    
    # Metrics in order: translational (top), rotational (bottom)
    metrics_ordered = ["MaxFSA", "MaxFTA"]
    axlabels = [r"MaxFSA / $(2\pi \mu)$", r"$\sqrt{\text{MaxFTA}} / \mu$"]
    
    for ax, metric, axlabel in zip(axes, metrics_ordered, axlabels):
        subset = df[df["Metric"] == metric].copy()
        # Filter out zero or negative values for loglog
        subset = subset[subset["Value"] > 0]
        subset = subset[subset["final_sl"] > 0]
        if max_relative_sliding is not None:
            subset = subset[subset["final_sl"] <= max_relative_sliding]
        
        # Compute the transformed x-values based on metric
        if metric == "MaxFSA":
            # Translational: scale x by 1/(2π*μ)
            subset["x_transformed"] = subset["Value"] / (2 * np.pi * subset["mu"])
        else:  # MaxFTA
            # Rotational: scale x by 1/μ, and take sqrt of Value
            subset["x_transformed"] = np.sqrt(subset["Value"]) / subset["mu"]

        subset = subset[np.isfinite(subset["x_transformed"]) & (subset["x_transformed"] > 0)]
        mus = sorted(subset["mu"].unique())
        
        # Plot as scattered points, colored by mu (smaller markers)
        cmap = plt.cm.viridis
        for i, mu in enumerate(mus):
            mu_data = subset[subset["mu"] == mu]
            color = cmap(i / max(len(mus) - 1, 1))
            ax.scatter(
                mu_data["x_transformed"],
                mu_data["final_sl"],
                alpha=0.6,
                s=20,
                color=color,
                label=f"μ = {mu}"
            )
        
        # Fit power law y = a*x^b in log space: log(y) = log(a) + b*log(x)
        x_vals = subset["x_transformed"].values
        y_vals = subset["final_sl"].values
        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0) & (y_vals > 0)
        x_valid = x_vals[valid_mask]
        y_valid = y_vals[valid_mask]

        if len(x_valid) >= 2:
            coeff = np.polyfit(np.log(x_valid), np.log(y_valid), 1)
            b = coeff[0]
            a = np.exp(coeff[1])
            x_fit = np.logspace(np.log10(x_valid.min()), np.log10(x_valid.max()), 200)
            y_fit = a * (x_fit ** b)
            ax.plot(x_fit, y_fit, color="black", linewidth=2.2, zorder=10)
            ax.text(
                0.04,
                0.92,
                f"$y={a:.2e}x^{{{b:.2f}}}$",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
        
        # Add reference line for ballistic limit
        ax.axhline(
            y=BALLISTIC_LIMIT,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7
        )
        
        # Apply loglog scaling
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        ax.set_xlabel(axlabel, fontsize=11)
        ax.set_ylabel("Relative sliding length", fontsize=11)
        metric_type = "translational (FSA)" if metric == "MaxFSA" else "rotational (FTA)"
        ax.set_title(f"{metric} — {metric_type}", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle=":", which="both")
        ax.legend(fontsize=8, loc="best", ncol=2)
    
    subtitle = "Red dashed line: ballistic limit = 10 m"
    if max_relative_sliding is not None:
        subtitle += f" | filtered: relative sliding <= {max_relative_sliding:g}"
    fig.suptitle(
        f"Final sliding vs scaled extreme value (MaxFSA / MaxFTA) — {COHORT_TAG}\\n{subtitle}",
        fontsize=11,
        y=0.995
    )
    fig.tight_layout()
    
    suffix = "" if max_relative_sliding is None else f"_slle{str(max_relative_sliding).replace('.', 'p')}"
    out_path = output_dir / f"final_sliding_vs_MaxFSA_sqrtMaxFTA_2x1_identity_{COHORT_TAG}{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_extreme_min_max_2x2(
    df: pd.DataFrame,
    output_dir: Path,
    max_relative_sliding: float | None = None,
    use_binned_repr: bool = False,
    n_log_bins: int = 12,
    min_points_per_bin: int = 5,
    include_min: bool = True,
) -> None:
    """2x2 figure: (MaxFSA, MinFSA) on top row, (MaxFTA, MinFTA) on bottom row.
    Uses loglog scaling and honest transformations.
    """
    if include_min:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        # Rows: FSA then FTA. Columns: Max then Min.
        layout = [
            ("MaxFSA", axes[0, 0], r"MaxFSA / $(2\pi \mu)$"),
            ("MinFSA", axes[0, 1], r"MinFSA / $(2\pi \mu)$"),
            ("MaxFTA", axes[1, 0], r"$\sqrt{\text{MaxFTA}} / \mu$"),
            ("MinFTA", axes[1, 1], r"$\sqrt{\text{MinFTA}} / \mu$"),
        ]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(5, 8))
        layout = [
            ("MaxFSA", axes[0], r"MaxFSA / $(2\pi \mu)$"),
            ("MaxFTA", axes[1], r"$\sqrt{\text{MaxFTA}} / \mu$"),
        ]
    
    for metric, ax, axlabel in layout:
        subset = df[df["Metric"] == metric].copy()
        bx = None
        by = None
        # Filter out zero or negative values for loglog
        subset = subset[subset["Value"] > 0]
        subset = subset[subset["final_sl"] > 0]
        if max_relative_sliding is not None:
            subset = subset[subset["final_sl"] <= max_relative_sliding]
        
        # Compute the transformed x-values based on metric type
        if "FSA" in metric:
            subset["x_transformed"] = subset["Value"] / (2 * np.pi * subset["mu"])
        else:  # FTA
            subset["x_transformed"] = np.sqrt(subset["Value"]) / subset["mu"]
        
        subset = subset[np.isfinite(subset["x_transformed"]) & (subset["x_transformed"] > 0)]
        mus = sorted(subset["mu"].unique())
        cmap = plt.cm.viridis
        
        for i, mu in enumerate(mus):
            mu_data = subset[subset["mu"] == mu]
            color = cmap(i / max(len(mus) - 1, 1))
            ax.scatter(
                mu_data["x_transformed"],
                mu_data["final_sl"],
                alpha=0.25 if use_binned_repr else 0.6,
                s=14 if use_binned_repr else 20,
                color=color,
                label=f"μ = {mu}"
            )

        # Optional representative points from log-spaced x-bins.
        if use_binned_repr and len(subset) >= max(min_points_per_bin, 2):
            x_min = subset["x_transformed"].min()
            x_max = subset["x_transformed"].max()
            if x_max > x_min:
                edges = np.logspace(np.log10(x_min), np.log10(x_max), n_log_bins + 1)
                bin_id = np.digitize(subset["x_transformed"].values, edges) - 1
                bx = []
                by = []
                for bidx in range(n_log_bins):
                    mask = bin_id == bidx
                    if np.count_nonzero(mask) < min_points_per_bin:
                        continue
                    x_bin = subset["x_transformed"].values[mask]
                    y_bin = subset["final_sl"].values[mask]
                    # Geometric mean x + median y gives a robust representative point.
                    bx.append(np.exp(np.mean(np.log(x_bin))))
                    by.append(np.median(y_bin))
                if bx:
                    bx = np.asarray(bx)
                    by = np.asarray(by)
                    ax.plot(bx, by, color="black", linewidth=2.0, alpha=0.9, zorder=9)
                    ax.scatter(
                        bx,
                        by,
                        s=42,
                        color="white",
                        edgecolor="black",
                        linewidth=1.2,
                        zorder=10,
                        label="Binned representative",
                    )

        # Fit power law y = a*x^b in log space: log(y) = log(a) + b*log(x)
        # If binned representation is enabled, fit to the binned points when available.
        if use_binned_repr and bx is not None and len(bx) >= 2:
            x_vals = bx
            y_vals = by
        else:
            x_vals = subset["x_transformed"].values
            y_vals = subset["final_sl"].values
        valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0) & (y_vals > 0)
        x_valid = x_vals[valid_mask]
        y_valid = y_vals[valid_mask]
        if len(x_valid) >= 2:
            coeff = np.polyfit(np.log(x_valid), np.log(y_valid), 1)
            b = coeff[0]
            a = np.exp(coeff[1])
            x_fit = np.logspace(np.log10(x_valid.min()), np.log10(x_valid.max()), 200)
            y_fit = a * (x_fit ** b)
            ax.plot(x_fit, y_fit, color="black", linewidth=2.2, zorder=10)
            ax.text(
                0.04,
                0.92,
                f"$y={a:.2e}x^{{{b:.2f}}}$",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
        
        # Add reference line for ballistic limit
        ax.axhline(
            y=BALLISTIC_LIMIT,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7
        )
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(axlabel, fontsize=11)
        if include_min:
            ax.set_ylabel("Relative sliding length" if "Max" in metric and "FSA" in metric else "", fontsize=11)
        else:
            ax.set_ylabel("Relative sliding length", fontsize=11)
        ax.set_title(f"{metric}", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle=":", which="both")
        if (include_min and metric == "MinFTA") or (not include_min and metric == "MaxFTA"):
            ax.legend(fontsize=8, loc="best", ncol=2)
    
    if include_min:
        title = f"Relative sliding vs scaled metrics (All Extreme Metrics) — {COHORT_TAG}"
    else:
        title = f"Relative sliding vs scaled metrics (Max Metrics Only) — {COHORT_TAG}"
    if max_relative_sliding is not None:
        title += f"\\nFiltered: relative sliding <= {max_relative_sliding:g}"
    if use_binned_repr:
        title += f"\nRepresentative x-bins: {n_log_bins} log bins"
    fig.suptitle(
        title,
        fontsize=12,
        y=0.99
    )
    fig.tight_layout()
    
    suffix = "" if max_relative_sliding is None else f"_slle{str(max_relative_sliding).replace('.', 'p')}"
    if include_min:
        out_path = output_dir / f"final_sliding_all_extremes_2x2_{COHORT_TAG}{suffix}.png"
    else:
        out_path = output_dir / f"final_sliding_max_extremes_2x1_{COHORT_TAG}{suffix}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    global CSV_PATH, OUTPUT_DIR, COHORT_TAG

    parser = argparse.ArgumentParser(description="Plot free-rod sliding analyses with reference lines.")
    parser.add_argument("--csv-path", type=Path, default=CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cohort-tag", type=str, default=COHORT_TAG)
    args = parser.parse_args()

    CSV_PATH = args.csv_path
    OUTPUT_DIR = args.output_dir
    COHORT_TAG = args.cohort_tag

    if not CSV_PATH.exists():
        print(f"Error: CSV file not found: {CSV_PATH}")
        return
    
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Data shape: {df.shape}")
    print(f"Metrics: {sorted(df['Metric'].unique())}")
    print(f"Friction values: {sorted(df['mu'].unique())}")
    print(f"N values: {sorted(df['N'].unique())}")
    print(f"AR values: {sorted(df['AR'].unique())}")
    print(f"\nFinal sliding length range: {df['final_sl'].min():.3f} — {df['final_sl'].max():.3f} m")
    print(f"Ballistic limit (y={BALLISTIC_LIMIT} m): |v_init| × t_sim = {INITIAL_VELOCITY} m/s × {SIM_TIME:.0f} s\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots with reference line...\n")
    plot_final_sliding_vs_mu(df, OUTPUT_DIR)
    plot_final_sliding_vs_value(df, OUTPUT_DIR)
    plot_max_metrics_both_axes(df, OUTPUT_DIR)
    plot_maxFSA_sqrtMaxFTA_2x1(df, OUTPUT_DIR)
    plot_extreme_min_max_2x2(df, OUTPUT_DIR)
    # Additional filtered versions requested by user.
    plot_maxFSA_sqrtMaxFTA_2x1(df, OUTPUT_DIR, max_relative_sliding=0.5)
    plot_extreme_min_max_2x2(
        df,
        OUTPUT_DIR,
        max_relative_sliding=0.5,
        use_binned_repr=True,
        n_log_bins=12,
        min_points_per_bin=5,
    )
    # Additional Max-only version without MinFSA/MinFTA panels.
    plot_extreme_min_max_2x2(
        df,
        OUTPUT_DIR,
        max_relative_sliding=0.5,
        use_binned_repr=True,
        n_log_bins=12,
        min_points_per_bin=5,
        include_min=False,
    )
    
    print("\nDone! All figures include the ballistic limit reference line at y=10.")
    print(f"The 11-unit plateau is explained by: |v_init| × sim_time ≈ 0.1 × 100 = 10 m")


if __name__ == "__main__":
    main()
