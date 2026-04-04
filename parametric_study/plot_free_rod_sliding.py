#!/usr/bin/env python3
"""
Plot 2x2 relative sliding vs scaled extreme metrics with forced y=ax fit.

Reproduces the style of:
  analysis_outputs_cohort10/final_sliding_all_extremes_2x2_cohort10_linear_fit.svg

Input: sliding_length_summary.csv from compile_free_rod_sliding.py
Columns: N, AR, seed, metric, rod, mu, sliding_length, final_displacement,
         total_frames, final_time, metric_value
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Physical parameters (must match submission) ──────────────────────────────
SIGMA_V = 0.1       # m/s translational velocity scale
SIM_TIME = 100.0     # seconds (100k steps × 0.001 s/step)
BALLISTIC_LIMIT = SIGMA_V * SIM_TIME  # = 10

# ── Plot styling ─────────────────────────────────────────────────────────────
MU_COLORS = {
    0.05: "#9467bd",
    0.1:  "#1f77b4",
    0.15: "#17becf",
    0.2:  "#2ca02c",
    0.4:  "#ff7f0e",
    1.0:  "#d62728",
}
LAYOUT = [
    ("MaxFSA", r"MaxFSA / $(2\pi\mu)$"),
    ("MinFSA", r"MinFSA / $(2\pi\mu)$"),
    ("MaxFTA", r"$\sqrt{\mathrm{MaxFTA}} / \mu$"),
    ("MinFTA", r"$\sqrt{\mathrm{MinFTA}} / \mu$"),
]


def transform_x(metric: str, value: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Apply the scaling transform for x-axis."""
    if "FSA" in metric:
        return value / (2 * np.pi * mu)
    else:  # FTA
        return np.sqrt(value) / mu


def forced_linear_fit(x: np.ndarray, y: np.ndarray):
    """Fit y = a*x  (slope=1 in log-log, i.e. proportional).

    Returns coefficient `a` via: a = exp(mean(log(y) - log(x)))
    """
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 2:
        return None
    log_ratio = np.log(y[mask]) - np.log(x[mask])
    a = np.exp(np.mean(log_ratio))
    return a


def binned_representative(x: np.ndarray, y: np.ndarray,
                          n_bins: int = 15, min_pts: int = 5):
    """Compute binned representative points (geometric-mean x, median y)."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if len(x) < min_pts:
        return None, None
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    bin_id = np.digitize(x, edges) - 1
    bx, by = [], []
    for i in range(n_bins):
        m = bin_id == i
        if m.sum() < min_pts:
            continue
        bx.append(np.exp(np.mean(np.log(x[m]))))
        by.append(np.median(y[m]))
    if not bx:
        return None, None
    return np.array(bx), np.array(by)


def plot_2x2(df: pd.DataFrame, output_dir: Path, tag: str,
             mu_filter: list[float] | None = None,
             sl_threshold: float = 0.0,
             n_bins: int = 15) -> None:
    """Generate the 2x2 relative sliding vs scaled extreme metrics plot."""

    sub = df.copy()
    # Exclude mu=0 (no friction → no meaningful scaling)
    sub = sub[sub["mu"] > 0]
    if mu_filter is not None:
        sub = sub[sub["mu"].isin(mu_filter)]
    # Filter tiny sliding lengths (numerical noise)
    if sl_threshold > 0:
        sub = sub[sub["sliding_length"] >= sl_threshold]
    # Drop rows with missing metric value
    sub = sub.dropna(subset=["metric_value"])
    sub = sub[sub["metric_value"] > 0]

    mus_present = sorted(sub["mu"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panel_map = {
        "MaxFSA": axes[0, 0], "MinFSA": axes[0, 1],
        "MaxFTA": axes[1, 0], "MinFTA": axes[1, 1],
    }

    for metric, xlabel in LAYOUT:
        ax = panel_map[metric]
        ms = sub[sub["metric"] == metric].copy()
        if ms.empty:
            ax.set_title(metric)
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        ms["x_tr"] = transform_x(metric, ms["metric_value"].values, ms["mu"].values)
        ms = ms[np.isfinite(ms["x_tr"]) & (ms["x_tr"] > 0)]

        # Scatter by mu
        for mu in mus_present:
            md = ms[ms["mu"] == mu]
            if md.empty:
                continue
            color = MU_COLORS.get(mu, "gray")
            ax.scatter(
                md["x_tr"], md["sliding_length"],
                s=14, alpha=0.3, color=color, label=f"μ = {mu}", rasterized=True,
            )

        # Binned representative
        bx, by = binned_representative(ms["x_tr"].values, ms["sliding_length"].values,
                                       n_bins=n_bins)
        if bx is not None:
            ax.plot(bx, by, color="black", linewidth=2.0, alpha=0.9, zorder=9)
            ax.scatter(bx, by, s=44, color="white", edgecolor="black",
                       linewidth=1.2, zorder=10, label="Binned repr.")

        # Forced y = ax fit
        a = forced_linear_fit(ms["x_tr"].values, ms["sliding_length"].values)
        if a is not None:
            x_range = np.logspace(
                np.log10(ms["x_tr"].min()), np.log10(ms["x_tr"].max()), 200
            )
            ax.plot(x_range, a * x_range, color="red", linewidth=2.0, zorder=8,
                    label="Linear fitting")
            # Annotation
            ax.text(
                0.04, 0.92,
                f"$y = {a:.2f}\\,x$",
                transform=ax.transAxes, fontsize=11, color="red", va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        # Ballistic limit
        ax.axhline(BALLISTIC_LIMIT, color="red", linestyle="--", linewidth=2, alpha=0.7)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Relative sliding length", fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":", which="both")

    # Legend on bottom-right panel
    axes[1, 1].legend(fontsize=8, loc="lower right", ncol=2)

    mu_str = ", ".join(str(m) for m in mus_present)
    steps_k = int(round(sub["final_time"].max() / 0.001 / 1000))  # recover from dt
    dt_str = "10^{-3}"
    thresh_str = f"{sl_threshold:.0e}" if sl_threshold > 0 else "none"
    fig.suptitle(
        f"Relative sliding vs scaled extreme metrics — {tag}  [forced $y = ax$ fit]\n"
        rf"$\mu \in \{{{mu_str}\}}$  |  {steps_k}k steps, dt=${ dt_str }$"
        + (f", threshold={thresh_str}" if sl_threshold > 0 else ""),
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for fmt in ("png", "svg"):
        out = output_dir / f"final_sliding_all_extremes_2x2_{tag}_linear_fit.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def plot_2x1_max(df: pd.DataFrame, output_dir: Path, tag: str,
                 mu_filter: list[float] | None = None,
                 sl_threshold: float = 0.0,
                 n_bins: int = 15) -> None:
    """2x1 figure with only MaxFSA and MaxFTA."""
    sub = df.copy()
    sub = sub[sub["mu"] > 0]
    if mu_filter is not None:
        sub = sub[sub["mu"].isin(mu_filter)]
    if sl_threshold > 0:
        sub = sub[sub["sliding_length"] >= sl_threshold]
    sub = sub.dropna(subset=["metric_value"])
    sub = sub[sub["metric_value"] > 0]

    mus_present = sorted(sub["mu"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    max_layout = [
        ("MaxFSA", axes[0], r"MaxFSA / $(2\pi\mu)$"),
        ("MaxFTA", axes[1], r"$\sqrt{\mathrm{MaxFTA}} / \mu$"),
    ]

    for metric, ax, xlabel in max_layout:
        ms = sub[sub["metric"] == metric].copy()
        if ms.empty:
            ax.set_title(metric)
            continue
        ms["x_tr"] = transform_x(metric, ms["metric_value"].values, ms["mu"].values)
        ms = ms[np.isfinite(ms["x_tr"]) & (ms["x_tr"] > 0)]

        for mu in mus_present:
            md = ms[ms["mu"] == mu]
            if md.empty:
                continue
            color = MU_COLORS.get(mu, "gray")
            ax.scatter(md["x_tr"], md["sliding_length"],
                       s=14, alpha=0.3, color=color, label=f"μ = {mu}", rasterized=True)

        bx, by = binned_representative(ms["x_tr"].values, ms["sliding_length"].values, n_bins=n_bins)
        if bx is not None:
            ax.plot(bx, by, color="black", linewidth=2.0, alpha=0.9, zorder=9)
            ax.scatter(bx, by, s=44, color="white", edgecolor="black",
                       linewidth=1.2, zorder=10, label="Binned repr.")

        a = forced_linear_fit(ms["x_tr"].values, ms["sliding_length"].values)
        if a is not None:
            x_range = np.logspace(np.log10(ms["x_tr"].min()), np.log10(ms["x_tr"].max()), 200)
            ax.plot(x_range, a * x_range, color="red", linewidth=2.0, zorder=8)
            ax.text(0.04, 0.92, f"$y = {a:.2f}\\,x$",
                    transform=ax.transAxes, fontsize=11, color="red", va="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.axhline(BALLISTIC_LIMIT, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Relative sliding length", fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":", which="both")

    axes[1].legend(fontsize=8, loc="lower right", ncol=2)
    fig.suptitle(f"Relative sliding vs scaled Max metrics — {tag}  [forced $y = ax$ fit]",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for fmt in ("png", "svg"):
        out = output_dir / f"final_sliding_2x1_max_{tag}_linear_fit.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def plot_2x1_min(df: pd.DataFrame, output_dir: Path, tag: str,
                 mu_filter: list[float] | None = None,
                 sl_threshold: float = 0.0,
                 n_bins: int = 15) -> None:
    """2x1 figure with only MinFSA and MinFTA."""
    sub = df.copy()
    sub = sub[sub["mu"] > 0]
    if mu_filter is not None:
        sub = sub[sub["mu"].isin(mu_filter)]
    if sl_threshold > 0:
        sub = sub[sub["sliding_length"] >= sl_threshold]
    sub = sub.dropna(subset=["metric_value"])
    sub = sub[sub["metric_value"] > 0]

    mus_present = sorted(sub["mu"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    min_layout = [
        ("MinFSA", axes[0], r"MinFSA / $(2\pi\mu)$"),
        ("MinFTA", axes[1], r"$\sqrt{\mathrm{MinFTA}} / \mu$"),
    ]

    for metric, ax, xlabel in min_layout:
        ms = sub[sub["metric"] == metric].copy()
        if ms.empty:
            ax.set_title(metric)
            continue
        ms["x_tr"] = transform_x(metric, ms["metric_value"].values, ms["mu"].values)
        ms = ms[np.isfinite(ms["x_tr"]) & (ms["x_tr"] > 0)]

        for mu in mus_present:
            md = ms[ms["mu"] == mu]
            if md.empty:
                continue
            color = MU_COLORS.get(mu, "gray")
            ax.scatter(md["x_tr"], md["sliding_length"],
                       s=14, alpha=0.3, color=color, label=f"μ = {mu}", rasterized=True)

        bx, by = binned_representative(ms["x_tr"].values, ms["sliding_length"].values, n_bins=n_bins)
        if bx is not None:
            ax.plot(bx, by, color="black", linewidth=2.0, alpha=0.9, zorder=9)
            ax.scatter(bx, by, s=44, color="white", edgecolor="black",
                       linewidth=1.2, zorder=10, label="Binned repr.")

        a = forced_linear_fit(ms["x_tr"].values, ms["sliding_length"].values)
        if a is not None:
            x_range = np.logspace(np.log10(ms["x_tr"].min()), np.log10(ms["x_tr"].max()), 200)
            ax.plot(x_range, a * x_range, color="red", linewidth=2.0, zorder=8)
            ax.text(0.04, 0.92, f"$y = {a:.2f}\\,x$",
                    transform=ax.transAxes, fontsize=11, color="red", va="top",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.axhline(BALLISTIC_LIMIT, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Relative sliding length", fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":", which="both")

    axes[1].legend(fontsize=8, loc="lower right", ncol=2)
    fig.suptitle(f"Relative sliding vs scaled Min metrics — {tag}  [forced $y = ax$ fit]",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for fmt in ("png", "svg"):
        out = output_dir / f"final_sliding_2x1_min_{tag}_linear_fit.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def plot_sliding_vs_mu_boxplot(df: pd.DataFrame, output_dir: Path, tag: str) -> None:
    """Box plot of sliding length vs mu, one panel per metric."""
    sub = df[df["mu"] > 0].copy()
    metrics = ["MaxFSA", "MinFSA", "MaxFTA", "MinFTA"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panel_map = {
        "MaxFSA": axes[0, 0], "MinFSA": axes[0, 1],
        "MaxFTA": axes[1, 0], "MinFTA": axes[1, 1],
    }

    for metric in metrics:
        ax = panel_map[metric]
        ms = sub[sub["metric"] == metric]
        mus = sorted(ms["mu"].unique())
        data = [ms[ms["mu"] == mu]["sliding_length"].values for mu in mus]
        bp = ax.boxplot(data, labels=[f"{mu}" for mu in mus], patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#1f77b4")
            patch.set_alpha(0.5)
        ax.axhline(BALLISTIC_LIMIT, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel(r"Friction $\mu$")
        ax.set_ylabel("Sliding length")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y", linestyle=":")

    fig.suptitle(f"Sliding length distribution by friction — {tag}", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for fmt in ("png", "svg"):
        out = output_dir / f"sliding_vs_mu_boxplot_{tag}.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot relative sliding vs scaled extreme metrics (forced y=ax fit)."
    )
    parser.add_argument("--csv", type=Path, required=True,
                        help="sliding_length_summary.csv path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="free_rod_nsc_gaussian")
    parser.add_argument("--mu-filter", type=float, nargs="+", default=None,
                        help="Only include these mu values (e.g. 0.1 0.2 0.4 1.0)")
    parser.add_argument("--sl-threshold", type=float, default=1e-5,
                        help="Minimum sliding length to include")
    parser.add_argument("--n-bins", type=int, default=15,
                        help="Number of log-spaced bins for representative points")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"  N: {sorted(df.N.unique())}")
    print(f"  AR: {sorted(df.AR.unique())}")
    print(f"  mu: {sorted(df.mu.unique())}")
    print(f"  metrics: {sorted(df.metric.unique())}")
    print(f"  sliding_length range: {df.sliding_length.min():.2e} — {df.sliding_length.max():.2e}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    plot_2x2(df, args.output_dir, args.tag,
             mu_filter=args.mu_filter, sl_threshold=args.sl_threshold,
             n_bins=args.n_bins)
    plot_2x1_max(df, args.output_dir, args.tag,
                 mu_filter=args.mu_filter, sl_threshold=args.sl_threshold,
                 n_bins=args.n_bins)
    plot_2x1_min(df, args.output_dir, args.tag,
                 mu_filter=args.mu_filter, sl_threshold=args.sl_threshold,
                 n_bins=args.n_bins)
    plot_sliding_vs_mu_boxplot(df, args.output_dir, args.tag)

    print("\nDone!")


if __name__ == "__main__":
    main()
