#!/usr/bin/env python3
"""plot_thermal_init_summary.py

Aggregates summary.csv files from thermal-init NSC runs across all N values
and generates publication-quality plots:
  1. Normalized entanglement vs AR (one curve per friction, panels by N)
  2. Normalized entanglement vs friction (one curve per AR, panels by N)
  3. Normalized entanglement vs AR (one curve per N, panels by friction)
  4. Final KE vs AR (one curve per friction, panels by N)
  5. RMS relative displacement vs AR

Usage:
  python3 parametric_study/plot_thermal_init_summary.py [--out-dir DIR]
"""

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict
from itertools import groupby as itertools_groupby

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

RUNS_ROOT = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs")
BATCH_PATTERN = "dynamics_nsc_thermal_N{}_sweep"
N_VALUES = [10, 15, 20, 30, 50, 100, 200, 500, 1000]

def load_all_summaries():
    """Load and merge all summary.csv files."""
    rows = []
    for n in N_VALUES:
        csv_path = RUNS_ROOT / BATCH_PATTERN.format(n) / "analysis" / "summary.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping N={n}")
            continue
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row["N"] = int(float(row["N"]))
                    row["AR"] = int(float(row["AR"]))
                    row["friction"] = float(row["friction"])
                    row["ent_norm_end"] = float(row["ent_norm_end"])
                    row["ent_sum_end"] = float(row["ent_sum_end"])
                    row["KE0"] = float(row["KE0"])
                    row["KE_end"] = float(row["KE_end"])
                    row["rms_reldisp_end"] = float(row["rms_reldisp_end"])
                    row["rms_gyr_end"] = float(row["rms_gyr_end"])
                    rows.append(row)
                except (ValueError, KeyError):
                    continue
    print(f"Loaded {len(rows)} data points across {len(set(r['N'] for r in rows))} N values")
    return rows


def group_mean_std(rows, key_field, val_field):
    """Group rows by key_field, compute mean±std of val_field."""
    groups = defaultdict(list)
    for r in rows:
        val = r[val_field]
        if math.isfinite(val):
            groups[r[key_field]].append(val)
    X, Y, Yerr, counts = [], [], [], []
    for k in sorted(groups.keys()):
        vals = groups[k]
        X.append(k)
        Y.append(np.mean(vals))
        Yerr.append(np.std(vals))
        counts.append(len(vals))
    return np.array(X), np.array(Y), np.array(Yerr), counts


def plot_ent_vs_ar_by_friction(rows, out_dir):
    """Panel plot: ent_norm vs AR, one curve per friction, panels by N."""
    n_vals = sorted(set(r["N"] for r in rows))
    frictions = sorted(set(r["friction"] for r in rows))
    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(frictions), vmax=max(frictions))

    ncols = 3
    nrows = math.ceil(len(n_vals) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)

    for idx, n in enumerate(n_vals):
        ax = axes[idx // ncols][idx % ncols]
        n_rows = [r for r in rows if r["N"] == n]
        for fric in frictions:
            fric_rows = [r for r in n_rows if abs(r["friction"] - fric) < 1e-6]
            if not fric_rows:
                continue
            X, Y, Yerr, _ = group_mean_std(fric_rows, "AR", "ent_norm_end")
            if len(X) > 0:
                ax.errorbar(X, Y, yerr=Yerr, fmt="o-", ms=3, lw=1,
                            color=cmap(norm(fric)), label=f"μ={fric}")
        ax.set_xscale("log")
        ax.set_title(f"N={n}", fontsize=10)
        ax.set_xlabel("AR")
        ax.set_ylabel("Norm. Ent.")
        ax.grid(True, alpha=0.2)

    # Remove empty panels
    for idx in range(len(n_vals), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Friction μ", shrink=0.6)

    fig.suptitle("Normalized Entanglement vs AR (thermal init, σ_v=0.1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(out_dir / "ent_norm_vs_ar_by_friction.png", dpi=200)
    plt.close(fig)
    print("Saved ent_norm_vs_ar_by_friction.png")


def plot_ent_vs_friction_by_ar(rows, out_dir):
    """Panel plot: ent_norm vs friction, one curve per AR, panels by N."""
    n_vals = sorted(set(r["N"] for r in rows))
    all_ars = sorted(set(r["AR"] for r in rows))
    cmap = plt.cm.plasma
    norm = Normalize(vmin=np.log10(min(all_ars)), vmax=np.log10(max(all_ars)))

    ncols = 3
    nrows_grid = math.ceil(len(n_vals) / ncols)
    fig, axes = plt.subplots(nrows_grid, ncols, figsize=(4 * ncols, 3.2 * nrows_grid), squeeze=False)

    for idx, n in enumerate(n_vals):
        ax = axes[idx // ncols][idx % ncols]
        n_rows = [r for r in rows if r["N"] == n]
        for ar in all_ars:
            ar_rows = [r for r in n_rows if r["AR"] == ar]
            if not ar_rows:
                continue
            X, Y, Yerr, _ = group_mean_std(ar_rows, "friction", "ent_norm_end")
            if len(X) > 0:
                ax.errorbar(X, Y, yerr=Yerr, fmt="o-", ms=3, lw=1,
                            color=cmap(norm(np.log10(ar))), label=f"AR={ar}")
        ax.set_title(f"N={n}", fontsize=10)
        ax.set_xlabel("Friction μ")
        ax.set_ylabel("Norm. Ent.")
        ax.grid(True, alpha=0.2)

    for idx in range(len(n_vals), nrows_grid * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), label="log₁₀(AR)", shrink=0.6)

    fig.suptitle("Normalized Entanglement vs Friction (thermal init, σ_v=0.1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(out_dir / "ent_norm_vs_friction_by_ar.png", dpi=200)
    plt.close(fig)
    print("Saved ent_norm_vs_friction_by_ar.png")


def plot_ent_vs_ar_by_n(rows, out_dir):
    """Panel plot: ent_norm vs AR, one curve per N, panels by friction."""
    frictions = sorted(set(r["friction"] for r in rows))
    n_vals = sorted(set(r["N"] for r in rows))
    cmap = plt.cm.tab10

    ncols = min(4, len(frictions))
    nrows_grid = math.ceil(len(frictions) / ncols)
    fig, axes = plt.subplots(nrows_grid, ncols, figsize=(4 * ncols, 3.2 * nrows_grid), squeeze=False)

    for idx, fric in enumerate(frictions):
        ax = axes[idx // ncols][idx % ncols]
        fric_rows = [r for r in rows if abs(r["friction"] - fric) < 1e-6]
        for ci, n in enumerate(n_vals):
            n_rows = [r for r in fric_rows if r["N"] == n]
            if not n_rows:
                continue
            X, Y, Yerr, _ = group_mean_std(n_rows, "AR", "ent_norm_end")
            if len(X) > 0:
                ax.errorbar(X, Y, yerr=Yerr, fmt="o-", ms=3, lw=1,
                            color=cmap(ci / max(len(n_vals) - 1, 1)), label=f"N={n}")
        ax.set_xscale("log")
        ax.set_title(f"μ={fric}", fontsize=10)
        ax.set_xlabel("AR")
        ax.set_ylabel("Norm. Ent.")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.2)

    for idx in range(len(frictions), nrows_grid * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Normalized Entanglement vs AR by N (thermal init, σ_v=0.1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "ent_norm_vs_ar_by_n.png", dpi=200)
    plt.close(fig)
    print("Saved ent_norm_vs_ar_by_n.png")


def plot_ke_ratio_vs_ar(rows, out_dir):
    """Panel plot: KE_end/KE0 vs AR, panels by N, colored by friction."""
    n_vals = sorted(set(r["N"] for r in rows))
    frictions = sorted(set(r["friction"] for r in rows))
    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(frictions), vmax=max(frictions))

    ncols = 3
    nrows_grid = math.ceil(len(n_vals) / ncols)
    fig, axes = plt.subplots(nrows_grid, ncols, figsize=(4 * ncols, 3.2 * nrows_grid), squeeze=False)

    for idx, n in enumerate(n_vals):
        ax = axes[idx // ncols][idx % ncols]
        n_rows = [r for r in rows if r["N"] == n]
        for fric in frictions:
            fric_rows = [r for r in n_rows if abs(r["friction"] - fric) < 1e-6]
            # Compute KE ratio
            for r in fric_rows:
                r["ke_ratio"] = r["KE_end"] / r["KE0"] if r["KE0"] > 0 else float("nan")
            X, Y, Yerr, _ = group_mean_std(fric_rows, "AR", "ke_ratio")
            if len(X) > 0:
                ax.errorbar(X, Y, yerr=Yerr, fmt="o-", ms=3, lw=1,
                            color=cmap(norm(fric)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"N={n}", fontsize=10)
        ax.set_xlabel("AR")
        ax.set_ylabel("KE_end / KE_0")
        ax.grid(True, alpha=0.2)

    for idx in range(len(n_vals), nrows_grid * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Friction μ", shrink=0.6)

    fig.suptitle("KE Dissipation Ratio vs AR (thermal init, σ_v=0.1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(out_dir / "ke_ratio_vs_ar.png", dpi=200)
    plt.close(fig)
    print("Saved ke_ratio_vs_ar.png")


def plot_reldisp_vs_ar(rows, out_dir):
    """Panel plot: RMS rel. displacement vs AR, panels by N."""
    n_vals = sorted(set(r["N"] for r in rows))
    frictions = sorted(set(r["friction"] for r in rows))
    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(frictions), vmax=max(frictions))

    ncols = 3
    nrows_grid = math.ceil(len(n_vals) / ncols)
    fig, axes = plt.subplots(nrows_grid, ncols, figsize=(4 * ncols, 3.2 * nrows_grid), squeeze=False)

    for idx, n in enumerate(n_vals):
        ax = axes[idx // ncols][idx % ncols]
        n_rows = [r for r in rows if r["N"] == n]
        for fric in frictions:
            fric_rows = [r for r in n_rows if abs(r["friction"] - fric) < 1e-6]
            X, Y, Yerr, _ = group_mean_std(fric_rows, "AR", "rms_reldisp_end")
            if len(X) > 0:
                ax.errorbar(X, Y, yerr=Yerr, fmt="o-", ms=3, lw=1,
                            color=cmap(norm(fric)))
        ax.set_xscale("log")
        ax.set_title(f"N={n}", fontsize=10)
        ax.set_xlabel("AR")
        ax.set_ylabel("RMS Rel. Disp.")
        ax.grid(True, alpha=0.2)

    for idx in range(len(n_vals), nrows_grid * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), label="Friction μ", shrink=0.6)

    fig.suptitle("RMS Relative Displacement vs AR (thermal init, σ_v=0.1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(out_dir / "reldisp_vs_ar.png", dpi=200)
    plt.close(fig)
    print("Saved reldisp_vs_ar.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=RUNS_ROOT / "thermal_init_analysis",
                    help="Output directory for combined plots")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_all_summaries()
    if not rows:
        raise SystemExit("No data loaded.")

    # Print data summary
    n_vals = sorted(set(r["N"] for r in rows))
    frictions = sorted(set(r["friction"] for r in rows))
    ars = sorted(set(r["AR"] for r in rows))
    print(f"N values: {n_vals}")
    print(f"Frictions: {frictions}")
    print(f"ARs: {ars}")

    plot_ent_vs_ar_by_friction(rows, args.out_dir)
    plot_ent_vs_friction_by_ar(rows, args.out_dir)
    plot_ent_vs_ar_by_n(rows, args.out_dir)
    plot_ke_ratio_vs_ar(rows, args.out_dir)
    plot_reldisp_vs_ar(rows, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
