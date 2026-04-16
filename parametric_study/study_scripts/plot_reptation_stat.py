#!/usr/bin/env python3
"""plot_reptation_stat.py — Compare NSC vs soft contact stat sweeps.

Produces a 3-panel figure per solver (6 panels total, 2 rows):
  Col 1: Sliding path mean ± std vs gap/μ  (one curve per μ)
  Col 2: Normalized collapse  path/(gap/μ) vs gap/μ
  Col 3: CV (std/mean) heatmap over (gap, μ) — variability map
"""

import csv
import os
import sys
import numpy as np

NSC_SUMMARY  = "results/reptation_stat_nsc/summary.csv"
SOFT_SUMMARY = "results/reptation_stat_soft/summary.csv"
OUT_DIR      = "results"


def load_summary(path):
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (float(v) if k != "solver" else v)
                         for k, v in r.items()})
    return rows


def plot_solver(axes_row, rows, title_prefix, cmap_name):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    cmap = plt.colormaps[cmap_name]

    # Separate mu=0 for annotation but exclude from collapse panels
    all_rows   = rows
    rows_nonzero = [r for r in rows if r["mu"] > 1e-12]

    gaps  = np.array([r["gap"]       for r in rows_nonzero])
    mus   = np.array([r["mu"]        for r in rows_nonzero])
    means = np.array([r["path_mean"] for r in rows_nonzero])
    stds  = np.array([r["path_std"]  for r in rows_nonzero])
    n_reps = rows_nonzero[0]["n_reps"] if rows_nonzero else 8
    ratio = gaps / mus

    umus  = np.sort(np.unique(mus))
    ugaps = np.sort(np.unique(gaps))
    colors = cmap(np.linspace(0.15, 0.92, len(umus)))

    ax0, ax1, ax2 = axes_row

    # ── Col 1: path mean ± std vs gap/μ ─────────────────────────────
    for mu_val, col in zip(umus, colors):
        mask  = mus == mu_val
        order = np.argsort(ratio[mask])
        x = ratio[mask][order]
        y = means[mask][order]
        e = stds[mask][order]
        sem = e / np.sqrt(n_reps)
        ax0.plot(x, y, "o-", ms=4, lw=1.4, color=col, label=f"μ={mu_val:.2f}")
        ax0.fill_between(x, y - sem, y + sem, color=col, alpha=0.25)

    ax0.set_xlabel("gap / μ  (rod-lengths)", fontsize=9)
    ax0.set_ylabel("Sliding path length  (rod-lengths)", fontsize=9)
    ax0.set_title(f"{title_prefix}\nmean ± SEM  (N={int(n_reps)} replicates)",
                  fontsize=9)
    ax0.legend(fontsize=6, ncol=2, loc="upper left")
    ax0.set_xlim(left=0)
    ax0.set_ylim(bottom=0)
    ax0.grid(True, lw=0.4, alpha=0.4)

    # ── Col 2: normalized collapse path/(gap/μ) ─────────────────────
    norm_means = means / ratio
    norm_sems  = (stds / np.sqrt(n_reps)) / ratio

    for mu_val, col in zip(umus, colors):
        mask  = mus == mu_val
        order = np.argsort(ratio[mask])
        x = ratio[mask][order]
        y = norm_means[mask][order]
        e = norm_sems[mask][order]
        ax1.plot(x, y, "o-", ms=4, lw=1.4, color=col, label=f"μ={mu_val:.2f}")
        ax1.fill_between(x, y - e, y + e, color=col, alpha=0.25)

    ax1.set_xlabel("gap / μ  (rod-lengths)", fontsize=9)
    ax1.set_ylabel("path / (gap/μ)  [dimensionless]", fontsize=9)
    ax1.set_title(f"{title_prefix}\nCollapse check  (perfect → flat line)",
                  fontsize=9)
    ax1.legend(fontsize=6, ncol=2, loc="upper right")
    ax1.set_xlim(left=0)
    ax1.grid(True, lw=0.4, alpha=0.4)

    # ── Col 3: CV heatmap (std/mean) over (gap, μ) ──────────────────
    # Build grid: rows=μ index, cols=gap index
    cv_grid = np.full((len(umus), len(ugaps)), np.nan)
    for r in rows_nonzero:
        gi = np.searchsorted(ugaps, r["gap"])
        mi = np.searchsorted(umus,  r["mu"])
        cv = r["path_std"] / r["path_mean"] if r["path_mean"] > 1e-12 else np.nan
        cv_grid[mi, gi] = cv

    im = ax2.imshow(cv_grid, origin="lower", aspect="auto",
                    cmap="YlOrRd", vmin=0, vmax=2.0,
                    extent=[ugaps[0], ugaps[-1], umus[0], umus[-1]])
    ax2.set_xlabel("gap  (rod-lengths)", fontsize=9)
    ax2.set_ylabel("μ", fontsize=9)
    ax2.set_title(f"{title_prefix}\nCV = std/mean  (variability map)", fontsize=9)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04,
                 label="CV = σ/μ_path")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nsc_rows  = load_summary(NSC_SUMMARY)
    soft_rows = load_summary(SOFT_SUMMARY)

    if nsc_rows is None and soft_rows is None:
        print("No summary files found. Run run_reptation_stat.py first.",
              file=sys.stderr)
        sys.exit(1)

    datasets = []
    if nsc_rows:
        datasets.append((nsc_rows,  "NSC  (hard impulse)", "tab10"))
    if soft_rows:
        datasets.append((soft_rows, "Soft contact  (penalty + lin-damp)", "tab20"))

    n_solver_rows = len(datasets)
    fig, axes = plt.subplots(n_solver_rows, 3,
                             figsize=(16, 5.2 * n_solver_rows))
    if n_solver_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (rows, label, cmap) in enumerate(datasets):
        plot_solver(axes[row_idx], rows, label, cmap)

    fig.suptitle(
        "Reptation study — statistical sampling over initial velocity directions\n"
        f"Gap × μ grid: 10×10  |  N=8 replicates/point  |  "
        r"|v$_0$|=0.141, |ω$_0$|=0.15 rad/s",
        fontsize=10, y=1.01)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_path = os.path.join(OUT_DIR, "reptation_stat_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {fig_path}")


if __name__ == "__main__":
    main()
