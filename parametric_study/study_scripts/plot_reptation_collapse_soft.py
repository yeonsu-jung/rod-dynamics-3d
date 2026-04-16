#!/usr/bin/env python3
"""Plot sliding length vs gap/mu (data collapse) for soft-contact reptation sweep."""

import csv
import os
import sys
import numpy as np

DATA_CSV = "results/reptation_soft/combined.csv"
OUT_DIR  = "results/reptation_soft"


def main():
    if not os.path.isfile(DATA_CSV):
        print(f"Missing {DATA_CSV}. Run run_reptation_soft.py first.", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(DATA_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) for k, v in r.items()})

    # Filter out mu = 0
    rows = [r for r in rows if r["mu"] > 1e-12]

    gaps  = np.array([r["gap"]               for r in rows])
    mus   = np.array([r["mu"]                for r in rows])
    path  = np.array([r["total_path_length"] for r in rows])
    ratio = gaps / mus  # gap/mu

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Soft-contact reptation sweep", fontsize=13)

    umus = np.sort(np.unique(mus))

    # ── (a) Sliding length vs gap/mu ─────────────────────────────
    ax = axes[0]
    for mu_val in umus:
        mask  = mus == mu_val
        order = np.argsort(ratio[mask])
        ax.plot(ratio[mask][order], path[mask][order],
                "o-", ms=5, label=f"μ={mu_val:.2f}")
    ax.set_xlabel("gap / μ")
    ax.set_ylabel("Sliding length (rod-lengths)")
    ax.set_title("Sliding length vs gap/μ  (soft contact)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # ── (b) Normalized: sliding_length / (gap/mu) vs gap/mu ──────
    ax = axes[1]
    norm_path = path / ratio
    for mu_val in umus:
        mask  = mus == mu_val
        order = np.argsort(ratio[mask])
        ax.plot(ratio[mask][order], norm_path[mask][order],
                "o-", ms=5, label=f"μ={mu_val:.2f}")
    ax.set_xlabel("gap / μ")
    ax.set_ylabel("Sliding length / (gap/μ)")
    ax.set_title("Normalized (collapse check)  (soft contact)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(left=0)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_path = os.path.join(OUT_DIR, "reptation_collapse_soft.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"✓ Plot saved: {fig_path}")


if __name__ == "__main__":
    main()
