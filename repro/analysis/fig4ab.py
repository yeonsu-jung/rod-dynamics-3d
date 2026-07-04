#!/usr/bin/env python3
"""Fig 4A,B: gap-to-length ratios gbar_t, gbar_r vs N/(Z alpha).

Static analysis from assets/packings_metadata.csv (free-volume columns
filled by repro/fill_free_volume.py). Only finite-A* realizations carry a
meaningful gbar_t (as in the paper); diverging ones are shown as open
markers pinned at the search cap.

Usage: python3 -m repro.analysis.fig4ab
"""
import csv

import numpy as np

from .common import FIGS, PACKINGS, paper_style

Z_REF = 4.0


def main():
    rows = [r for r in csv.DictReader(PACKINGS.open()) if r["gbar_t"]]
    if not rows:
        raise SystemExit("free-volume columns empty; run "
                         "repro/fill_free_volume.py first")
    n = np.array([float(r["N"]) for r in rows])
    alpha = np.array([float(r["alpha_nominal"]) for r in rows])
    gt = np.array([float(r["gbar_t"]) for r in rows])
    gr = np.array([float(r["gbar_r"]) for r in rows])
    finite = np.array([r["astar_finite"] == "1" for r in rows])
    x = n / (Z_REF * alpha)

    plt = paper_style()
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=alpha.min(), vmax=alpha.max())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, y, lab in ((axes[0], gt, r"$\bar g_t$"),
                       (axes[1], gr, r"$\bar g_r$")):
        sc = ax.scatter(x[finite], y[finite], c=alpha[finite], norm=norm,
                        cmap="viridis", s=22, label="finite $A^*$")
        ax.scatter(x[~finite], y[~finite], facecolors="none",
                   edgecolors=cm.viridis(norm(alpha[~finite])), s=22,
                   label="diverging $A^*$ (capped)")
        ax.axvline(1.0 / 3.0, color="r", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N/(Z\alpha)$  $(Z=4)$")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    cb = fig.colorbar(sc, ax=axes, shrink=0.85)
    cb.set_label(r"$\alpha$")
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "fig4ab.pdf")
    fig.savefig(FIGS / "fig4ab.png")
    print(f"wrote {FIGS/'fig4ab.pdf'} ({len(rows)} packings, "
          f"{int(finite.sum())} finite)")


if __name__ == "__main__":
    main()
