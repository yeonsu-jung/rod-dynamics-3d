#!/usr/bin/env python3
"""Fig 2: normalized contact spread R/l vs N/(Z alpha) with Eq. 3/4 theory.

Static analysis — uses assets/packings_metadata.csv only (Z fixed to the
paper's reference value 4 for the x-axis, as in the figure caption).

Usage: python3 -m repro.analysis.fig2
"""
import csv

import numpy as np

from .common import FIGS, PACKINGS, paper_style

Z_REF = 4.0


def main():
    rows = list(csv.DictReader(PACKINGS.open()))
    n = np.array([float(r["N"]) for r in rows])
    alpha = np.array([float(r["alpha_nominal"]) for r in rows])
    rl = np.array([float(r["R_over_l"]) for r in rows])
    x = n / (Z_REF * alpha)

    plt = paper_style()
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    hi = rl > 0.5
    ax.loglog(x[hi], rl[hi], "o", ms=4, color="tab:blue",
              label=r"$R/l > 0.5$")
    ax.loglog(x[~hi], rl[~hi], "o", ms=4, color="tab:green",
              label=r"$R/l \leq 0.5$")

    xs = np.logspace(np.log10(x.min()) - 0.3, np.log10(x.max()) + 0.3, 100)
    ax.loglog(xs, (1.5 * xs) ** (1.0 / 3.0), "r-", lw=1.2,
              label=r"$(\frac{3}{2}\frac{N}{Z\alpha})^{1/3}$ (Eq. 3)")
    ax.loglog(xs, 1.5 * xs, "r--", lw=1.2,
              label=r"$\frac{3}{2}\frac{N}{Z\alpha}$ (Eq. 4)")
    ax.axvline(1.0 / 3.0, color="k", ls=":", lw=0.8,
               label=r"$N/(Z\alpha)=1/3$")

    ax.set_xlabel(r"$N/(Z\alpha)$  $(Z=4)$")
    ax.set_ylabel(r"$R/l$")
    ax.legend(frameon=False, loc="upper left")
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "fig2.pdf")
    fig.savefig(FIGS / "fig2.png")
    print(f"wrote {FIGS/'fig2.pdf'} ({len(rows)} packings)")


if __name__ == "__main__":
    main()
