#!/usr/bin/env python3
"""End Matter validation: relative contact velocity decays as
v_rel(t) ~ (g0/mu) / t  (Eq. 13 / Eq. 29).

Reads the geomspace-sampled per-contact CSVs from the c3 runs. Because
late-time contacts are sparse single collision events (~1 per sampled
frame), contacts are pooled in log-spaced time bins; the binned mean
|v_rel| is the per-collision relative-velocity scale of the paper.

Gate 5: fitted log-log slope in the decay window ~ -1, and the prefactor
decreases with mu (1/mu ordering on the cohesive packing).

Usage: python3 -m repro.analysis.end_matter
"""
import csv
import sys
from collections import defaultdict

import numpy as np

from .common import FIGS, T_U, is_done, load_manifest, paper_style, run_dir


def load_contacts(run_id):
    """-> arrays (t, |v_rel|), one entry per sampled contact."""
    ts, vs = [], []
    with (run_dir(run_id) / "contacts_sampled.csv").open() as fh:
        for row in csv.DictReader(fh):
            v = (float(row["v_rel_x"]) ** 2 + float(row["v_rel_y"]) ** 2 +
                 float(row["v_rel_z"]) ** 2) ** 0.5
            ts.append(int(row["frame"]) * 1e-3)
            vs.append(v)
    return np.asarray(ts), np.asarray(vs)


def log_binned(t, v, nbins=24, tmin=1e-2):
    keep = t >= tmin
    t, v = t[keep], v[keep]
    if not len(t):
        return np.array([]), np.array([]), np.array([])
    edges = np.logspace(np.log10(t.min()), np.log10(t.max() * 1.001), nbins)
    xs, ys, ns = [], [], []
    for i in range(len(edges) - 1):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if m.sum() >= 3:
            xs.append(np.sqrt(edges[i] * edges[i + 1]))
            ys.append(v[m].mean())
            ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def main():
    runs = load_manifest(["c3"])
    missing = [r["run_id"] for r in runs if not is_done(r["run_id"])]
    if missing:
        print(f"waiting on c3 runs: {missing}")
        sys.exit(1)

    plt = paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    colors = {0.1: "tab:purple", 0.2: "tab:orange", 0.4: "tab:olive"}
    panel = {300: axes[0], 1000: axes[1]}

    slopes = defaultdict(dict)
    prefactors = defaultdict(dict)
    for r in sorted(runs, key=lambda r: float(r["mu"])):
        mu, alpha = float(r["mu"]), int(r["alpha"])
        t, v = load_contacts(r["run_id"])
        xs, ys, ns = log_binned(t, v)
        if not len(xs):
            continue
        ax = panel[alpha]
        ax.loglog(xs, ys, "o-", ms=3, lw=0.8, color=colors[mu],
                  label=f"$\\mu={mu}$")

        win = (xs > 2 * T_U) & (xs < 60 * T_U)
        if win.sum() >= 4:
            p = np.polyfit(np.log(xs[win]), np.log(ys[win]), 1)
            slopes[alpha][mu] = p[0]
            # prefactor: mean of v*t in the window (v ~ C/t -> C)
            prefactors[alpha][mu] = float(np.mean(ys[win] * xs[win]))

    for alpha, ax in panel.items():
        ts = np.logspace(np.log10(T_U), np.log10(100 * T_U), 50)
        ax.loglog(ts, 0.2 * ts ** -1.0, "k--", lw=0.8, label=r"$\propto 1/t$")
        ax.set_xlabel(r"$t$")
        ax.set_title(f"$\\alpha={alpha}$", fontsize=9)
    axes[0].set_ylabel(r"$\langle |v_{rel}| \rangle$ per collision")
    axes[0].legend(frameon=False)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "end_matter.pdf")
    fig.savefig(FIGS / "end_matter.png")
    print(f"wrote {FIGS/'end_matter.pdf'}")

    print("[gate5] log-log slope in [2 t_u, 60 t_u] (expect ~ -1) and "
          "prefactor C = <v t> (expect decreasing with mu):")
    for alpha in sorted(slopes):
        for mu in sorted(slopes[alpha]):
            print(f"  alpha={alpha} mu={mu}: slope={slopes[alpha][mu]:.2f} "
                  f"C={prefactors[alpha][mu]:.4f}")


if __name__ == "__main__":
    main()
