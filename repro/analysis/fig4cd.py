#!/usr/bin/env python3
"""Fig 4C,D (+ S3): entanglement retention ebar(t_f)/ebar(0) vs gap-to-
length ratios, binned, colored by mu.

Uses every finished c1 run: retention from the profile-CSV entanglement
series, gbar_t/gbar_r/astar_finite from assets/packings_metadata.csv.

  fig4cd.pdf : binned retention vs gbar_t (C) and gbar_r (D), marker size
               ~ sqrt(N_bin), shaded band = standard error (paper style)
  figS3.pdf  : retention histograms split by diverging/finite A*, by mu

Usage: python3 -m repro.analysis.fig4cd  [--min-runs N]
"""
import argparse
import sys
from collections import defaultdict

import numpy as np

from .common import (FIGS, ent_series, is_done, load_manifest,
                     load_packings, load_profile, paper_style)

MUS = [0.0, 0.1, 0.2, 0.4]
MU_COLORS = {0.0: "#3b0f70", 0.1: "#8c2981", 0.2: "#de4968", 0.4: "#fe9f6d"}


def collect():
    packs = load_packings()
    recs = []
    runs = load_manifest(["c1"])
    done = [r for r in runs if is_done(r["run_id"])]
    for r in done:
        prof = load_profile(r["run_id"], ["ent_sum", "ent_pairs"])
        t, ebar = ent_series(prof, r["ent_period"])
        if len(ebar) < 2 or not np.isfinite(ebar[0]) or ebar[0] <= 0:
            continue
        p = packs[r["packing_id"]]
        recs.append({
            "mu": float(r["mu"]), "retention": float(ebar[-1] / ebar[0]),
            "gbar_t": float(p["gbar_t"]) if p["gbar_t"] else np.nan,
            "gbar_r": float(p["gbar_r"]) if p["gbar_r"] else np.nan,
            "finite": p["astar_finite"] == "1",
            "alpha": float(p["alpha_nominal"]), "N": int(p["N"]),
        })
    return recs, len(runs)


def binned_panel(ax, recs, key):
    vals = np.array([r[key] for r in recs])
    ok = np.isfinite(vals) & (vals > 0)
    lo, hi = vals[ok].min(), vals[ok].max()
    edges = np.logspace(np.log10(lo * 0.99), np.log10(hi * 1.01), 9)
    for mu in MUS:
        xs, ys, es, ns = [], [], [], []
        for i in range(len(edges) - 1):
            sel = [r for r in recs
                   if r["mu"] == mu and np.isfinite(r[key])
                   and edges[i] <= r[key] < edges[i + 1]]
            if not sel:
                continue
            rr = np.array([r["retention"] for r in sel])
            xs.append(np.sqrt(edges[i] * edges[i + 1]))
            ys.append(rr.mean())
            es.append(rr.std(ddof=1) / np.sqrt(len(rr)) if len(rr) > 1
                      else 0.0)
            ns.append(len(rr))
        xs, ys, es, ns = map(np.array, (xs, ys, es, ns))
        if not len(xs):
            continue
        ax.fill_between(xs, ys - es, ys + es, color=MU_COLORS[mu],
                        alpha=0.25, lw=0)
        ax.plot(xs, ys, "-", color=MU_COLORS[mu], lw=1)
        ax.scatter(xs, ys, s=8 * np.sqrt(ns), color=MU_COLORS[mu],
                   label=f"$\\mu={mu}$", zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-runs", type=int, default=1)
    args = ap.parse_args()

    recs, total = collect()
    print(f"{len(recs)} finished c1 runs of {total}")
    if len(recs) < args.min_runs:
        sys.exit(1)

    plt = paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    binned_panel(axes[0], recs, "gbar_t")
    binned_panel(axes[1], recs, "gbar_r")
    axes[0].set_xlabel(r"$\bar g_t$")
    axes[1].set_xlabel(r"$\bar g_r$")
    axes[0].set_ylabel(r"$\bar e(t_f)/\bar e(0)$")
    axes[0].legend(frameon=False)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "fig4cd.pdf")
    fig.savefig(FIGS / "fig4cd.png")
    print(f"wrote {FIGS/'fig4cd.pdf'}")

    # ── Fig S3: histograms by diverging/finite A* ──
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)
    bins = np.linspace(0, 1, 21)
    for ax, fin, title in ((axes[0], False, "Diverging $A^*$"),
                           (axes[1], True, "Finite $A^*$")):
        stack = [[r["retention"] for r in recs
                  if r["finite"] == fin and r["mu"] == mu] for mu in MUS]
        ax.hist(stack, bins=bins, stacked=True,
                color=[MU_COLORS[m] for m in MUS],
                label=[f"$\\mu={m}$" for m in MUS])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$\bar e(t_f)/\bar e(0)$")
    axes[0].set_ylabel("count")
    axes[1].legend(frameon=False)
    fig.savefig(FIGS / "figS3.pdf")
    fig.savefig(FIGS / "figS3.png")
    print(f"wrote {FIGS/'figS3.pdf'}")


if __name__ == "__main__":
    main()
