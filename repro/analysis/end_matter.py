#!/usr/bin/env python3
"""End Matter validation: relative contact velocity decays as
v_rel(t) ~ (g0/mu) / t  (Eq. 13 / Eq. 29).

Reads the geomspace-sampled per-contact CSVs from the c3 runs
(runs/<id>/contacts_sampled.csv with per-contact v_n, v_t at sampled
frames), plots mean |v_rel| over contacts vs t (log-log) with a 1/t
guide, plus the mu-rescaled collapse (mu * v_rel vs t).

Gate 5: fitted log-log slope in the decay window should be ~ -1 and the
prefactor should decrease with mu.

Usage: python3 -m repro.analysis.end_matter
"""
import csv
import sys
from collections import defaultdict

import numpy as np

from .common import FIGS, T_U, is_done, load_manifest, paper_style, run_dir


def load_contact_series(run_id):
    """-> t (s), mean |v_rel| per sampled frame, contact count."""
    by_frame = defaultdict(list)
    with (run_dir(run_id) / "contacts_sampled.csv").open() as fh:
        for row in csv.DictReader(fh):
            v = np.array([float(row["v_rel_x"]), float(row["v_rel_y"]),
                          float(row["v_rel_z"])])
            by_frame[int(row["frame"])].append(np.linalg.norm(v))
    frames = np.array(sorted(by_frame))
    vmean = np.array([np.mean(by_frame[f]) for f in frames])
    counts = np.array([len(by_frame[f]) for f in frames])
    return frames * 1e-3, vmean, counts


def main():
    runs = load_manifest(["c3"])
    missing = [r["run_id"] for r in runs if not is_done(r["run_id"])]
    if missing:
        print(f"waiting on c3 runs: {missing}")
        sys.exit(1)

    plt = paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    colors = {0.1: "tab:purple", 0.2: "tab:orange", 0.4: "tab:olive"}

    slopes = {}
    for r in sorted(runs, key=lambda r: float(r["mu"])):
        mu = float(r["mu"])
        t, v, cnt = load_contact_series(r["run_id"])
        keep = cnt >= 5  # need enough contacts for a mean
        axes[0].loglog(t[keep], v[keep], ".", ms=3, color=colors[mu],
                       label=f"$\\mu={mu}$")
        axes[1].loglog(t[keep], mu * v[keep], ".", ms=3, color=colors[mu])

        # slope fit in the algebraic window: t in [2 t_u, 30 t_u]
        win = keep & (t > 2 * T_U) & (t < 30 * T_U) & (v > 0)
        if win.sum() >= 10:
            p = np.polyfit(np.log(t[win]), np.log(v[win]), 1)
            slopes[mu] = p[0]

    for ax in axes:
        ts = np.logspace(np.log10(2 * T_U), np.log10(100 * T_U), 50)
        ax.loglog(ts, 0.3 * ts ** -1.0, "k--", lw=0.8, label=r"$\propto 1/t$")
        ax.set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$\langle |v_{rel}| \rangle_{contacts}$")
    axes[1].set_ylabel(r"$\mu \, \langle |v_{rel}| \rangle$ (collapse)")
    axes[0].legend(frameon=False)
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "end_matter.pdf")
    fig.savefig(FIGS / "end_matter.png")
    print(f"wrote {FIGS/'end_matter.pdf'}")

    print("[gate5] log-log slopes in [2 t_u, 30 t_u] (expect ~ -1):")
    for mu, s in sorted(slopes.items()):
        print(f"  mu={mu}: slope={s:.2f}")


if __name__ == "__main__":
    main()
