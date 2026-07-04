#!/usr/bin/env python3
"""Fig S4: collision count per time step vs t/t_u, mu x aspect ratio.

Also runs Gate 3 (t_u sanity: total-entanglement halving time for the
alpha=100, mu=0.2 run should be O(t_u)=O(3.2)) and prints the Gate 4
qualitative checks (early collision drop; late-time rise for mu>=0.3 at
high alpha).

Usage: python3 -m repro.analysis.figS4   (from repo root)
"""
import sys

import numpy as np

from .common import (T_U, FIGS, ent_series, halving_time, is_done,
                     load_manifest, load_profile, paper_style, rolling_mean)


def main():
    runs = [r for r in load_manifest(["c2"])]
    missing = [r["run_id"] for r in runs if not is_done(r["run_id"])]
    if missing:
        print(f"waiting on {len(missing)} c2 runs, e.g. {missing[0]}")
        sys.exit(1)

    plt = paper_style()
    fig, ax = plt.subplots(figsize=(4.2, 2.9))

    colors = {0.2: "tab:blue", 0.3: "tab:red", 0.4: "tab:olive"}
    styles = {100: ":", 200: "--", 1000: "-"}
    gate4 = {}

    for r in sorted(runs, key=lambda r: (float(r["mu"]), int(r["alpha"]))):
        mu, alpha = float(r["mu"]), int(r["alpha"])
        prof = load_profile(r["run_id"], ["collisions"])
        t = prof["frame"] * 1e-3 / T_U
        col = prof["collisions"]
        # paper plots per-step counts; smooth lightly for readability
        smooth = rolling_mean(col, 320)  # ~0.1 t_u window at stride 1
        keep = t <= 32
        ax.plot(t[keep], smooth[keep], lw=0.8, color=colors[mu],
                ls=styles[alpha], label=f"mu={mu}, AR={alpha}")

        early = col[(t > 0) & (t < 2)].mean()
        late = col[(t > 20) & (t < 32)].mean()
        gate4[(mu, alpha)] = (early, late)

    ax.set_xlabel(r"$t/t_u$")
    ax.set_ylabel("collision count")
    ax.set_xlim(0, 32)
    ax.legend(ncol=2, frameon=False)
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "figS4.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"wrote {out}")

    # ── Gate 3: t_u sanity on alpha=100, mu=0.2 ──
    r = next(r for r in runs
             if float(r["mu"]) == 0.2 and int(r["alpha"]) == 100)
    prof = load_profile(r["run_id"], ["ent_sum", "ent_pairs"])
    t, ebar = ent_series(prof, r["ent_period"])
    t_half = halving_time(t, ebar)
    print(f"[gate3] ebar(0)={ebar[0]:.3f}; total-entanglement halving time "
          f"= {t_half:.2f} (t_u={T_U}; expect same order)")

    # ── Gate 4: qualitative S4 signature ──
    print("[gate4] mean collisions early (t<2 t_u) vs late (20-32 t_u):")
    for (mu, alpha), (early, late) in sorted(gate4.items()):
        print(f"  mu={mu} AR={alpha:4d}: early={early:7.1f} late={late:7.1f}")
    drop = all(late < early for (early, late) in
               [gate4[(0.2, a)] for a in (100, 200)])
    rise = gate4[(0.4, 1000)][1] > 5 * max(1e-9, gate4[(0.2, 100)][1])
    print(f"  early-drop (mu=0.2): {'ok' if drop else 'NO'}; "
          f"late-rise (mu=0.4, AR=1000 vs mu=0.2, AR=100): "
          f"{'ok' if rise else 'NO'}")


if __name__ == "__main__":
    main()
