#!/usr/bin/env python3
"""Fill gbar_t / gbar_r / astar columns of assets/packings_metadata.csv by
running the rod-free-volume tool (github.com/yeonsu-jung/rod-free-volume,
SI Table S2 defaults: 360 angular samples, 16 bisection steps).

Definitions (paper Eq. 8-9):
  A*  = max_k A_k   (free translation area),   gbar_t = sqrt(A*) / l
  W*  = max_k Omega_k (free solid angle),      gbar_r = sqrt(W* / (2 pi))

The tool caps the translational search at one rod length, so a diverging A*
(escapable boundary rod) appears as a large capped area. We flag
astar_finite = 0 when gbar_t >= DIVERGENCE_GBAR_T (clear bimodal separation:
self-caged packings sit at gbar_t ~ 0.01, capped-escape ones at >~ 0.2).

Usage: python3 repro/fill_free_volume.py --binary <path/to/rod_free_volume>
"""
import argparse
import csv
import math
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
META = REPO / "assets" / "packings_metadata.csv"
DIVERGENCE_GBAR_T = 0.2


def measure(binary, packing_path, threads):
    out = subprocess.run(
        [str(binary), "--threads", str(threads), str(packing_path)],
        capture_output=True, text=True, check=True, cwd=REPO)
    a_star = w_star = 0.0
    for line in out.stdout.splitlines():
        if line.startswith("#") or line.startswith("rod_index"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        a_star = max(a_star, float(parts[1]))
        w_star = max(w_star, float(parts[2]))
    return a_star, w_star


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    rows = list(csv.DictReader(META.open()))
    for r in rows:
        a_star, w_star = measure(args.binary, REPO / r["path"], args.threads)
        l = float(r["L_mean"])
        gbar_t = math.sqrt(a_star) / l
        gbar_r = math.sqrt(w_star / (2 * math.pi))
        r["gbar_t"] = round(gbar_t, 6)
        r["gbar_r"] = round(gbar_r, 6)
        r["astar_finite"] = 0 if gbar_t >= DIVERGENCE_GBAR_T else 1
        print(f"{r['packing_id']}: gbar_t={gbar_t:.4f} gbar_r={gbar_r:.4f} "
              f"finite={r['astar_finite']}")

    with META.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"updated {META}")


if __name__ == "__main__":
    main()
