#!/usr/bin/env python3
"""run_reptation_soft.py — Reptation sweep using soft-contact (penalty) solver.

Setup:
  - Single rod: L=1.0, d=0.01 (radius=0.005)
  - v_axial = v_transverse = 0.1, no rotation
  - gap = tube_radius - rod_radius  ∈ linspace(0.001, 0.1, 10)
  - mu ∈ linspace(0, 1, 10)
  - 100 total runs
  - Wall contact: penalty spring k_scaler=10000 + smooth Coulomb friction
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# ── Parameters ──────────────────────────────────────────────────────
EXE        = "./build-headless/rigidbody_viewer_3d"
BASE_SCENE = "assets/scenes/reptation_soft.json"
OUT_DIR    = "results/reptation_soft"
COMBINED_CSV = os.path.join(OUT_DIR, "combined.csv")

ROD_RADIUS = 0.005   # diameter 0.01
V0         = 0.1     # rod-lengths/s

GAPS = np.linspace(0.001, 0.1, 10)
MUS  = np.linspace(0.0,   1.0, 10)

MAX_STEPS      = 100_000
STOP_KE        = 1e-8
KE_AVG_WIN     = 5
STOP_SLIDE_VEL = 1e-6


def make_scene(base, cyl_radius, mu):
    """Return modified scene dict with given cylinder radius and soft-contact mu."""
    s = json.loads(json.dumps(base))  # deep copy
    s["scene"]["cylinder"]["radius"] = float(cyl_radius)
    s["physics"]["soft_contact"]["mu"] = float(mu)
    return s


def parse_summary(path):
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {k: float(v) for k, v in row.items()}
    return {}


def run_one(exe, scene_dict, gap, mu, out_dir, verbose=False):
    tag = f"gap{gap:.4f}_mu{mu:.4f}"
    summary_path = os.path.join(out_dir, f"rept_soft_{tag}.csv")

    # Delete stale CSV (C++ appends to the file)
    if os.path.isfile(summary_path):
        os.unlink(summary_path)

    fd, tmp_scene = tempfile.mkstemp(suffix=".json", prefix="rept_soft_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(scene_dict, f)

        cmd = [
            exe,
            "--headless", "--steps", str(MAX_STEPS),
            "--scene", tmp_scene,
            "--set-velocity", "0", f"{V0}", f"{V0}", "0",
            "--stop-ke-threshold", f"{STOP_KE}",
            "--stop-ke-avg-window", str(KE_AVG_WIN),
            "--stop-slide-vel-threshold", f"{STOP_SLIDE_VEL}",
            "--stop-slide-vel-min-steps", "50",
            "--test-rod-id", "0",
            "--reptation-summary", summary_path,
        ]
        if not verbose:
            cmd.append("--quiet")
        if verbose:
            print("  CMD:", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if verbose:
            if result.stdout:
                print("  STDOUT:", result.stdout[:500])
            if result.stderr:
                print("  STDERR:", result.stderr[:500])
        if result.returncode != 0:
            print(f"  ⚠ exit={result.returncode}: {result.stderr[:200]}",
                  file=sys.stderr)
    finally:
        os.unlink(tmp_scene)

    return summary_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Single mid-range case for quick validation")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(BASE_SCENE) as f:
        base = json.load(f)

    if args.smoke:
        gap = 0.01
        mu  = 0.3
        cyl_r = gap + ROD_RADIUS
        print(f"=== SMOKE TEST (soft) ===  gap={gap}  mu={mu}  R_cyl={cyl_r}")
        scene = make_scene(base, cyl_r, mu)
        summary_path = run_one(EXE, scene, gap, mu, OUT_DIR, verbose=True)
        parsed = parse_summary(summary_path)
        print(f"\n  Summary CSV: {summary_path}")
        for k, v in parsed.items():
            print(f"    {k:25s} = {v}")
        return

    total    = len(GAPS) * len(MUS)
    results  = []
    run_idx  = 0

    for gap in GAPS:
        cyl_r = gap + ROD_RADIUS
        for mu in MUS:
            run_idx += 1
            print(f"[{run_idx:3d}/{total}]  gap={gap:.4f}  mu={mu:.4f}  "
                  f"R_cyl={cyl_r:.4f}")

            scene = make_scene(base, cyl_r, mu)
            summary_path = run_one(EXE, scene, gap, mu, OUT_DIR)

            row = {"gap": gap, "mu": mu, "cyl_r": cyl_r}
            parsed = parse_summary(summary_path)
            row.update(parsed)
            results.append(row)

            sl = row.get("total_path_length", "?")
            wh = row.get("wall_hits", "?")
            print(f"       → path_length={sl}  wall_hits={wh}")

    # Write combined CSV
    with open(COMBINED_CSV, "w") as f:
        cols = ["gap", "mu", "cyl_r", "net_displacement", "total_path_length",
                "wall_hits", "sim_time", "final_KE"]
        f.write(",".join(cols) + "\n")
        for r in results:
            vals = [str(r.get(c, "")) for c in cols]
            f.write(",".join(vals) + "\n")

    print(f"\n✓ Combined CSV: {COMBINED_CSV}  ({len(results)} rows)")


if __name__ == "__main__":
    main()
