#!/usr/bin/env python3
"""run_reptation_stat.py — Reptation parameter sweep with statistical sampling.

Randomises initial translational velocity direction (fixed speed |v|=V0_MAG)
and initial angular velocity direction (fixed |ω|=W0_MAG) over N_REPS replicates
per (gap, mu) point.  Runs for both NSC (hard impulse) and soft-contact (penalty)
solvers; use --solver nsc|soft to select.

Output
------
results/reptation_stat_{solver}/runs/   — one CSV per replicate
results/reptation_stat_{solver}/combined.csv — all replicates, one row each
results/reptation_stat_{solver}/summary.csv  — mean ± std per (gap, mu)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# ── Parameters ──────────────────────────────────────────────────────
EXE = "./build-headless/rigidbody_viewer_3d"

ROD_RADIUS = 0.005          # diameter 0.01, rod length 1.0
V0_MAG     = 0.1 * np.sqrt(2)   # same KE as canonical (0.1, 0.1, 0)
W0_MAG     = 0.15           # rad/s — comparable tip-velocity to V0

GAPS = np.linspace(0.001, 0.1, 10)
MUS  = np.linspace(0.0,   1.0, 10)

MAX_STEPS      = 100_000    # 100 s at dt=0.001
STOP_KE        = 1e-8
KE_AVG_WIN     = 5
STOP_SLIDE_VEL = 1e-6
STOP_SLIDE_MIN_STEPS = 50

SCENE_NSC  = "assets/scenes/reptation.json"
SCENE_SOFT = "assets/scenes/reptation_soft.json"


def random_unit_vector(rng):
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def make_scene_nsc(base, cyl_radius, mu):
    s = json.loads(json.dumps(base))
    s["scene"]["cylinder"]["radius"] = float(cyl_radius)
    # mu will be set via --nsc-mu CLI flag; keep JSON as-is
    return s


def make_scene_soft(base, cyl_radius, mu):
    s = json.loads(json.dumps(base))
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


def run_one(solver, exe, scene_dict, gap, mu, rep, v0, w0, out_runs_dir,
            verbose=False):
    tag = f"gap{gap:.4f}_mu{mu:.4f}_rep{rep:02d}"
    summary_path = os.path.join(out_runs_dir, f"rept_{tag}.csv")
    if os.path.isfile(summary_path):
        os.unlink(summary_path)

    fd, tmp_scene = tempfile.mkstemp(suffix=".json", prefix="rept_stat_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(scene_dict, f)

        vx, vy, vz = v0
        wx, wy, wz = w0

        cmd = [
            exe,
            "--headless", "--steps", str(MAX_STEPS),
            "--scene", tmp_scene,
            "--set-velocity", "0", f"{vx:.8f}", f"{vy:.8f}", f"{vz:.8f}",
            "--set-ang-velocity", "0", f"{wx:.8f}", f"{wy:.8f}", f"{wz:.8f}",
            "--stop-ke-threshold", f"{STOP_KE}",
            "--stop-ke-avg-window", str(KE_AVG_WIN),
            "--stop-slide-vel-threshold", f"{STOP_SLIDE_VEL}",
            "--stop-slide-vel-min-steps", str(STOP_SLIDE_MIN_STEPS),
            "--test-rod-id", "0",
            "--reptation-summary", summary_path,
        ]
        if solver == "nsc":
            cmd += ["--nsc", "--nsc-mu", f"{mu:.8f}"]
        if not verbose:
            cmd.append("--quiet")
        if verbose:
            print("  CMD:", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=300)
        if verbose:
            if result.stdout:
                print("  STDOUT:", result.stdout[:400])
            if result.stderr:
                print("  STDERR:", result.stderr[:400])
        if result.returncode != 0:
            print(f"  ⚠ exit={result.returncode}: {result.stderr[:200]}",
                  file=sys.stderr)
    finally:
        os.unlink(tmp_scene)

    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["nsc", "soft"], required=True,
                        help="Contact solver: nsc or soft")
    parser.add_argument("--reps", type=int, default=8,
                        help="Replicates per (gap, mu) point")
    parser.add_argument("--smoke", action="store_true",
                        help="Single case (gap=0.01, mu=0.3) with --reps replicates")
    args = parser.parse_args()

    N_REPS  = args.reps
    solver  = args.solver
    OUT_DIR = f"results/reptation_stat_{solver}"
    RUNS_DIR = os.path.join(OUT_DIR, "runs")
    COMBINED_CSV = os.path.join(OUT_DIR, "combined.csv")
    SUMMARY_CSV  = os.path.join(OUT_DIR, "summary.csv")

    os.makedirs(RUNS_DIR, exist_ok=True)

    scene_file = SCENE_NSC if solver == "nsc" else SCENE_SOFT
    with open(scene_file) as f:
        base = json.load(f)

    make_scene = make_scene_nsc if solver == "nsc" else make_scene_soft

    gaps = np.array([0.01]) if args.smoke else GAPS
    mus  = np.array([0.3])  if args.smoke else MUS
    total = len(gaps) * len(mus) * N_REPS

    all_rows = []
    run_idx  = 0

    for gap in gaps:
        cyl_r = gap + ROD_RADIUS
        for mu in mus:
            scene = make_scene(base, cyl_r, mu)

            rep_paths = []
            for rep in range(N_REPS):
                run_idx += 1
                rng  = np.random.default_rng(seed=rep)
                v0   = V0_MAG * random_unit_vector(rng)
                w0   = W0_MAG * random_unit_vector(rng)

                if not args.smoke or rep == 0:
                    print(f"[{run_idx:4d}/{total}]  solver={solver}  "
                          f"gap={gap:.4f}  mu={mu:.4f}  rep={rep}  "
                          f"v=({v0[0]:.3f},{v0[1]:.3f},{v0[2]:.3f})")

                verbose = args.smoke and rep == 0
                summary_path = run_one(
                    solver, EXE, scene, gap, mu, rep, v0, w0,
                    RUNS_DIR, verbose=verbose)

                parsed = parse_summary(summary_path)
                row = {
                    "solver": solver,
                    "gap": gap, "mu": mu, "cyl_r": cyl_r,
                    "rep": rep,
                    "vx0": v0[0], "vy0": v0[1], "vz0": v0[2],
                    "wx0": w0[0], "wy0": w0[1], "wz0": w0[2],
                }
                row.update(parsed)
                all_rows.append(row)
                rep_paths.append(row.get("total_path_length", float("nan")))

            paths = [p for p in rep_paths if not np.isnan(p)]
            print(f"       → path  mean={np.mean(paths):.4f}  "
                  f"std={np.std(paths):.4f}  n={len(paths)}")

    # ── Write combined CSV ──────────────────────────────────────────────
    cols = ["solver", "gap", "mu", "cyl_r", "rep",
            "vx0", "vy0", "vz0", "wx0", "wy0", "wz0",
            "net_displacement", "total_path_length",
            "wall_hits", "sim_time", "final_KE"]
    with open(COMBINED_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    # ── Write summary CSV ───────────────────────────────────────────────
    from itertools import groupby
    summary_rows = []
    key_fn = lambda r: (r["gap"], r["mu"])
    for (gap, mu), grp in groupby(
            sorted(all_rows, key=lambda r: (r["gap"], r["mu"])), key_fn):
        grp = list(grp)
        paths = [r.get("total_path_length", float("nan")) for r in grp]
        paths = [p for p in paths if not np.isnan(p)]
        if not paths:
            continue
        summary_rows.append({
            "solver": solver,
            "gap": gap, "mu": mu, "cyl_r": gap + ROD_RADIUS,
            "n_reps": len(paths),
            "path_mean": np.mean(paths),
            "path_std":  np.std(paths),
            "path_min":  np.min(paths),
            "path_max":  np.max(paths),
        })

    s_cols = ["solver", "gap", "mu", "cyl_r", "n_reps",
              "path_mean", "path_std", "path_min", "path_max"]
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=s_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n✓ Combined CSV ({len(all_rows)} rows): {COMBINED_CSV}")
    print(f"✓ Summary CSV  ({len(summary_rows)} rows): {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
