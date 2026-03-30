#!/usr/bin/env python3
"""run_reptation_minimal.py — Minimal reptation sweep: sliding length vs gap & mu.

Setup:
  - Single rod: L=1.0, d=0.01 (radius=0.005)
  - v_axial = v_transverse = 0.1 (rod-lengths/s), no rotation
  - gap = tube_radius - rod_radius  ∈ linspace(0.001, 0.1, 10)
  - mu ∈ linspace(0, 1, 10)
  - 100 total runs
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# ── Parameters ──────────────────────────────────────────────────────
EXE = "./build-headless/rigidbody_viewer_3d"
BASE_SCENE = "assets/scenes/reptation.json"
OUT_DIR = "results/reptation_minimal"
COMBINED_CSV = os.path.join(OUT_DIR, "combined.csv")

ROD_RADIUS = 0.005   # diameter 0.01
V0 = 0.1             # 0.1 rod-lengths/s  (rod length = 1.0)

GAPS = np.linspace(0.001, 0.1, 10)
MUS  = np.linspace(0.0,   1.0, 10)

MAX_STEPS   = 100_000      # 100 s sim-time at dt=0.001
STOP_KE     = 1e-8
KE_AVG_WIN  = 5
STOP_SLIDE_VEL = 1e-6     # stop when |v·axis| < this


def make_scene(base, cyl_radius, mu):
    """Return modified scene dict with given cylinder radius and friction."""
    s = json.loads(json.dumps(base))  # deep copy
    s["scene"]["cylinder"]["radius"] = float(cyl_radius)
    s["physics"]["nsc"]["mu"] = float(mu)
    return s


def parse_summary(path):
    """Parse a 1-row reptation summary CSV into a dict."""
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {k: float(v) for k, v in row.items()}
    return {}


def run_one(exe, scene_dict, gap, mu, out_dir, verbose=False):
    """Write temp scene, run sim, return summary CSV path."""
    tag = f"gap{gap:.4f}_mu{mu:.4f}"
    summary_path = os.path.join(out_dir, f"rept_{tag}.csv")

    # Delete stale CSV (C++ appends to the file)
    if os.path.isfile(summary_path):
        os.unlink(summary_path)

    # Write temp scene JSON
    fd, tmp_scene = tempfile.mkstemp(suffix=".json", prefix="rept_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(scene_dict, f)

        cmd = [
            exe,
            "--headless", "--steps", str(MAX_STEPS),
            "--scene", tmp_scene,
            "--nsc",
            "--nsc-mu", f"{mu:.8f}",
            "--set-velocity", "0", f"{V0}", f"{V0}", "0",   # vx=V0 (transverse), vy=V0 (axial)
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
            print(f"  ⚠ exit={result.returncode}: {result.stderr[:200]}", file=sys.stderr)
    finally:
        os.unlink(tmp_scene)

    return summary_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run a single case (mid-range gap & mu) for quick validation")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load base scene
    with open(BASE_SCENE) as f:
        base = json.load(f)

    if args.smoke:
        # Single mid-range test
        gap = 0.01
        mu = 0.3
        cyl_r = gap + ROD_RADIUS
        print(f"=== SMOKE TEST ===  gap={gap}  mu={mu}  R_cyl={cyl_r}")
        scene = make_scene(base, cyl_r, mu)
        summary_path = run_one(EXE, scene, gap, mu, OUT_DIR, verbose=True)
        parsed = parse_summary(summary_path)
        print(f"\n  Summary CSV: {summary_path}")
        for k, v in parsed.items():
            print(f"    {k:25s} = {v}")
        return

    total = len(GAPS) * len(MUS)
    results = []   # list of dicts
    run_idx = 0

    for gap in GAPS:
        cyl_r = gap + ROD_RADIUS
        for mu in MUS:
            run_idx += 1
            print(f"[{run_idx:3d}/{total}]  gap={gap:.4f}  mu={mu:.4f}  R_cyl={cyl_r:.4f}")

            scene = make_scene(base, cyl_r, mu)
            summary_path = run_one(EXE, scene, gap, mu, OUT_DIR)

            # Parse the 1-row summary
            row = {"gap": gap, "mu": mu, "cyl_r": cyl_r}
            parsed = parse_summary(summary_path)
            row.update(parsed)

            results.append(row)
            sl = row.get("total_path_length", "?")
            wh = row.get("wall_hits", "?")
            print(f"       → path_length={sl}  wall_hits={wh}")

    # ── Write combined CSV ──────────────────────────────────────────
    with open(COMBINED_CSV, "w") as f:
        cols = ["gap", "mu", "cyl_r", "net_displacement", "total_path_length",
                "wall_hits", "sim_time", "final_KE"]
        f.write(",".join(cols) + "\n")
        for r in results:
            vals = [str(r.get(c, "")) for c in cols]
            f.write(",".join(vals) + "\n")
    print(f"\n✓ Combined CSV: {COMBINED_CSV}  ({len(results)} rows)")

    # ── Quick plots ─────────────────────────────────────────────────
    try:
        plot_results(results)
    except Exception as e:
        print(f"Plotting failed ({e}); CSV is still available.", file=sys.stderr)


def plot_results(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gaps = np.array([r["gap"] for r in results])
    mus  = np.array([r["mu"]  for r in results])
    path = np.array([r.get("total_path_length", np.nan) for r in results])
    net  = np.array([r.get("net_displacement", np.nan)   for r in results])

    ugaps = np.sort(np.unique(gaps))
    umus  = np.sort(np.unique(mus))

    # 2D heatmap of total_path_length
    Z = np.full((len(umus), len(ugaps)), np.nan)
    for r in results:
        gi = np.searchsorted(ugaps, r["gap"])
        mi = np.searchsorted(umus,  r["mu"])
        Z[mi, gi] = r.get("total_path_length", np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Heatmap
    ax = axes[0]
    im = ax.imshow(Z, origin="lower", aspect="auto",
                   extent=[ugaps[0], ugaps[-1], umus[0], umus[-1]])
    ax.set_xlabel("Gap (tube_r − rod_r)")
    ax.set_ylabel("μ (friction)")
    ax.set_title("Total path length")
    fig.colorbar(im, ax=ax, label="path length")

    # (b) Path length vs gap, one curve per mu
    ax = axes[1]
    for mu_val in umus:
        mask = mus == mu_val
        order = np.argsort(gaps[mask])
        ax.plot(gaps[mask][order], path[mask][order],
                "o-", ms=4, label=f"μ={mu_val:.2f}")
    ax.set_xlabel("Gap")
    ax.set_ylabel("Total path length")
    ax.set_title("Path length vs gap")
    ax.legend(fontsize=7, ncol=2)

    # (c) Path length vs mu, one curve per gap
    ax = axes[2]
    for g_val in ugaps:
        mask = gaps == g_val
        order = np.argsort(mus[mask])
        ax.plot(mus[mask][order], path[mask][order],
                "s-", ms=4, label=f"gap={g_val:.3f}")
    ax.set_xlabel("μ (friction)")
    ax.set_ylabel("Total path length")
    ax.set_title("Path length vs μ")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "reptation_minimal.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"✓ Plot saved: {fig_path}")


if __name__ == "__main__":
    main()
