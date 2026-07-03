#!/usr/bin/env python3
"""Run the same scene under several contact models and compare them by
wall-clock cost and basic physics observables.

Usage:
    python3 scripts/compare_contact_models.py --scene assets/scenes/X.json \
        [--binary build-headless/rigidbody_viewer_3d] [--steps 500] \
        [--models nsc harmonic hertz-mindlin]

For each model this runs the headless binary with --contact-model and a
profile CSV, then reports mean step_ms (the common cost axis across
models), contact counts, and final KE. Wall-clock per step is what makes
results comparable across models with different notions of "iteration".
"""

import argparse
import csv
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_model(binary, scene, model, steps, workdir):
    profile = Path(workdir) / f"profile_{model}.csv"
    cmd = [
        str(binary), "--scene", str(scene), "--headless",
        "--steps", str(steps), "--quiet",
        "--contact-model", model,
        "--csv", str(profile),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[{model}] FAILED: {proc.stderr.strip().splitlines()[-1:] }")
        return None

    rows = list(csv.DictReader(open(profile)))
    if len(rows) < 3:
        print(f"[{model}] no profile rows")
        return None
    # Skip warmup frames.
    body = rows[max(1, len(rows) // 10):]
    step_ms = [float(r["step_ms"]) for r in body if "step_ms" in r]
    contacts = [int(r["contacts"]) for r in body if r.get("contacts")]
    ke_final = float(rows[-1]["KE"])
    return {
        "model": model,
        "step_ms_mean": statistics.mean(step_ms) if step_ms else float("nan"),
        "step_ms_std": statistics.pstdev(step_ms) if step_ms else float("nan"),
        "contacts_mean": statistics.mean(contacts) if contacts else 0,
        "ke_final": ke_final,
        "sim_s_per_wall_s": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--binary",
                    default=str(REPO / "build-headless" / "rigidbody_viewer_3d"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--models", nargs="+",
                    default=["nsc", "harmonic"])
    args = ap.parse_args()

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for model in args.models:
            r = run_model(args.binary, args.scene, model, args.steps, tmp)
            if r:
                results.append(r)

    if not results:
        print("No successful runs.")
        return 1

    print(f"\nScene: {args.scene}   steps: {args.steps}")
    print(f"{'model':>14} {'step_ms':>10} {'±std':>8} {'contacts':>10} "
          f"{'final KE':>12}")
    for r in results:
        print(f"{r['model']:>14} {r['step_ms_mean']:>10.3f} "
              f"{r['step_ms_std']:>8.3f} {r['contacts_mean']:>10.0f} "
              f"{r['ke_final']:>12.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
