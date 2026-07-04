#!/usr/bin/env python3
"""Build assets/packings_metadata.csv: one row per committed packing.

Static per-packing quantities used across the reproduction pipeline:
  - N, nominal/actual aspect ratio, rod length L and diameter d
  - contact count at the paper's static criterion d_ij < 1.01 d  (SI S6),
    coordination number Z = 2 Nc / N
  - contact spread R/l (Eq. 5 of the paper; radius of gyration of contact
    points about the packing centroid)  -> Fig 2
  - ebar0 = initial normalized entanglement, evaluated by the production
    binary (1-step headless run with --entanglement)  -> Fig 4C,D retention

Columns gbar_t, gbar_r, astar_finite are left empty; they come from an
offline pass of https://github.com/yeonsu-jung/rod-free-volume (Fig 4A,B).

Usage:  python3 repro/build_packing_metadata.py [--binary PATH] [--skip-ent]
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO / "assets" / "initial-configs"
OUT_CSV = REPO / "assets" / "packings_metadata.csv"
CONTACT_FACTOR = 1.01  # SI S6: contact iff separation < 1.01 d


def parse_packing(path):
    """Read an endpoint file: header comments + N rows of 6 floats."""
    radius = None
    length_header = None
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.search(r"rod_radius\s*=\s*([0-9.eE+-]+)", line)
            if m:
                radius = float(m.group(1))
            m = re.search(r"rod_length\s*=\s*([0-9.eE+-]+)", line)
            if m:
                length_header = float(m.group(1))
            continue
        vals = line.split()
        if len(vals) >= 6:
            rows.append([float(v) for v in vals[:6]])
    arr = np.asarray(rows)
    return arr[:, 0:3], arr[:, 3:6], radius, length_header


def segseg_distance(p1, q1, p2, q2):
    """Vectorized closest-distance between segment batches (Ericson 5.1.9).

    All inputs (M,3). Returns (dist, closest points c1, c2) each (M,) / (M,3).
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    denom = a * e - b * b

    # general case; degenerate (parallel) handled by denom ~ 0 -> s = 0
    s = np.where(denom > 1e-12 * a * e + 1e-30,
                 np.clip((b * f - c * e) / np.where(denom == 0, 1, denom),
                         0.0, 1.0),
                 0.0)
    tnom = b * s + f
    t = np.clip(tnom / np.where(e == 0, 1, e), 0.0, 1.0)
    # re-clamp s for clamped t
    s = np.where(tnom < 0, np.clip(-c / np.where(a == 0, 1, a), 0, 1), s)
    s = np.where(tnom > e, np.clip((b - c) / np.where(a == 0, 1, a), 0, 1), s)

    c1 = p1 + s[:, None] * d1
    c2 = p2 + t[:, None] * d2
    dist = np.linalg.norm(c1 - c2, axis=1)
    return dist, c1, c2


def static_metrics(p0, p1, diameter):
    n = len(p0)
    ii, jj = np.triu_indices(n, k=1)
    dist, c1, c2 = segseg_distance(p0[ii], p1[ii], p0[jj], p1[jj])
    contact = dist < CONTACT_FACTOR * diameter
    nc = int(contact.sum())
    z = 2.0 * nc / n
    centers = 0.5 * (p0 + p1)
    r0 = centers.mean(axis=0)
    if nc:
        cpts = 0.5 * (c1[contact] + c2[contact])
        rr = float(np.sqrt(np.mean(np.sum((cpts - r0) ** 2, axis=1))))
    else:
        rr = float("nan")
    return nc, z, rr


def ebar0_from_binary(binary, csv_path_rel, tmpdir):
    """1-step headless run of the production binary; ebar0 = ent_sum/ent_pairs
    at frame 0."""
    scene = {
        "scene": {"initCsvPath": str(csv_path_rel)},
        "physics": {"dt": 0.001, "gravity": [0, 0, 0],
                    "nsc": {"enabled": True, "mu": 0.0,
                            "velocity_iters": 1, "beta": 0.0,
                            "cfm": 0.05, "omega": 1.0}},
    }
    sp = Path(tmpdir) / "ent_scene.json"
    sp.write_text(json.dumps(scene))
    out = Path(tmpdir) / "ent.csv"
    if out.exists():
        out.unlink()
    subprocess.run(
        [str(binary), "--scene", str(sp), "--headless", "--steps", "1",
         "--csv", str(out), "--entanglement", "--ent-period", "1",
         "--status-stride", "1000000", "--no-headless-progress"],
        cwd=REPO, check=True, capture_output=True)
    with out.open() as fh:
        row = next(r for r in csv.DictReader(fh) if r["frame"] == "0")
    pairs = float(row["ent_pairs"])
    return float(row["ent_sum"]) / pairs if pairs else float("nan"), pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary",
                    default=REPO / "build-headless" / "rigidbody_viewer_3d")
    ap.add_argument("--skip-ent", action="store_true",
                    help="skip ebar0 (no binary runs)")
    args = ap.parse_args()

    files = sorted(CONFIG_ROOT.rglob("x_relaxed.txt"))
    if not files:
        sys.exit("no packings found under " + str(CONFIG_ROOT))

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            rel = f.relative_to(REPO)
            dirname = f.parent.name
            seed_group = f.parent.parent.name
            m = re.search(r"N(\d+)-AR(\d+)", dirname)
            n_nom = int(m.group(1)) if m else -1
            ar_nom = int(m.group(2)) if m else -1

            p0, p1, radius, _ = parse_packing(f)
            n = len(p0)
            lengths = np.linalg.norm(p1 - p0, axis=1)
            L = float(lengths.mean())
            d = 2.0 * radius
            nc, z, rr = static_metrics(p0, p1, d)

            ebar0, ent_pairs = (float("nan"), float("nan"))
            if not args.skip_ent:
                ebar0, ent_pairs = ebar0_from_binary(args.binary, rel, td)

            rows.append({
                "packing_id": f"{seed_group}/{dirname}",
                "path": str(rel),
                "seed_group": seed_group,
                "N": n,
                "alpha_nominal": ar_nom,
                "L_mean": round(L, 6),
                "d": d,
                "alpha_actual": round(L / d, 3),
                "n_contacts": nc,
                "Z": round(z, 4),
                "R_over_l": round(rr / L, 5) if rr == rr else "",
                "ebar0": round(ebar0, 6) if ebar0 == ebar0 else "",
                "ent_pairs": int(ent_pairs) if ent_pairs == ent_pairs else "",
                "gbar_t": "",       # from rod-free-volume (offline pass)
                "gbar_r": "",       # from rod-free-volume (offline pass)
                "astar_finite": "",  # from rod-free-volume (offline pass)
            })
            print(f"{rows[-1]['packing_id']}: N={n} alpha={ar_nom} "
                  f"Z={z:.2f} R/l={rows[-1]['R_over_l']} "
                  f"ebar0={rows[-1]['ebar0']}")
            if n != n_nom:
                print(f"  WARNING: row count {n} != dirname N {n_nom}")

    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV} ({len(rows)} packings)")


if __name__ == "__main__":
    main()
