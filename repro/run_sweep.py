#!/usr/bin/env python3
"""Resume-safe sweep runner for the reproduction manifest.

Runs every manifest row as a single-threaded headless process; parallelism
comes from a process pool (--jobs). Each run gets runs/<run_id>/ with:
  scene.json      copy of the generated scene (provenance)
  meta.json       git SHA, command, timings, exit status, config hash
  log.txt         stdout+stderr of the binary
  profile.csv     per-frame CSV (KE, contacts, collisions, ent_*, step_ms)

A run is skipped when meta.json says status=done AND the stored config hash
matches (scene content + relevant manifest fields), so interrupted sweeps
resume with:  python3 repro/run_sweep.py [--group c1] [--jobs N]

Dry-run cost preview:  python3 repro/run_sweep.py --dry-run
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "repro" / "manifest.csv"
RUNS = REPO / "runs"
BINARY = REPO / "build-headless" / "rigidbody_viewer_3d"


def config_hash(row, scene_path):
    h = hashlib.sha256()
    h.update(scene_path.read_bytes())
    for k in ("steps", "csv_stride", "ent_period", "extra_args"):
        h.update(str(row[k]).encode())
    return h.hexdigest()[:16]


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_cmd(row, run_dir):
    cmd = [str(BINARY),
           "--scene", str(run_dir / "scene.json"),
           "--headless", "--steps", str(row["steps"]),
           "--csv", str(run_dir / "profile.csv"),
           "--csv-stride", str(row["csv_stride"]),
           "--entanglement", "--ent-period", str(row["ent_period"]),
           "--ent-threads", "1",
           "--status-stride", "1000000", "--no-headless-progress"]
    extra = row["extra_args"].replace("{run_dir}", str(run_dir)).split()
    return cmd + extra


def one_run(row, sha, force=False):
    run_dir = RUNS / row["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    scene_src = REPO / row["scene"]
    scene_dst = run_dir / "scene.json"
    scene_dst.write_bytes(scene_src.read_bytes())
    chash = config_hash(row, scene_dst)

    meta_path = run_dir / "meta.json"
    if meta_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("status") == "done" and meta.get("hash") == chash:
                return row["run_id"], "skipped", 0.0
        except Exception:
            pass

    cmd = build_cmd(row, run_dir)
    t0 = time.time()
    with (run_dir / "log.txt").open("w") as log:
        proc = subprocess.run(cmd, cwd=REPO, stdout=log,
                              stderr=subprocess.STDOUT)
    wall = time.time() - t0
    meta = {"run_id": row["run_id"], "hash": chash, "git_sha": sha,
            "cmd": " ".join(cmd), "wall_s": round(wall, 1),
            "exit_code": proc.returncode,
            "status": "done" if proc.returncode == 0 else "failed",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta_path.write_text(json.dumps(meta, indent=1))
    return row["run_id"], meta["status"], wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append",
                    help="only run these manifest groups (repeatable)")
    ap.add_argument("--jobs", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, help="run at most this many")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if done")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not BINARY.exists():
        sys.exit(f"binary not found: {BINARY} (build with "
                 "cmake -B build-headless -DBUILD_HEADLESS=ON && "
                 "cmake --build build-headless -j)")
    if not MANIFEST.exists():
        sys.exit("no manifest; run repro/gen_manifest.py first")

    rows = list(csv.DictReader(MANIFEST.open()))
    if args.group:
        rows = [r for r in rows if r["group"] in args.group]
    if args.limit:
        rows = rows[:args.limit]
    # long runs first for better pool packing (N=500 before N=200)
    rows.sort(key=lambda r: -int(r["N"]) * int(r["steps"]))

    if args.dry_run:
        for r in rows[:10]:
            print(r["run_id"])
        print(f"... {len(rows)} runs total, {args.jobs} jobs")
        return

    sha = git_sha()
    done = skipped = failed = 0
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(one_run, r, sha, args.force): r for r in rows}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rid, status, wall = fut.result()
            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"  FAILED: {rid} (see runs/{rid}/log.txt)")
            if status != "skipped":
                el = time.time() - t0
                print(f"[{i}/{len(rows)}] {rid}: {status} "
                      f"({wall:.0f}s, elapsed {el/60:.1f}m)")
    print(f"\ndone={done} skipped={skipped} failed={failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
