#!/usr/bin/env python3
"""Reproduce Supplementary Videos 2-4 (equivalents from committed packings).

  video2_cohesion  N=200 alpha=300  mu=0.4  (paper: same parameters)
  video3_fragile   N=200 alpha=20   mu=1.0  (paper: alpha=25)
  video4_woven     N=200 alpha=1000 mu=1.0  (paper: N=331, alpha=1000)

Each video: (1) headless run with the Table S1 protocol + NDJSON snapshots
over 32 t_u, (2) GL playback -> PNG frames -> ffmpeg mp4 (needs a display;
run on the desktop session or under xvfb-run).

Usage: python3 repro/make_videos.py [--only video2_cohesion] [--sim-only]
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "repro"))
from gen_manifest import scene_json  # noqa: E402

META = REPO / "assets" / "packings_metadata.csv"
OUT = REPO / "videos"
HEADLESS = REPO / "build-headless" / "rigidbody_viewer_3d"
GL = REPO / "build-gl" / "rigidbody_viewer_3d"

STEPS = 102_400        # 32 t_u
SNAP_STRIDE = 100      # -> 1024 frames = ~34 s at 30 fps
FPS = 30

VIDEOS = {
    "video2_cohesion": ("2025-02-16_17_EntangledRelaxedPacking-N0200-AR0300",
                        0.4),
    "video3_fragile": ("2025-02-18_18_EntangledRelaxedPacking-N0200-AR0020",
                       1.0),
    "video4_woven": ("2025-11-05_21_EntangledRelaxedPacking-N0200-AR1000",
                     1.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append")
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--cam-scale", type=float, default=2.2)
    args = ap.parse_args()

    packs = {p["packing_id"]: p for p in csv.DictReader(META.open())}
    OUT.mkdir(exist_ok=True)

    for name, (dirfrag, mu) in VIDEOS.items():
        if args.only and name not in args.only:
            continue
        p = next(v for k, v in packs.items()
                 if k.startswith("6,7,8/") and dirfrag in k)
        vd = OUT / name
        vd.mkdir(exist_ok=True)
        scene = vd / "scene.json"
        scene.write_text(json.dumps(scene_json(p, mu, seed=1), indent=1))
        snap = vd / "snapshots.ndjson"

        if not snap.exists():
            print(f"[{name}] simulating {STEPS} steps (mu={mu}, "
                  f"alpha={p['alpha_nominal']})...")
            subprocess.run(
                [str(HEADLESS), "--scene", str(scene), "--headless",
                 "--steps", str(STEPS), "--no-csv",
                 "--snap-stride", str(SNAP_STRIDE),
                 "--snap-frames", str(STEPS // SNAP_STRIDE),
                 "--snap-path", str(snap),
                 "--status-stride", "1000000", "--no-headless-progress"],
                cwd=REPO, check=True)
        else:
            print(f"[{name}] snapshots exist, skipping simulation")

        if args.sim_only:
            continue

        # The kick leaves net momentum, so the packing drifts out of the
        # (first-frame) auto-framed view. Re-center every snapshot on the
        # per-axis median rod position (robust to escaped rods).
        centered = vd / "snapshots_centered.ndjson"
        recenter(snap, centered)

        mp4 = vd / f"{name}.mp4"
        print(f"[{name}] rendering -> {mp4}")
        subprocess.run(
            [str(GL), "--playback", str(centered),
             "--export", str(vd / "frames"), "--frames-only",
             "--auto-frame", "--cam-scale", str(args.cam_scale),
             "--fps", str(FPS), "--movie", str(mp4)],
            cwd=REPO, check=True)
        print(f"[{name}] done")


def recenter(src, dst):
    import statistics
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            j = json.loads(line)
            bodies = j.get("bodies", [])
            if bodies:
                med = [statistics.median(b["pos"][k] for b in bodies)
                       for k in range(3)]
                for b in bodies:
                    b["pos"] = [b["pos"][k] - med[k] for k in range(3)]
            fout.write(json.dumps(j, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
