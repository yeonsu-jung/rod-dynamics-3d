#!/usr/bin/env python3
"""Local sweep runner for N=200 init configs in initial-configs/6,7,8.

Key constraints (per user request):
- Folder name like "6,7,8" encodes random seeds; do NOT use for box sizing.
- Run in a nonperiodic domain.
- Use assets/scenes/default_entangled.json as the scene template, but override
  rod diameter.
- Produce per-run all-pairs RMS distance time series and a PNG plot.

Examples:
    # Fresh sweep: derive diameter per run from AR#### in folder label
    python3 scripts/run_network_sweep.py --steps 2000 --perrod-max 2000 --analysis-frames 2000

    # Rerun an existing sweep (copies each subfolder's scene.json into a new run folder)
    python3 scripts/run_network_sweep.py \
        --rerun-from study/network/runs/20251218-011703_sweep_6,7,8 \
        --steps 2000 --perrod-max 2000 --analysis-frames 2000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "CMakeLists.txt").exists() and (p / "assets" / "scenes").exists():
            return p
    raise SystemExit("Could not find repo root (expected CMakeLists.txt + assets/scenes)")


def iter_init_files(init_root: Path, require_substring: str, filename: str) -> Iterable[Tuple[str, Path]]:
    """Yield (label, init_csv_path) for matching init configs."""
    for root, _dirs, files in os.walk(init_root):
        if filename not in files:
            continue
        root_path = Path(root)
        # Match by folder naming convention (e.g. contains N0200)
        if require_substring and require_substring not in root_path.name:
            continue
        yield root_path.name, root_path / filename


def parse_ar_from_label(label: str) -> Optional[int]:
    """Parse aspect ratio from a folder label like ...-AR0200-... -> 200."""
    m = re.search(r"AR(\d+)", label)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def iter_run_subfolders(runs_root: Path, require_substring: str) -> Iterable[str]:
    """Yield subfolder names under runs_root (one per run)."""
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if require_substring and require_substring not in name:
            continue
        yield name


def parse_int_like(s: str) -> int:
    """Parse ints from strings like '100000' or scientific notation like '1e5'."""
    try:
        v = float(s)
        iv = int(v)
        if iv <= 0:
            raise ValueError
        return iv
    except Exception:
        raise argparse.ArgumentTypeError(f"Expected a positive integer (or 1e5-style), got: {s}")


def load_scene_template(scene_template: Path) -> dict:
    with scene_template.open("r") as f:
        return json.load(f)


def apply_dt_override(scene: dict, dt: Optional[float]) -> None:
    if dt is None:
        return
    if dt <= 0:
        raise SystemExit("--dt must be > 0")
    try:
        scene.setdefault("physics", {})["dt"] = float(dt)
    except Exception:
        pass


def write_scene_override(scene: dict, out_path: Path, diameter: float) -> float:
    """Write scene.json with nonperiodic + diameter override. Returns rod_length."""
    if diameter <= 0:
        raise SystemExit("--diameter must be > 0")

    # Force nonperiodic
    try:
        scene.setdefault("scene", {}).setdefault("periodic", {})["enabled"] = False
    except Exception:
        pass

    # Override diameter (and populate.radius for consistency)
    rod_length = 1.0
    try:
        bodies = scene.setdefault("scene", {}).setdefault("bodies", [])
        if not bodies:
            bodies.append({})
        bodies[0]["diameter"] = float(diameter)
        rod_length = float(bodies[0].get("length", rod_length))
    except Exception:
        pass

    try:
        populate = scene.setdefault("scene", {}).setdefault("populate", {})
        populate["radius"] = float(diameter) * 0.5
    except Exception:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(scene, f, indent=2)
        f.write("\n")
    return rod_length


def run_cmd(cmd: List[str], cwd: Path, dry_run: bool) -> None:
    print("  $", " ".join(cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise SystemExit(f"Command failed (rc={proc.returncode})")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Run group name (default: timestamped)",
    )
    ap.add_argument(
        "--rerun-from",
        type=Path,
        default=None,
        help=(
            "Rerun an existing sweep folder: reads subfolders under this path, copies each subfolder's "
            "scene.json into a new run folder, then re-runs sim + analysis."
        ),
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Optional existing runs folder to write into (overrides --job-name)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    ap.add_argument(
        "--rewrite-scenes",
        action="store_true",
        help="Only (re)write per-run scene.json files; do not run sim or analysis",
    )
    ap.add_argument(
        "--init-root",
        type=Path,
        default=None,
        help="Root folder to search (default: <repo>/initial-configs/6,7,8)",
    )
    ap.add_argument(
        "--require-substring",
        type=str,
        default="N0200",
        help="Only run init folders whose name contains this substring (default: N0200)",
    )
    ap.add_argument(
        "--init-filename",
        type=str,
        default="x_relaxed.txt",
        help="Init file name inside each folder (default: x_relaxed.txt)",
    )
    ap.add_argument(
        "--scene-template",
        type=Path,
        default=None,
        help="Scene template JSON (default: assets/scenes/default_entangled.json)",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Optional override for physics.dt (applied to every per-run scene.json)",
    )
    ap.add_argument(
        "--diameter",
        type=float,
        default=None,
        help=(
            "Optional constant rod diameter override. If omitted, diameter is derived per run from "
            "the init folder label's AR: diameter = rod_length / AR."
        ),
    )
    ap.add_argument(
        "--steps",
        type=parse_int_like,
        default=parse_int_like("2000"),
        help="Simulation steps (default: 2000). Accepts 1e5-style notation.",
    )
    ap.add_argument(
        "--perrod-max",
        type=int,
        default=2000,
        help="Max frames to log to perrod.csv (default: 2000)",
    )
    ap.add_argument(
        "--analysis-start-frame",
        type=int,
        default=0,
        help="Start frame for all-pairs RMS analysis (default: 0)",
    )
    ap.add_argument(
        "--analysis-frames",
        type=int,
        default=2000,
        help="Number of frames for all-pairs RMS analysis (default: 2000)",
    )
    ap.add_argument(
        "--analysis-stride",
        type=int,
        default=1,
        help="Only analyze every k-th frame in perrod (passed to contact_analysis.py --frame-stride). Default: 1",
    )
    ap.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="If >0, limit number of init configs processed",
    )
    ap.add_argument(
        "--log-wave-period",
        type=int,
        default=0,
        help="Square wave logging period (frames). e.g., 1000",
    )
    ap.add_argument(
        "--log-wave-width",
        type=int,
        default=0,
        help="Square wave logging width (frames). e.g., 100",
    )
    ap.add_argument("--force", action="store_true", help="Force rerun even if results exist")
    args = ap.parse_args(argv)

    repo_root = find_repo_root(Path(__file__).resolve())
    build_dir = repo_root / "build"
    exe = build_dir / "rigidbody_viewer_3d"
    if not exe.exists():
        raise SystemExit(f"Executable not found: {exe} (build it first)")

    init_root = args.init_root or (repo_root / "initial-configs" / "6,7,8")
    if not init_root.exists():
        raise SystemExit(f"Init root not found: {init_root}")

    scene_template = args.scene_template or (repo_root / "assets" / "scenes" / "default_entangled.json")
    if not scene_template.exists():
        raise SystemExit(f"Scene template not found: {scene_template}")

    if args.rerun_from is not None:
        if args.rewrite_scenes:
            raise SystemExit("--rerun-from cannot be combined with --rewrite-scenes")
        if not args.rerun_from.exists():
            raise SystemExit(f"--rerun-from not found: {args.rerun_from}")

    default_job = datetime.now().strftime("%Y%m%d-%H%M%S") + "_sweep_6,7,8"
    if args.rerun_from is not None:
        default_job = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_rerun_{args.rerun_from.name}"
    job_name = args.job_name or default_job
    runs_root = args.runs_root or (repo_root / "study" / "network" / "runs" / job_name)
    if not args.dry_run:
        runs_root.mkdir(parents=True, exist_ok=True)

    analysis_script = repo_root / "study" / "network" / "contact_analysis.py"
    if not analysis_script.exists():
        raise SystemExit(f"Analysis script not found: {analysis_script}")

    # Prefer repo-local venv python if available (keeps deps consistent)
    venv_py = repo_root / ".venv" / "bin" / "python"
    python_exe = str(venv_py) if venv_py.exists() else sys.executable

    if args.rerun_from is None:
        init_items = list(iter_init_files(init_root, args.require_substring, args.init_filename))
        if not init_items:
            raise SystemExit(
                f"No init configs found under {init_root} matching name contains '{args.require_substring}' "
                f"and file '{args.init_filename}'"
            )
        print(f"Found {len(init_items)} init configs under {init_root}")
        labels = [lbl for (lbl, _p) in init_items]
    else:
        labels = list(iter_run_subfolders(args.rerun_from, args.require_substring))
        if not labels:
            raise SystemExit(f"No run subfolders found under {args.rerun_from}")
        print(f"Found {len(labels)} run subfolders under {args.rerun_from}")
    processed = 0
    total_runs = len(labels) if not args.max_runs else min(len(labels), int(args.max_runs))
    for label in labels:
        if args.max_runs and processed >= args.max_runs:
            break

        # Init file resolved by label (same folder naming convention)
        init_path = init_root / label / args.init_filename
        if not init_path.exists():
            raise SystemExit(f"Missing init file for label '{label}': {init_path}")

        run_dir = runs_root / label
        init_local = run_dir / args.init_filename
        perrod_csv = run_dir / "perrod.csv"
        allpairs_csv = run_dir / "allpairs_rms_dmin.csv"
        allpairs_png = run_dir / "allpairs_rms_dmin.png"
        scene_out = run_dir / "scene.json"

        need_sim = args.force or (not perrod_csv.exists())
        need_analysis = args.force or (not allpairs_csv.exists()) or (not allpairs_png.exists())

        print(f"[{processed+1}/{total_runs}] {label}")
        print(f"  init: {init_path}")
        print(f"  run:  {run_dir}")

        if args.rerun_from is not None:
            # Rerun mode: copy scene.json from the source sweep
            src_dir = args.rerun_from / label
            src_scene = src_dir / "scene.json"
            if not src_scene.exists():
                raise SystemExit(f"Missing scene.json in source run: {src_scene}")
            # Determine rod_length for analysis from the source per-run scene.json
            scene_loaded = load_scene_template(src_scene)
            rod_length = float(
                ((scene_loaded.get("scene", {}).get("bodies", []) or [{}])[0]).get("length", 1.0)
            )
            # Apply dt override (if requested) and write to destination scene.json
            apply_dt_override(scene_loaded, args.dt)
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                # copy init-csv into run folder for reproducibility / easy rerun
                shutil.copy2(init_path, init_local)
                with scene_out.open("w") as f:
                    json.dump(scene_loaded, f, indent=2)
                    f.write("\n")
                # carry template too if present
                src_tpl = src_dir / "scene_template.json"
                if src_tpl.exists():
                    shutil.copy2(src_tpl, run_dir / "scene_template.json")
        else:
            ar = parse_ar_from_label(label)
            if args.diameter is None:
                if not ar or ar <= 0:
                    raise SystemExit(
                        f"Could not parse AR from label '{label}'. "
                        "Provide --diameter explicitly or ensure label contains AR####."
                    )
            else:
                ar = None

            # Keep a copy of the original template for reproducibility
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                # copy init-csv into run folder for reproducibility / easy rerun
                shutil.copy2(init_path, init_local)
                shutil.copy2(scene_template, run_dir / "scene_template.json")

            scene = load_scene_template(scene_template)
            apply_dt_override(scene, args.dt)
            # Determine rod length from template (used for AR->diameter conversion)
            rod_length_template = 1.0
            try:
                b = (scene.get("scene", {}).get("bodies", []) or [{}])[0]
                rod_length_template = float(b.get("length", rod_length_template))
            except Exception:
                pass
            diameter_used = float(args.diameter) if args.diameter is not None else (rod_length_template / float(ar))
            print(f"  diameter: {diameter_used:.6g}" + (f" (from AR={ar})" if ar else " (override)"))
            rod_length = write_scene_override(scene, scene_out, diameter=diameter_used)

        if args.rewrite_scenes:
            processed += 1
            print("  (rewrite-scenes: skipping sim + analysis)")
            print("-" * 60)
            continue

        if need_sim:
            sim_cmd = [
                str(exe),
                "--headless",
                "--scene",
                scene_out.name,
                "--init-csv",
                args.init_filename,
                "--steps",
                str(int(args.steps)),
            ]
            sim_cmd.extend(["--perrod", str(perrod_csv)])
            sim_cmd.extend(["--perrod-max", str(args.perrod_max)])
            if args.log_wave_period > 0 and args.log_wave_width > 0:
                 sim_cmd.extend(["--log-wave-period", str(args.log_wave_period)])
                 sim_cmd.extend(["--log-wave-width", str(args.log_wave_width)])
            
            # If network path is not explicitly set, network.csv is default, but we should probably explicit it 
            # or rely on default. Let's explicit it to be safe if we want to ensure it's logged.
            sim_cmd.extend(["--network", str(run_dir / "network.csv")])
            sim_cmd.extend(["--soft-pe", str(run_dir / "energy.csv")])
            sim_cmd.extend(["--output", str(run_dir / "ke.csv")])
            
            # Entanglement defaults
            sim_cmd.append("--entanglement")
            sim_cmd.extend(["--entanglement-cutoff", "0.0"]) # Keep all crossings
            sim_cmd.extend(["--entanglement-period", "100"])
            
            run_cmd(sim_cmd, cwd=run_dir, dry_run=args.dry_run)
        else:
            print("  (skip sim: perrod.csv exists)")

        if need_analysis:
            analysis_cmd = [
                python_exe,
                str(analysis_script),
                "--all-pairs-rms",
                "--perrod",
                str(perrod_csv),
                "--rod-length",
                str(float(rod_length)),
                "--start-frame",
                str(int(args.analysis_start_frame)),
                "--frames",
                str(int(args.analysis_frames)),
                "--frame-stride",
                str(int(args.analysis_stride)),
                "--out-allpairs-stats",
                str(allpairs_csv),
                "--plot-allpairs",
                "--plot-out",
                str(allpairs_png),
            ]
            run_cmd(analysis_cmd, cwd=repo_root, dry_run=args.dry_run)
        else:
            print("  (skip analysis: allpairs outputs exist)")

        processed += 1
        print("-" * 60)

    print(f"Done. Runs in: {runs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
