#!/usr/bin/env python3
"""submit_entangled_packings_n200.py

Submit SLURM jobs that run the simulator on pre-generated relaxed packings.

Targets files named `x_relaxed_AR*.txt` inside subfolders under:
  initial-configs/entangled_packings/N200

Each run:
- writes a per-run `scene.json` (based on a template)
- sets `scene.initCsv` to the chosen `x_relaxed_AR*.txt`
- sets box + rod radius from `voronoi_analysis_x_relaxed_AR*.json` if present
- generates `Sbatch.sh` and optionally submits via `sbatch`

Notes
- The C++ loader `App::loadInitialConfigCSV` supports whitespace-separated
  endpoint files with optional `# rod_radius=...` header.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple


def find_root_dir(start: Optional[Path] = None, target_name: str = "rod-dynamics-3d") -> Path:
    p = Path.cwd() if start is None else start.resolve()
    for ancestor in [p, *p.parents]:
        if ancestor.name == target_name:
            return ancestor
    raise SystemExit(f"Could not find repository root named '{target_name}' starting from {p}")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_executable(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if not os.access(path, os.X_OK):
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


def parse_ar_from_filename(name: str) -> Optional[int]:
    # x_relaxed_AR10.txt -> 10
    m = re.search(r"x_relaxed_AR(\d+)\.txt$", name)
    if not m:
        return None
    return int(m.group(1))


def parse_rod_radius_from_txt(path: Path) -> Optional[float]:
    # First non-empty line may be: # rod_radius = 0.05
    try:
        with path.open("r") as f:
            for _ in range(10):
                line = f.readline()
                if not line:
                    return None
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#") and "rod_radius" in line:
                    m = re.search(r"rod_radius\s*=\s*([0-9eE\+\-\.]+)", line)
                    if m:
                        return float(m.group(1))
                # stop if we hit data
                if not line.startswith("#"):
                    return None
    except OSError:
        return None
    return None


@dataclass(frozen=True)
class PackingParams:
    num_rods: int
    rod_radius: float
    box_size: float


def load_voronoi_params(voronoi_json: Path) -> Optional[PackingParams]:
    if not voronoi_json.exists():
        return None
    try:
        with voronoi_json.open("r") as f:
            obj = json.load(f)
        params = obj.get("params", {})
        num_rods = int(params.get("num_rods", 200))
        rod_radius = float(params["rod_radius"]) if "rod_radius" in params else None
        box_size = float(params["box_size"]) if "box_size" in params else None
        if rod_radius is None or box_size is None:
            return None
        return PackingParams(num_rods=num_rods, rod_radius=rod_radius, box_size=box_size)
    except Exception:
        return None


def deep_copy_json(obj):
    return json.loads(json.dumps(obj))


def iter_packing_files(configs_dir: Path) -> Sequence[Tuple[Path, Path, int]]:
    """Return list of (folder, file, ar) for all x_relaxed_AR*.txt."""
    out = []
    for folder in sorted([d for d in configs_dir.iterdir() if d.is_dir()]):
        for txt in sorted(folder.glob("x_relaxed_AR*.txt")):
            ar = parse_ar_from_filename(txt.name)
            if ar is None:
                continue
            out.append((folder, txt, ar))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Submit SLURM jobs for entangled packings (N200) using x_relaxed_AR*.txt files."
    )
    ap.add_argument("--job-name", type=str, default="entangled_packings_N200", help="Run group name under runs/")
    ap.add_argument("--runs-root", type=Path, default=None, help="Override runs root directory")
    ap.add_argument("--configs-dir", type=Path, default=None, help="Override configs dir (default: initial-configs/entangled_packings/N200)")
    ap.add_argument("--scene-template", type=Path, default=None, help="Scene template JSON (default: assets/scenes/default_entangled.json)")

    ap.add_argument("--ar", type=int, action="append", default=[], help="Only run this AR (repeatable)")
    ap.add_argument("--max-runs", type=int, default=0, help="If >0, limit total number of runs")

    ap.add_argument("--dt", type=float, default=0.0005, help="physics.dt")
    ap.add_argument("--steps", type=int, default=5000, help="Simulation steps")
    ap.add_argument("--threads", type=int, default=4, help="Simulation threads (--threads)")

    ap.add_argument("--fSigma", type=float, action="append", default=[0.0], help="randomForce.fSigma (repeatable)")
    ap.add_argument("--tauMag", type=float, default=0.0, help="randomForce.tauMag")

    ap.add_argument("--pbc", dest="pbc", action="store_true", help="Enable PBC (default)")
    ap.add_argument("--no-pbc", dest="pbc", action="store_false", help="Disable PBC")
    ap.set_defaults(pbc=True)

    ap.add_argument("--default-box-size", type=float, default=1.0, help="Fallback box size if metadata missing")

    ap.add_argument("--dry-run", action="store_true", help="Write files but do not submit")

    # SLURM knobs
    ap.add_argument("--partition", type=str, default="seas_compute")
    ap.add_argument("--time", type=str, default="7-00:00")
    ap.add_argument("--mem", type=str, default="100")
    ap.add_argument("--cpus", type=int, default=4)

    args = ap.parse_args()

    root = find_root_dir()

    runs_root = args.runs_root
    if runs_root is None:
        runs_root = Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs") / args.job_name
    runs_root.mkdir(parents=True, exist_ok=True)

    # Copy script for provenance
    try:
        shutil.copy2(Path(__file__), runs_root / Path(__file__).name)
    except Exception:
        pass

    binary_src = root / "build" / "rigidbody_viewer_3d"
    ensure_executable(binary_src)

    scene_template = args.scene_template
    if scene_template is None:
        scene_template = root / "assets" / "scenes" / "default_entangled.json"
    if not scene_template.exists():
        raise SystemExit(f"Scene template not found: {scene_template}")

    configs_dir = args.configs_dir
    if configs_dir is None:
        configs_dir = root / "initial-configs" / "entangled_packings" / "N200"
    if not configs_dir.exists():
        raise SystemExit(f"Configs directory not found: {configs_dir}")

    with scene_template.open("r") as f:
        base_scene = json.load(f)

    timestamp = now_ts()

    packings = iter_packing_files(configs_dir)
    if args.ar:
        allowed = set(args.ar)
        packings = [p for p in packings if p[2] in allowed]

    if not packings:
        print(f"No x_relaxed_AR*.txt found under {configs_dir}")
        return

    run_count = 0
    for folder, txt_path, ar in packings:
        # Prefer voronoi_analysis json for params
        voronoi_json = folder / f"voronoi_analysis_x_relaxed_AR{ar}.json"
        params = load_voronoi_params(voronoi_json)
        if params is None:
            rod_radius = parse_rod_radius_from_txt(txt_path)
            if rod_radius is None:
                print(f"Skipping {txt_path}: missing rod_radius (no voronoi JSON, no header)")
                continue
            params = PackingParams(num_rods=200, rod_radius=rod_radius, box_size=float(args.default_box_size))

        diameter = 2.0 * params.rod_radius
        rod_length = float(ar) * diameter

        for fSigma in args.fSigma:
            run_name = f"{timestamp}_{folder.name}_AR{ar}_fSig{fSigma:.1e}"
            run_dir = runs_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            # Copy binary into run dir (consistent with other submit scripts)
            shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")

            scene = deep_copy_json(base_scene)

            # physics
            scene.setdefault("physics", {})
            scene["physics"]["dt"] = float(args.dt)

            # scene fields
            scene.setdefault("scene", {})
            scene["scene"]["initCsv"] = str(txt_path)

            # populate is used as metadata + count limiter in the loader
            scene["scene"].setdefault("populate", {})
            scene["scene"]["populate"]["count"] = int(params.num_rods)
            scene["scene"]["populate"]["length"] = float(rod_length)
            scene["scene"]["populate"]["radius"] = float(params.rod_radius)

            # bodies template (for material props)
            if "bodies" in scene["scene"] and scene["scene"]["bodies"]:
                scene["scene"]["bodies"][0]["length"] = float(rod_length)
                scene["scene"]["bodies"][0]["diameter"] = float(diameter)

            # PBC box
            half_box = 0.5 * float(params.box_size)
            scene["scene"].setdefault("periodic", {})
            scene["scene"]["periodic"]["enabled"] = bool(args.pbc)
            scene["scene"]["periodic"]["min"] = [-half_box, -half_box, -half_box]
            scene["scene"]["periodic"]["max"] = [half_box, half_box, half_box]

            # random forcing
            scene["scene"].setdefault("randomForce", {})
            scene["scene"]["randomForce"]["enabled"] = bool(fSigma != 0.0 or args.tauMag != 0.0)
            scene["scene"]["randomForce"]["fSigma"] = float(fSigma)
            scene["scene"]["randomForce"]["tauMag"] = float(args.tauMag)

            # disable random init by default for relaxed packings
            scene["scene"].setdefault("randomInit", {})
            scene["scene"]["randomInit"]["enabled"] = False

            (run_dir / "scene.json").write_text(json.dumps(scene, indent=2))

            # snapshots
            target_frames = 500
            snap_stride = max(1, int(args.steps) // target_frames)

            sim_cmd = (
                "time "
                "./rigidbody_viewer_3d "
                "--headless "
                "--scene scene.json "
                "--output output.csv "
                f"--snap-stride {snap_stride} "
                f"--snap-frames {target_frames} "
                "--perrod perrod.csv "
                f"--perrod-max {target_frames} "
                f"--perrod-stride {snap_stride} "
                f"--steps {int(args.steps)} "
                f"--threads {int(args.threads)}"
            )

            sbatch = f"""#!/bin/bash
#SBATCH -n 1
#SBATCH -c {int(args.cpus)}
#SBATCH -N 1
#SBATCH -t {args.time}
#SBATCH -p {args.partition}
#SBATCH --mem={args.mem}
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err
#SBATCH --mail-type=END
#SBATCH --job-name={args.job_name}_{folder.name}_AR{ar}

set -euo pipefail
module load python

echo "======================================"
echo "Packing: {folder.name} | AR={ar} | fSigma={fSigma:.3e}"
echo "Init: {txt_path}"
echo "PWD: $(pwd)"
echo "======================================"

echo "{sim_cmd}"
{sim_cmd}
"""

            sbatch_path = run_dir / "Sbatch.sh"
            sbatch_path.write_text(sbatch)
            os.chmod(sbatch_path, 0o755)

            run_count += 1

            if args.dry_run:
                print(f"[Dry Run] {run_dir}")
            else:
                subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir, check=True)

            if args.max_runs > 0 and run_count >= args.max_runs:
                print(f"Reached --max-runs={args.max_runs}; stopping.")
                return


if __name__ == "__main__":
    main()
