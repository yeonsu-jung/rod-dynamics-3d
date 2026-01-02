#!/usr/bin/env python3
"""submit_entangled_n200.py

Submit SLURM jobs to run rod dynamics starting from entangled packings.

This script scans:
  initial-configs/entangled_packings/N200/**/x_relaxed_AR*.txt

For each x_relaxed_AR*.txt, it creates a run folder containing:
  - rigidbody_viewer_3d binary (copied)
  - scene.json (copied from assets/scenes/default_entangled.json)
  - Sbatch.sh

Each job runs:
  ./rigidbody_viewer_3d --headless --scene scene.json --init-csv <x_relaxed...>

Notes:
- The x_relaxed_AR*.txt files are treated as an init-csv (endpoints format).
- Output is written to output.csv (compact metrics) and profile.csv.
"""



import argparse
import json
import os
import re
import shutil
import stat
import subprocess

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


DEFAULT_RUNS_ROOT = Path(
    "/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs"
)


def find_root_dir(start: Optional[Path] = None, target_name: str = "rod-dynamics-3d") -> Path:
    p = (Path.cwd() if start is None else start).resolve()
    for ancestor in [p, *p.parents]:
        if ancestor.name == target_name:
            return ancestor
    raise SystemExit(
        f"Could not find repository root named '{target_name}' starting from {p}"
    )


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_executable(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if not os.access(path, os.X_OK):
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


_AR_RE = re.compile(r"x_relaxed_AR(\d+)\.txt$")


def iter_x_relaxed_files(root: Path) -> Iterable[Tuple[Path, int]]:
    """Yield (file_path, AR) for all x_relaxed_AR*.txt under root."""
    for p in root.rglob("x_relaxed_AR*.txt"):
        m = _AR_RE.search(p.name)
        if not m:
            continue
        yield (p, int(m.group(1)))


def safe_name(s: str) -> str:
    # SLURM job-name + folder names: avoid spaces and weird chars
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", s)


class SlurmCfg:
    def __init__(self, partition="seas_compute", time="7-00:00", mem_gb=16, ntasks=1, cpus=8, nodes=1, mail_user=os.environ.get("USER_EMAIL", ""), mail_type="END", module_line="module load python"):
        self.partition = partition
        self.time = time
        self.mem_gb = mem_gb
        self.ntasks = ntasks
        self.cpus = cpus
        self.nodes = nodes
        self.mail_user = mail_user
        self.mail_type = mail_type
        self.module_line = module_line


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Submit SLURM jobs for entangled N200 x_relaxed_AR*.txt packings."
    )
    ap.add_argument(
        "--job-name",
        type=str,
        default="entangled_N200",
        help="Job name prefix (and runs subfolder name).",
    )
    ap.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root folder containing N200 subfolders (default: repo/initial-configs/entangled_packings/N200).",
    )
    ap.add_argument(
        "--scene",
        type=Path,
        default=None,
        help="Base scene JSON (default: repo/assets/scenes/default_entangled.json).",
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Where to create run folders.",
    )
    ap.add_argument("--steps", type=int, default=200000, help="Headless steps.")
    ap.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Thread limit passed to --threads (0=auto).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate run folders and scripts but do not sbatch.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only submit the first N jobs (for quick tests).",
    )

    ap.add_argument(
        "--output-stride",
        type=int,
        default=1,
        help="Log output metrics every N frames.",
    )
    ap.add_argument(
        "--output-max",
        type=int,
        default=0,
        help="Max output frames to write (0=unlimited).",
    )
    ap.add_argument(
        "--perrod-stride",
        type=int,
        default=0,
        help="Log per-rod data every N frames (0=disabled).",
    )
    ap.add_argument(
        "--perrod-max",
        type=int,
        default=0,
        help="Max per-rod frames to write (0=unlimited).",
    )

    ap.add_argument(
        "--random-accel-sigma",
        type=float,
        default=0.0,
        help="Enable random acceleration (constant constant force) with this sigma. Disables random initial velocity.",
    )
    ap.add_argument(
        "--frictions",
        type=str,
        default=None,
        help="Comma-separated list of friction coefficients (e.g. '0.0,0.4,1.0'). Default: use scene value.",
    )
    ap.add_argument(
        "--init-velocity-sigma",
        type=float,
        default=None,
        help="Set random initial velocity sigma (Kick). Enables randomInit. Default: 0.1 in scene.",
    )

    # Metrics / diagnostics
    ap.add_argument(
        "--no-entanglement",
        action="store_true",
        help="Disable entanglement computation/logging.",
    )
    ap.add_argument(
        "--ent-period",
        type=int,
        default=60,
        help="Compute entanglement every N frames (passed via CLI).",
    )
    ap.add_argument(
        "--ent-threads",
        type=int,
        default=0,
        help="Threads for entanglement (0=auto).",
    )
    ap.add_argument(
        "--ent-cutoff",
        type=float,
        default=5.0,
        help="Distance cutoff for entanglement linking (passed via CLI).",
    )

    ap.add_argument(
        "--no-network",
        action="store_true",
        help="Disable contact network logging.",
    )
    ap.add_argument(
        "--network-stride",
        type=int,
        default=1,
        help="Log network every N frames.",
    )
    ap.add_argument(
        "--network-max",
        type=int,
        default=0,
        help="Max network frames to write (0=unlimited).",
    )
    ap.add_argument(
        "--network-emit-empty",
        action="store_true",
        help="Emit sentinel rows for frames with zero contacts.",
    )
    ap.add_argument(
        "--network-wave-period",
        type=int,
        default=0,
        help="Period for square wave logging (e.g. 1000).",
    )
    ap.add_argument(
        "--network-wave-width",
        type=int,
        default=0,
        help="Width of the square wave 'on' phase (e.g. 100).",
    )
    args = ap.parse_args()

    root_dir = find_root_dir()

    input_root = (
        args.input_root
        if args.input_root is not None
        else root_dir / "initial-configs" / "entangled_packings" / "N200"
    )
    if not input_root.is_dir():
        raise SystemExit(f"Input root not found: {input_root}")

    scene_src = (
        args.scene
        if args.scene is not None
        else root_dir / "assets" / "scenes" / "default_entangled.json"
    )
    if not scene_src.exists():
        raise SystemExit(f"Scene file not found: {scene_src}")

    binary_src = root_dir / "build" / "rigidbody_viewer_3d"
    ensure_executable(binary_src)

    slurm = SlurmCfg(cpus=max(1, int(args.threads)))

    runs_root = args.runs_root / args.job_name
    runs_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), runs_root / Path(__file__).name)

    timestamp = now_ts()

    jobs: List[Tuple[Path, int]] = sorted(
        iter_x_relaxed_files(input_root), key=lambda t: (t[0].parent.name, t[1])
    )
    if not jobs:
        raise SystemExit(f"No x_relaxed_AR*.txt files found under {input_root}")

    if args.limit > 0:
        jobs = jobs[: args.limit]

    print(f"Found {len(jobs)} x_relaxed files under {input_root}")
    print(f"Runs root: {runs_root}")

    # Load base scene once; we copy it per-run unchanged (user requested to run with it)
    base_scene = json.loads(scene_src.read_text())
    
    # If random acceleration is requested, disable random velocity initialization
    if args.random_accel_sigma > 0:
        print(f"Random acceleration enabled (sigma={args.random_accel_sigma}). Disabling random initial velocity.")
        if "scene" in base_scene and "randomInit" in base_scene["scene"]:
            base_scene["scene"]["randomInit"]["enabled"] = False
        # Also ensure randomForce (Brownian) is off if that was on (it defaults off usually)
        if "scene" in base_scene and "randomForce" in base_scene["scene"]:
             base_scene["scene"]["randomForce"]["enabled"] = False

    # If explicit initial velocity is requested
    if args.init_velocity_sigma is not None:
        if args.random_accel_sigma > 0:
             print("Warning: Both --random-accel-sigma and --init-velocity-sigma specified. Acceleration takes precedence for run dynamics, but init might be disabled.")
        else:
             print(f"Setting initial velocity sigma to {args.init_velocity_sigma}")
             # We will apply this per-run in the loop

    submitted = 0
    
    # Parse friction values
    friction_values = [None] # Default: use whatever is in scene
    if args.frictions is not None:
        friction_values = [float(f) for f in args.frictions.split(",") if f.strip()]
        print(f"Sweeping over frictions: {friction_values}")

    for x_path, ar in jobs:
        for friction in friction_values:
            seed_folder = x_path.parent.name
            x_name = x_path.name
            
            # Create run name
            suffix_parts = []
            if friction is not None:
                suffix_parts.append(f"Friction{friction}")
            
            if args.init_velocity_sigma is not None:
                suffix_parts.append(f"Kick{args.init_velocity_sigma}")

            suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
            run_name = safe_name(f"{timestamp}_{seed_folder}_AR{ar}{suffix}")
            
            run_dir = runs_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy binary
            shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")

            # Prepare scene
            scene_data = json.loads(json.dumps(base_scene)) # Deep copy
            
            # Apply friction override
            if friction is not None:
                if "physics" in scene_data and "soft_contact" in scene_data["physics"]:
                    scene_data["physics"]["soft_contact"]["mu"] = friction
                    scene_data["physics"]["soft_contact"]["mu_static"] = friction

            # Apply separate init velocity if requested
            if args.init_velocity_sigma is not None:
                 if "scene" not in scene_data: scene_data["scene"] = {}
                 if "randomInit" not in scene_data["scene"]:
                      scene_data["scene"]["randomInit"] = {"enabled": True, "wSpeed": 0.01, "seed": 42}
                 scene_data["scene"]["randomInit"]["enabled"] = True
                 scene_data["scene"]["randomInit"]["vSigma"] = args.init_velocity_sigma

            # Write scene.json
            (run_dir / "scene.json").write_text(json.dumps(scene_data, indent=2))

            # Symlink x_relaxed
            sym_x = run_dir / "x_relaxed.txt"
            if sym_x.exists():
                sym_x.unlink()
            sym_x.symlink_to(x_path)

            # Build command args
            sim_parts = [
                "./rigidbody_viewer_3d",
                "--headless",
                "--scene scene.json",
                f"--init-csv x_relaxed.txt",
                "--output output.csv",
                f"--steps {int(args.steps)}",
                f"--threads {int(args.threads)}",
            ]
            
            if args.random_accel_sigma > 0:
                sim_parts.append(f"--random-accel-sigma {args.random_accel_sigma}")
                
            if int(args.output_stride) > 1:
                sim_parts.append(f"--output-stride {int(args.output_stride)}")
            if int(args.output_max) > 0:
                sim_parts.append(f"--output-max {int(args.output_max)}")
                
            if int(args.perrod_stride) > 0:
                sim_parts.append(f"--perrod-stride {int(args.perrod_stride)}")
                sim_parts.append("--perrod perrod.csv")
                
            if int(args.perrod_max) > 0:
                sim_parts.append(f"--perrod-max {int(args.perrod_max)}")

            # Enable entanglement metrics in output.csv (ent_sum, ent_pairs)
            if not args.no_entanglement:
                sim_parts.append("--entanglement")
                sim_parts.append(f"--entanglement-period {int(args.ent_period)}")
                sim_parts.append(f"--entanglement-threads {int(args.ent_threads)}")
                sim_parts.append(f"--entanglement-cutoff {float(args.ent_cutoff)}")

            # Enable contact network logging (contains per-contact distance)
            if not args.no_network:
                sim_parts.append("--network network.csv")
                if int(args.network_stride) > 1:
                    sim_parts.append(f"--network-stride {int(args.network_stride)}")
                if int(args.network_max) > 0:
                    sim_parts.append(f"--network-max {int(args.network_max)}")
                if args.network_emit_empty:
                    sim_parts.append("--network-emit-empty")
                if int(args.network_wave_period) > 0 and int(args.network_wave_width) > 0:
                    sim_parts.append(f"--log-wave-period {int(args.network_wave_period)}")
                    sim_parts.append(f"--log-wave-width {int(args.network_wave_width)}")

            sim_cmd = " ".join(sim_parts)

            sb = f"""#!/bin/bash
#SBATCH -n {slurm.ntasks}
#SBATCH -c {slurm.cpus}
#SBATCH -N {slurm.nodes}
#SBATCH -t {slurm.time}
#SBATCH -p {slurm.partition}
#SBATCH --mem={slurm.mem_gb}G
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err
#SBATCH --mail-type={slurm.mail_type}
{f"#SBATCH --mail-user={slurm.mail_user}" if slurm.mail_user else ""}
#SBATCH --job-name={safe_name(args.job_name)}_{safe_name(seed_folder)}_AR{ar}

set -euo pipefail
{slurm.module_line}

echo "======================================"
echo "Entangled N200 dynamics"
echo "seed_folder: {seed_folder}"
echo "x_relaxed: {x_name}"
echo "AR: {ar}"
echo "PWD: $(pwd)"
echo "======================================"

echo "Running simulation..."
echo "{sim_cmd}"
{sim_cmd}

echo ""
echo "Job complete."
"""

            sbatch_path = run_dir / "Sbatch.sh"
            sbatch_path.write_text(sb)
            
            if not args.dry_run:
                print(f"Submitting {run_name}...")
                subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir, check=True)
                submitted += 1
            else:
                print(f"Dry run: Created {run_dir}")

    print(f"Submitted {submitted} jobs.")



def shlex_quote(s: str) -> str:
    # Minimal shell quoting (avoid importing shlex to keep script small)
    if s == "":
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_@%+=:,./\-]+", s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()
