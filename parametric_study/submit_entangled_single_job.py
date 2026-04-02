#!/usr/bin/env python3
"""Submit one Slurm job that runs many entangled-packings simulations.

This mirrors the run-folder and command generation used by
parametric_study/submit_entangled_array.py, but instead of creating a Slurm
array it writes one master sbatch script that executes every generated command
inside a single allocation.

Typical use:
  python3 parametric_study/submit_entangled_single_job.py \
      --n-rods 200 \
      --input-root initial-configs/relaxation_3rd_multithreading/N200 \
      --job-name debug_n200_single \
      --runs-root /path/to/runs \
      --frictions 0.0,0.05,0.1,0.15,0.2,0.4,1.0 \
    --sigma-v 0.1 \
    --sigma-w 0.2 \
      --seed-filter 945,12,381 \
      --ar 200 \
      --nsc --nsc-iters 40 --nsc-beta 0.2 --nsc-pos-iters 5 --nsc-pos-psor 50
"""

from __future__ import annotations

import argparse
import json
import math
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
    path = (Path.cwd() if start is None else start).resolve()
    for ancestor in [path, *path.parents]:
        if ancestor.name == target_name:
            return ancestor
    raise SystemExit(
        f"Could not find repository root named '{target_name}' starting from {path}"
    )


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_executable(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    if not os.access(path, os.X_OK):
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", value)


def shlex_quote(text: str) -> str:
    if text == "":
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_@%+=:,./\-]+", text):
        return text
    return "'" + text.replace("'", "'\\''") + "'"


_AR_RE = re.compile(r"x_relaxed_AR(\d+)\.txt$")


def iter_x_relaxed_files(root: Path) -> Iterable[Tuple[Path, int]]:
    for path in root.rglob("x_relaxed_AR*.txt"):
        match = _AR_RE.search(path.name)
        if match:
            yield path, int(match.group(1))


class SlurmCfg:
    def __init__(
        self,
        partition: str = "seas_compute",
        time: str = "3-00:00:00",
        mem_gb: int = 8,
        ntasks: int = 1,
        cpus: int = 8,
        nodes: int = 1,
        mail_user: str = os.environ.get("USER_EMAIL", ""),
        mail_type: str = "END",
        module_line: str = "module load python",
    ) -> None:
        self.partition = partition
        self.time = time
        self.mem_gb = mem_gb
        self.ntasks = ntasks
        self.cpus = cpus
        self.nodes = nodes
        self.mail_user = mail_user
        self.mail_type = mail_type
        self.module_line = module_line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit one Slurm job that runs many entangled rod simulations sequentially."
    )
    parser.add_argument("--n-rods", type=int, required=True)
    parser.add_argument("--ar", type=int, nargs="+", default=None)
    parser.add_argument("--job-name", type=str, default=None)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--max-parallel-subprocesses",
        type=int,
        default=1,
        help="Maximum number of simulation subprocesses to run concurrently inside the single Slurm job.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed-limit", type=int, default=0)
    parser.add_argument(
        "--seed-filter",
        type=str,
        action="append",
        default=None,
        help="Exact seed folder name to include. Repeat the flag for multiple seeds, e.g. --seed-filter '945,12,381'.",
    )
    parser.add_argument("--output-stride", type=int, default=1)
    parser.add_argument("--output-max", type=int, default=0)
    parser.add_argument("--perrod-stride", type=int, default=1000)
    parser.add_argument("--perrod-max", type=int, default=0)
    parser.add_argument("--random-accel-sigma", type=float, default=0.0)
    parser.add_argument("--frictions", type=str, default=None)
    parser.add_argument("--init-velocity-sigma", type=float, default=None)
    parser.add_argument("--sigma-v", type=float, default=None)
    parser.add_argument(
        "--sigma-w",
        type=float,
        default=None,
        help="Independent angular velocity sigma. When set with --sigma-v, randomInit mode='gaussian' is used.",
    )
    parser.add_argument("--no-entanglement", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--ent-period", type=int, default=60)
    parser.add_argument("--ent-threads", type=int, default=0)
    parser.add_argument("--ent-cutoff", type=float, default=1000.0)
    parser.add_argument("--w-speed", type=float, default=None)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--network-stride", type=int, default=1)
    parser.add_argument("--network-max", type=int, default=0)
    parser.add_argument("--network-emit-empty", action="store_true")
    parser.add_argument("--network-wave-period", type=int, default=0)
    parser.add_argument("--network-wave-width", type=int, default=0)
    parser.add_argument("--nsc", action="store_true")
    parser.add_argument("--nsc-iters", type=int, default=40)
    parser.add_argument("--nsc-beta", type=float, default=0.2)
    parser.add_argument("--nsc-cfm", type=float, default=0.0)
    parser.add_argument("--nsc-omega", type=float, default=1.0)
    parser.add_argument("--nsc-pos-iters", type=int, default=5)
    parser.add_argument("--nsc-pos-psor", type=int, default=50)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going if one subprocess fails inside the single Slurm allocation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.sigma_v is not None and (
        args.init_velocity_sigma is not None or args.w_speed is not None
    ):
        raise SystemExit(
            "--sigma-v is mutually exclusive with --init-velocity-sigma / --w-speed"
        )
    if args.sigma_w is not None and args.sigma_v is None:
        raise SystemExit("--sigma-w requires --sigma-v")
    if args.sigma_w is not None and (
        args.init_velocity_sigma is not None or args.w_speed is not None
    ):
        raise SystemExit(
            "--sigma-w is mutually exclusive with --init-velocity-sigma / --w-speed"
        )
    if args.max_parallel_subprocesses < 1:
        raise SystemExit("--max-parallel-subprocesses must be >= 1")

    root_dir = find_root_dir()
    input_root = (
        args.input_root
        if args.input_root is not None
        else root_dir / "initial-configs" / "entangled_packings" / f"N{args.n_rods}"
    )
    if not input_root.is_dir():
        raise SystemExit(f"Input root not found: {input_root}")

    if args.scene is not None:
        scene_src = args.scene
    elif args.nsc:
        scene_src = root_dir / "assets" / "scenes" / "default_entangled_nsc.json"
    else:
        scene_src = root_dir / "assets" / "scenes" / "default_entangled.json"
    if not scene_src.exists():
        raise SystemExit(f"Scene file not found: {scene_src}")

    if args.use_cuda:
        binary_src = root_dir / "build_cuda" / "rigidbody_viewer_3d"
        slurm = SlurmCfg(
            partition="gpu",
            time="0-12:00:00",
            mem_gb=8,
            ntasks=1,
            cpus=max(1, int(args.threads)) * int(args.max_parallel_subprocesses),
            nodes=1,
            module_line="module load gcc/13.2.0-fasrc01\nmodule load cuda/12.9.1-fasrc01",
        )
    else:
        binary_src = root_dir / "build_head" / "rigidbody_viewer_3d"
        slurm = SlurmCfg(
            cpus=max(1, int(args.threads)) * int(args.max_parallel_subprocesses)
        )
    ensure_executable(binary_src)

    job_name = args.job_name if args.job_name else f"entangled_single_N{args.n_rods}"
    runs_root = args.runs_root / job_name
    runs_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), runs_root / Path(__file__).name)

    timestamp = now_ts()
    jobs: List[Tuple[Path, int]] = sorted(
        iter_x_relaxed_files(input_root), key=lambda item: (item[0].parent.name, item[1])
    )
    if not jobs:
        raise SystemExit(f"No x_relaxed_AR*.txt files found under {input_root}")

    if args.seed_limit > 0:
        unique_seeds = []
        seen = set()
        for path, _ in jobs:
            seed_name = path.parent.name
            if seed_name not in seen:
                unique_seeds.append(seed_name)
                seen.add(seed_name)
        allowed = set(unique_seeds[: args.seed_limit])
        jobs = [job for job in jobs if job[0].parent.name in allowed]

    if args.seed_filter is not None:
        allowed = {seed for seed in args.seed_filter if seed}
        jobs = [job for job in jobs if job[0].parent.name in allowed]

    if args.limit > 0:
        jobs = jobs[: args.limit]

    if args.ar is not None:
        allowed_ars = set(args.ar)
        jobs = [job for job in jobs if job[1] in allowed_ars]

    if not jobs:
        raise SystemExit("No runs matched the requested filters.")

    print(f"Found {len(jobs)} x_relaxed files under {input_root}")
    print(f"Runs root: {runs_root}")

    base_scene = json.loads(scene_src.read_text())
    if "scene" in base_scene and "populate" in base_scene["scene"]:
        base_scene["scene"]["populate"]["count"] = args.n_rods

    friction_values = [None]
    if args.frictions is not None:
        friction_values = [float(value) for value in args.frictions.split(",") if value.strip()]

    commands: List[str] = []
    skipped = 0

    for x_path, ar in jobs:
        for friction in friction_values:
            seed_folder = x_path.parent.name
            suffix_parts = []
            if friction is not None:
                suffix_parts.append(f"Friction{friction}")
            if args.sigma_v is not None:
                suffix_parts.append(f"SigV{args.sigma_v}")
            if args.sigma_w is not None:
                suffix_parts.append(f"SigW{args.sigma_w}")
            elif args.init_velocity_sigma is not None:
                suffix_parts.append(f"Kick{args.init_velocity_sigma}")
            suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
            run_name = safe_name(f"{timestamp}_{seed_folder}_AR{ar}{suffix}")
            run_dir = runs_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            if (run_dir / "output.csv").exists():
                skipped += 1
                continue

            shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")
            scene_data = json.loads(json.dumps(base_scene))

            if friction is not None:
                if args.nsc:
                    scene_data.setdefault("physics", {}).setdefault("nsc", {})["mu"] = friction
                else:
                    soft_contact = scene_data.setdefault("physics", {}).setdefault("soft_contact", {})
                    soft_contact["mu"] = friction
                    soft_contact["mu_static"] = friction

            if args.sigma_v is not None:
                if args.sigma_w is not None:
                    scene_data.setdefault("scene", {})["randomInit"] = {
                        "enabled": True,
                        "mode": "gaussian",
                        "vSigma": args.sigma_v,
                        "wSigma": args.sigma_w,
                        "seed": 42,
                        "projectParallelSpin": True,
                    }
                else:
                    rod_length = 1.0
                    rod_diameter = rod_length / ar
                    rod_radius = rod_diameter / 2.0
                    rod_density = 1000.0
                    if "scene" in scene_data:
                        if "populate" in scene_data["scene"]:
                            rod_density = scene_data["scene"]["populate"].get("density", rod_density)
                        elif scene_data["scene"].get("bodies"):
                            rod_density = scene_data["scene"]["bodies"][0].get("density", rod_density)
                    rod_mass = rod_density * math.pi * rod_radius**2 * rod_length
                    kbt = rod_mass * args.sigma_v**2
                    scene_data.setdefault("scene", {})["randomInit"] = {
                        "enabled": True,
                        "mode": "thermal",
                        "kBT": kbt,
                        "seed": 42,
                        "projectParallelSpin": True,
                    }
            elif args.init_velocity_sigma is not None:
                random_init = scene_data.setdefault("scene", {}).setdefault(
                    "randomInit", {"enabled": True, "wSpeed": 0.01, "seed": 42}
                )
                random_init["enabled"] = True
                random_init["vSigma"] = args.init_velocity_sigma

            if args.w_speed is not None and args.sigma_v is None:
                random_init = scene_data.setdefault("scene", {}).setdefault(
                    "randomInit", {"enabled": True, "vSigma": 0.0, "seed": 42}
                )
                random_init["wSpeed"] = args.w_speed

            if args.random_accel_sigma > 0:
                scene_data.setdefault("scene", {}).setdefault("randomInit", {})["enabled"] = False
                scene_data.setdefault("scene", {}).setdefault("randomForce", {})["enabled"] = False

            if args.use_cuda:
                scene_data.setdefault("physics", {}).setdefault("soft_contact", {})["use_cuda"] = True

            if args.delta is not None:
                scene_data.setdefault("physics", {}).setdefault("soft_contact", {})["delta"] = args.delta

            dt_val = args.dt if args.dt is not None else base_scene["physics"]["dt"]
            if args.dt is not None:
                scene_data["physics"]["dt"] = args.dt

            (run_dir / "scene.json").write_text(json.dumps(scene_data, indent=2))

            sym_x = run_dir / "x_relaxed.txt"
            if sym_x.exists() or sym_x.is_symlink():
                sym_x.unlink()
            sym_x.symlink_to(x_path)

            sim_parts = [
                "./rigidbody_viewer_3d",
                "--headless",
                "--scene scene.json",
                "--init-csv x_relaxed.txt",
                "--output output.csv",
                f"--steps {int(args.steps)}",
                f"--dt {dt_val}",
                f"--threads {int(args.threads)}",
            ]

            if args.nsc:
                sim_parts.append("--nsc")
                if friction is not None:
                    sim_parts.append(f"--nsc-mu {friction}")
                sim_parts.append(f"--nsc-iters {args.nsc_iters}")
                sim_parts.append(f"--nsc-beta {args.nsc_beta}")
                if args.nsc_cfm != 0.0:
                    sim_parts.append(f"--nsc-cfm {args.nsc_cfm}")
                if args.nsc_omega != 1.0:
                    sim_parts.append(f"--nsc-omega {args.nsc_omega}")
                sim_parts.append(f"--nsc-pos-iters {args.nsc_pos_iters}")
                sim_parts.append(f"--nsc-pos-psor {args.nsc_pos_psor}")

            if args.random_accel_sigma > 0:
                sim_parts.append(f"--random-accel-sigma {args.random_accel_sigma}")
            if args.no_csv:
                sim_parts.append("--no-csv")
            if int(args.output_stride) > 1:
                sim_parts.append(f"--output-stride {int(args.output_stride)}")
            if int(args.output_max) > 0:
                sim_parts.append(f"--output-max {int(args.output_max)}")
            if int(args.perrod_stride) > 0:
                sim_parts.append(f"--perrod-stride {int(args.perrod_stride)}")
                sim_parts.append("--perrod perrod.csv")
            if int(args.perrod_max) > 0:
                sim_parts.append(f"--perrod-max {int(args.perrod_max)}")
            if not args.no_entanglement:
                sim_parts.append("--entanglement")
                sim_parts.append(f"--entanglement-period {int(args.ent_period)}")
                sim_parts.append(f"--entanglement-threads {int(args.ent_threads)}")
                sim_parts.append(f"--entanglement-cutoff {float(args.ent_cutoff)}")
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
            full_cmd = f"cd {shlex_quote(str(run_dir.absolute()))} && {sim_cmd}"
            commands.append(full_cmd)

    if not commands:
        print(f"No new jobs to run. Skipped {skipped} existing runs.")
        return

    commands_file = runs_root / "single_job_commands.txt"
    commands_file.write_text("\n".join(commands) + "\n")

    runner_lines = [
        "#!/bin/bash",
        f"#SBATCH -n {slurm.ntasks}",
        f"#SBATCH -c {slurm.cpus}",
        f"#SBATCH -N {slurm.nodes}",
        f"#SBATCH -t {slurm.time}",
        f"#SBATCH -p {slurm.partition}",
        f"#SBATCH --mem={slurm.mem_gb}G",
        f"#SBATCH -o {runs_root.absolute()}/output_%j.out",
        f"#SBATCH -e {runs_root.absolute()}/errors_%j.err",
        f"#SBATCH --mail-type={slurm.mail_type}",
    ]
    if args.use_cuda:
        runner_lines.append("#SBATCH --gres=gpu:1")
    if slurm.mail_user:
        runner_lines.append(f"#SBATCH --mail-user={slurm.mail_user}")
    runner_lines.append(f"#SBATCH --job-name={safe_name(job_name)}_single")
    runner_lines.extend(
        [
            "",
            "set -euo pipefail",
            slurm.module_line,
            f"CMDS_FILE={shlex_quote(str(commands_file.absolute()))}",
            f"TOTAL=$(wc -l < {shlex_quote(str(commands_file.absolute()))})",
            f"MAX_PARALLEL={int(args.max_parallel_subprocesses)}",
            'echo "Running single bundled job with $TOTAL subprocesses"',
            'echo "Threads per subprocess: ' + str(int(args.threads)) + '; max concurrent subprocesses: $MAX_PARALLEL; allocated cpus: ' + str(slurm.cpus) + '"',
            'task_index=0',
            'active_jobs=0',
            'job_failed=0',
            'while IFS= read -r CMD || [[ -n "$CMD" ]]; do',
            '  task_index=$((task_index + 1))',
            '  [[ -z "$CMD" ]] && continue',
            '  echo "[$task_index/$TOTAL] $CMD"',
        ]
    )
    if args.max_parallel_subprocesses == 1:
        if args.continue_on_error:
            runner_lines.extend(
                [
                    '  if ! eval "$CMD"; then',
                    '    echo "Command failed but continuing: $CMD" >&2',
                    '  fi',
                ]
            )
        else:
            runner_lines.append('  eval "$CMD"')
    else:
        runner_lines.extend(
            [
                '  bash -lc "$CMD" &',
                '  active_jobs=$((active_jobs + 1))',
                '  if (( active_jobs >= MAX_PARALLEL )); then',
                '    if ! wait -n; then',
                '      job_failed=1',
            ]
        )
        if args.continue_on_error:
            runner_lines.extend(
                [
                    '      echo "Command failed but continuing." >&2',
                    '    fi',
                    '    active_jobs=$((active_jobs - 1))',
                    '  fi',
                ]
            )
        else:
            runner_lines.extend(
                [
                    '      echo "A subprocess failed; stopping further launches." >&2',
                    '      break',
                    '    fi',
                    '    active_jobs=$((active_jobs - 1))',
                    '  fi',
                ]
            )
    runner_lines.extend(
        [
            'done < "$CMDS_FILE"',
            'while (( active_jobs > 0 )); do',
            '  if ! wait -n; then',
            '    job_failed=1',
            '  fi',
            '  active_jobs=$((active_jobs - 1))',
            'done',
            'echo "Bundled job complete."',
        ]
    )
    if not args.continue_on_error:
        runner_lines[-1:-1] = [
            'if (( job_failed != 0 )); then',
            '  exit 1',
            'fi',
        ]

    master_sbatch_path = runs_root / "Master_Single_Sbatch.sh"
    master_sbatch_path.write_text("\n".join(runner_lines) + "\n")
    os.chmod(master_sbatch_path, os.stat(master_sbatch_path).st_mode | stat.S_IXUSR)

    if args.dry_run:
        print(
            f"Dry run: Created {master_sbatch_path} with {len(commands)} subprocess commands. "
            f"Skipped {skipped} existing runs."
        )
        return

    print(
        f"Submitting single bundled job for N={args.n_rods} with {len(commands)} subprocess commands..."
    )
    result = subprocess.run(["sbatch", master_sbatch_path.name], cwd=runs_root)
    if result.returncode != 0:
        raise SystemExit("sbatch failed for bundled single job.")
    print(f"Skipped {skipped} existing runs.")


if __name__ == "__main__":
    main()