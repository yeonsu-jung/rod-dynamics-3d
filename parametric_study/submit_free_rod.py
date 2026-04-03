#!/usr/bin/env python3
"""submit_free_rod.py

Submit SLURM jobs for single-free-rod perturbation experiments.

Output format is endpoint-only trajectory for the free rod using:
  --test-rod-endpoints

Modes:
    Default (one job per realization, all frictions in sequence):
        run_dir/ rigidbody_viewer_3d, scene_mu{f}.json x frictions, x_relaxed.txt,
        Sbatch.sh
        outputs: free_rod_endpoints_mu*.csv

    Separate-frictions (one job per friction):
    run_dir/ rigidbody_viewer_3d, scene.json, x_relaxed.txt, Sbatch.sh
    output: free_rod_endpoints.csv

  Bundle-all (one job for all filtered entries x frictions):
    run_dir/ rigidbody_viewer_3d, scene_N{N}_mu{f}.json x needed files, Sbatch.sh
    outputs: endpoints_N*_AR*_..._mu*.csv

Input CSV columns: N,AR,ID,Metric,RodIndex,Value,FilePath
ID uses underscores (278_868_121), while folder uses commas (278,868,121).
Input files: <input-root>/N{N}/{ID with commas}/x_relaxed_AR{AR}.txt
"""

import argparse
import concurrent.futures
import csv
import json
import os
import re
import shlex
import shutil
import stat
import subprocess

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


DEFAULT_RUNS_ROOT = Path(
    "/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs"
    if Path("/n/holylabs").exists()
    else str(Path(__file__).resolve().parent.parent / "runs")
)
_LOCAL_INPUT_BASE = Path(__file__).resolve().parent.parent / "initial-configs" / "relaxation_3rd_multithreading"
DEFAULT_INPUT_BASE = Path(
    "/n/home01/yjung/Github/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading"
    if Path("/n/home01").exists()
    else str(
        _LOCAL_INPUT_BASE
        if _LOCAL_INPUT_BASE.exists()
        else Path.home() / "Github" / "entanglement-optimization-cpp" / "examples" / "relaxation_3rd_multithreading"
    )
)


def find_root_dir(start: Optional[Path] = None, target_name: str = "rod-dynamics-3d") -> Path:
    p = (Path.cwd() if start is None else start).resolve()
    for ancestor in [p, *p.parents]:
        if ancestor.name == target_name:
            return ancestor
    raise SystemExit(f"Could not find repo root '{target_name}' from {p}")


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_executable(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Binary not found: {path}")
    if not os.access(path, os.X_OK):
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", s)


def mu_file_tag(mu: Optional[float]) -> str:
    if mu is None:
        return "default"
    return str(mu).replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class LocalRunTask:
    label: str
    cmd: List[str]
    cwd: Path


def make_scene(base: dict, N: int, friction: Optional[float],
               w_speed: float, v_sigma: Optional[float],
               use_cuda: bool, delta: Optional[float], dt: Optional[float],
               use_nsc: bool = False, sigma_v: Optional[float] = None,
               ar: Optional[int] = None) -> dict:
    d = json.loads(json.dumps(base))
    d.setdefault("scene", {}).setdefault("populate", {})["count"] = N

    # Thermal mode: compute kBT from sigma_v and rod mass
    if sigma_v is not None and ar is not None:
        import math
        rod_length = 1.0
        rod_diameter = rod_length / ar
        rod_radius = rod_diameter / 2.0
        rod_density = 2500.0
        if "scene" in d:
            if "populate" in d["scene"]:
                rod_density = d["scene"]["populate"].get("density", rod_density)
            elif "bodies" in d["scene"] and d["scene"]["bodies"]:
                rod_density = d["scene"]["bodies"][0].get("density", rod_density)
        rod_mass = rod_density * math.pi * rod_radius**2 * rod_length
        kBT = rod_mass * sigma_v**2
        d["scene"]["randomInit"] = {
            "enabled": True,
            "mode": "thermal",
            "kBT": kBT,
            "seed": 42,
            "projectParallelSpin": True,
        }
    else:
        # Legacy mode
        ri = d["scene"].setdefault(
            "randomInit", {"enabled": True, "vSigma": 0.0, "wSpeed": 0.01, "seed": 42}
        )
        ri["enabled"] = True
        ri["wSpeed"] = w_speed
        if v_sigma is not None:
            ri["vSigma"] = v_sigma

    if friction is not None:
        if use_nsc:
            nsc = d.setdefault("physics", {}).setdefault("nsc", {})
            nsc["mu"] = friction
        else:
            sc = d.setdefault("physics", {}).setdefault("soft_contact", {})
            sc["mu"] = friction
            sc["mu_static"] = friction
    if use_cuda:
        d.setdefault("physics", {}).setdefault("soft_contact", {})["use_cuda"] = True
    if delta is not None:
        d.setdefault("physics", {}).setdefault("soft_contact", {})["delta"] = delta
    if dt is not None:
        d["physics"]["dt"] = dt
    return d


class SlurmCfg:
    def __init__(self, partition="seas_compute", time="0-01:00:00", mem_gb=1,
                 ntasks=1, cpus=4, nodes=1,
                 mail_user=os.environ.get("USER_EMAIL", ""),
                 mail_type="END", module_line=""):
        self.partition = partition
        self.time = time
        self.mem_gb = mem_gb
        self.ntasks = ntasks
        self.cpus = cpus
        self.nodes = nodes
        self.mail_user = mail_user
        self.mail_type = mail_type
        self.module_line = module_line


def load_extreme_rods_csv(csv_path: Path) -> List[dict]:
    entries = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            entries.append({
                "N": int(row["N"]),
                "AR": int(row["AR"]),
                "id": row["ID"],
                "seed": row["ID"].replace("_", ","),
                "metric": row["Metric"],
                "free_rod": int(row["RodIndex"]),
                "value": float(row["Value"]),
                "file_path": row.get("FilePath", "").strip() or None,
            })
    return entries


def _candidate_relative_input_path(file_path: Path) -> Optional[Path]:
    parts = file_path.parts
    marker = ("examples", "relaxation_3rd_multithreading")
    for idx in range(len(parts) - len(marker) + 1):
        if parts[idx:idx + len(marker)] == marker:
            return Path(*parts[idx + len(marker):])
    return None


def resolve_input_path(entry: dict, input_roots: List[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    file_path_value = entry.get("file_path")
    if file_path_value:
        csv_path = Path(file_path_value)
        candidates.append(csv_path)
        rel = _candidate_relative_input_path(csv_path)
        if rel is not None:
            candidates.extend(root / rel for root in input_roots)

    candidates.extend(
        root / f"N{entry['N']}" / entry["seed"] / f"x_relaxed_AR{entry['AR']}.txt"
        for root in input_roots
    )

    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Sbatch builders
# ---------------------------------------------------------------------------

def _header(slurm: SlurmCfg, job_name: str, use_cuda: bool) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH -n {slurm.ntasks}",
        f"#SBATCH -c {slurm.cpus}",
        f"#SBATCH -N {slurm.nodes}",
        f"#SBATCH -t {slurm.time}",
        f"#SBATCH -p {slurm.partition}",
        f"#SBATCH --mem={slurm.mem_gb}G",
    ]
    if use_cuda:
        lines.append("#SBATCH --gres=gpu:1")
    lines += [
        "#SBATCH -o output_%j.out",
        "#SBATCH -e errors_%j.err",
        f"#SBATCH --mail-type={slurm.mail_type}",
    ]
    if slurm.mail_user:
        lines.append(f"#SBATCH --mail-user={slurm.mail_user}")
    lines += [f"#SBATCH --job-name={job_name}", "", "set -euo pipefail"]
    if slurm.module_line:
        lines.append(slurm.module_line)
    return "\n".join(lines)


def _build_sim_args(scene_expr: str, init_expr: str, frames: int, dt_val: float,
                    cpus: int, free_rod: int, out_expr: str,
                    endpoint_stride: int, endpoint_max: Optional[int],
                    stop_ke_threshold: Optional[float],
                    stop_ke_min_steps: Optional[int],
                    stop_slide_vel_threshold: Optional[float],
                    stop_slide_vel_min_steps: Optional[int],
                    nsc_args: Optional[dict] = None) -> List[str]:
    pieces = [
        "./rigidbody_viewer_3d",
        "--headless",
        "--scene", scene_expr,
        "--init-csv", init_expr,
        "--steps", str(frames),
        "--dt", str(dt_val),
        "--threads", str(cpus),
        "--fix-every-except", str(free_rod),
        "--test-rod-endpoints", out_expr,
    ]
    if endpoint_stride is not None and endpoint_stride > 0:
        pieces.extend(["--test-rod-endpoints-stride", str(endpoint_stride)])
    if endpoint_max is not None and endpoint_max > 0:
        pieces.extend(["--test-rod-endpoints-max", str(endpoint_max)])
    if stop_ke_threshold is not None and stop_ke_threshold > 0:
        pieces.extend(["--stop-ke-threshold", str(stop_ke_threshold)])
    if stop_ke_min_steps is not None and stop_ke_min_steps > 0:
        pieces.extend(["--stop-ke-min-steps", str(stop_ke_min_steps)])
    if stop_slide_vel_threshold is not None and stop_slide_vel_threshold > 0:
        pieces.extend(["--stop-slide-vel-threshold", str(stop_slide_vel_threshold)])
    if stop_slide_vel_min_steps is not None and stop_slide_vel_min_steps > 0:
        pieces.extend(["--stop-slide-vel-min-steps", str(stop_slide_vel_min_steps)])
    if nsc_args is not None:
        pieces.append("--nsc")
        pieces.extend(["--nsc-iters", str(nsc_args['iters'])])
        pieces.extend(["--nsc-beta", str(nsc_args['beta'])])
        if nsc_args.get('cfm', 0.0) != 0.0:
            pieces.extend(["--nsc-cfm", str(nsc_args['cfm'])])
        if nsc_args.get('omega', 1.0) != 1.0:
            pieces.extend(["--nsc-omega", str(nsc_args['omega'])])
        pieces.extend(["--nsc-pos-iters", str(nsc_args['pos_iters'])])
        pieces.extend(["--nsc-pos-psor", str(nsc_args['pos_psor'])])
    return pieces


def _build_sim_cmd(scene_expr: str, init_expr: str, frames: int, dt_val: float,
                   cpus: int, free_rod: int, out_expr: str,
                   endpoint_stride: int, endpoint_max: Optional[int],
                   stop_ke_threshold: Optional[float],
                   stop_ke_min_steps: Optional[int],
                   stop_slide_vel_threshold: Optional[float],
                   stop_slide_vel_min_steps: Optional[int],
                   nsc_args: Optional[dict] = None) -> str:
    pieces = [
        "./rigidbody_viewer_3d --headless",
        f"--scene {scene_expr}",
        f"--init-csv {init_expr}",
        f"--steps {frames} --dt {dt_val} --threads {cpus}",
        f"--fix-every-except {free_rod}",
        f"--test-rod-endpoints {out_expr}",
    ]
    if endpoint_stride is not None and endpoint_stride > 0:
        pieces.append(f"--test-rod-endpoints-stride {endpoint_stride}")
    if endpoint_max is not None and endpoint_max > 0:
        pieces.append(f"--test-rod-endpoints-max {endpoint_max}")
    if stop_ke_threshold is not None and stop_ke_threshold > 0:
        pieces.append(f"--stop-ke-threshold {stop_ke_threshold}")
    if stop_ke_min_steps is not None and stop_ke_min_steps > 0:
        pieces.append(f"--stop-ke-min-steps {stop_ke_min_steps}")
    if stop_slide_vel_threshold is not None and stop_slide_vel_threshold > 0:
        pieces.append(f"--stop-slide-vel-threshold {stop_slide_vel_threshold}")
    if stop_slide_vel_min_steps is not None and stop_slide_vel_min_steps > 0:
        pieces.append(f"--stop-slide-vel-min-steps {stop_slide_vel_min_steps}")
    if nsc_args is not None:
        pieces.append("--nsc")
        pieces.append(f"--nsc-iters {nsc_args['iters']}")
        pieces.append(f"--nsc-beta {nsc_args['beta']}")
        if nsc_args.get('cfm', 0.0) != 0.0:
            pieces.append(f"--nsc-cfm {nsc_args['cfm']}")
        if nsc_args.get('omega', 1.0) != 1.0:
            pieces.append(f"--nsc-omega {nsc_args['omega']}")
        pieces.append(f"--nsc-pos-iters {nsc_args['pos_iters']}")
        pieces.append(f"--nsc-pos-psor {nsc_args['pos_psor']}")
    return " ".join(pieces)


def run_local_tasks(tasks: List[LocalRunTask], workers: int) -> tuple[int, int]:
    if not tasks:
        return 0, 0

    submitted = 0
    failed = 0

    def _run(task: LocalRunTask) -> tuple[LocalRunTask, int]:
        result = subprocess.run(task.cmd, cwd=task.cwd, check=False)
        return task, result.returncode

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_run, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            task, returncode = future.result()
            if returncode != 0:
                print(f"FAILED {task.label} (exit {returncode})")
                failed += 1
            else:
                print(f"DONE {task.label}")
                submitted += 1
    return submitted, failed


def build_sbatch_separate(slurm, job_name, use_cuda, N, ar, seed, metric,
                          free_rod, friction, frames, dt_val,
                          endpoint_stride, endpoint_max,
                          stop_ke_threshold, stop_ke_min_steps,
                          stop_slide_vel_threshold, stop_slide_vel_min_steps,
                          nsc_args=None):
    mu_str = f"mu={friction}" if friction is not None else "mu=scene"
    out_file = "free_rod_endpoints.csv"
    sim_cmd = _build_sim_cmd(
        "scene.json", "x_relaxed.txt", frames, dt_val, slurm.cpus,
        free_rod, out_file, endpoint_stride, endpoint_max,
        stop_ke_threshold, stop_ke_min_steps,
        stop_slide_vel_threshold, stop_slide_vel_min_steps,
        nsc_args=nsc_args
    )
    return f"""{_header(slurm, job_name, use_cuda)}

echo "Free-rod: N={N} AR={ar} seed={seed} metric={metric} rod={free_rod} {mu_str} frames={frames}"

{sim_cmd}

echo "Rows: $(grep -c '^[0-9]' {out_file} || true)"
echo "Job complete."
"""


def build_sbatch_bundle(slurm, job_name, use_cuda, resolved_entries,
                        friction_values, frames, dt_val,
                        endpoint_stride, endpoint_max,
                        stop_ke_threshold, stop_ke_min_steps,
                        stop_slide_vel_threshold, stop_slide_vel_min_steps,
                        nsc_args=None):
    """One SLURM job that runs every entry x every friction sequentially."""
    mu_list = " ".join(str(f) for f in friction_values)
    sim_cmd = _build_sim_cmd(
        '"scene_N${N}_AR${AR}_mu${MU}.json"',
        '"$X_PATH"',
        frames,
        dt_val,
        slurm.cpus,
        '${FREE_ROD}',
        '"$OUT_FILE"',
        endpoint_stride,
        endpoint_max,
        stop_ke_threshold,
        stop_ke_min_steps,
        stop_slide_vel_threshold,
        stop_slide_vel_min_steps,
        nsc_args=nsc_args,
    )

    inner = (
        f"    for MU in {mu_list}; do\n"
        f'        echo "[$(date +%H:%M:%S)] N=$N AR=$AR $SEED_ID $METRIC rod=$FREE_ROD mu=$MU"\n'
        f'        MU_TAG=${{MU//./p}}\n'
        f'        OUT_FILE="endpoints_N${{N}}_AR${{AR}}_${{SEED_ID}}_${{METRIC}}_rod${{FREE_ROD}}_mu${{MU_TAG}}.csv"\n'
        f"        {sim_cmd}\n"
        f'        echo "  rows: $(grep -c \'^[0-9]\' \"$OUT_FILE\" || true) ($OUT_FILE)"\n'
        f"    done"
    )
    calls = "\n".join(
        f'run_one {e["N"]} {e["AR"]} {e["id"]} {e["metric"]} {e["free_rod"]} "{e["x_path"]}"'
        for e in resolved_entries
    )
    n_runs = len(resolved_entries) * len(friction_values)
    return f"""{_header(slurm, job_name, use_cuda)}

echo "Bundle: {len(resolved_entries)} entries x {len(friction_values)} frictions = {n_runs} runs"
echo "Frames={frames} dt={dt_val} threads={slurm.cpus}"

run_one() {{
    local N=$1 AR=$2 SEED_ID=$3 METRIC=$4 FREE_ROD=$5 X_PATH=$6
{inner}
}}

{calls}

echo "Bundle complete."
"""


def build_sbatch_combined(slurm, job_name, use_cuda, N, ar, seed, metric,
                          free_rod, friction_values, frames, dt_val,
                          endpoint_stride, endpoint_max,
                          stop_ke_threshold, stop_ke_min_steps,
                          stop_slide_vel_threshold, stop_slide_vel_min_steps,
                          nsc_args=None):
    mu_list = " ".join(str(f) for f in friction_values)
    sim_cmd = _build_sim_cmd(
        '"scene_mu${MU}.json"',
        "x_relaxed.txt",
        frames,
        dt_val,
        slurm.cpus,
        free_rod,
        '"$OUT_FILE"',
        endpoint_stride,
        endpoint_max,
        stop_ke_threshold,
        stop_ke_min_steps,
        stop_slide_vel_threshold,
        stop_slide_vel_min_steps,
        nsc_args=nsc_args,
    )
    return f"""{_header(slurm, job_name, use_cuda)}

echo "Free-rod combined: N={N} AR={ar} seed={seed} metric={metric} rod={free_rod}"
echo "Frictions: {mu_list} frames={frames} dt={dt_val}"

for MU in {mu_list}; do
    echo "  --- mu=$MU ---"
    MU_TAG=${{MU//./p}}
    OUT_FILE="free_rod_endpoints_mu${{MU_TAG}}.csv"
    {sim_cmd}
    echo "  rows: $(grep -c '^[0-9]' \"$OUT_FILE\" || true) ($OUT_FILE)"
done

echo "Job complete."
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Submit single-free-rod SLURM jobs with endpoint-only trajectory output."
    )
    ap.add_argument("--extreme-rods-csv", type=Path, required=True)
    ap.add_argument("--input-root", type=Path, nargs="+", default=[DEFAULT_INPUT_BASE])
    ap.add_argument("--job-name", type=str, default="free_rod")
    ap.add_argument("--scene", type=Path, default=None)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--stop-ke-threshold", type=float, default=None,
                    help="Stop headless run early when total KE drops below this threshold.")
    ap.add_argument("--stop-ke-min-steps", type=int, default=0,
                    help="Minimum steps before KE-based stop check is allowed.")
    ap.add_argument("--stop-slide-vel-threshold", type=float, default=None,
                    help="Stop early when |v dot rod_axis| falls below this threshold.")
    ap.add_argument("--stop-slide-vel-min-steps", type=int, default=0,
                    help="Minimum steps before sliding-velocity stop check is allowed.")
    ap.add_argument("--endpoint-stride", type=int, default=-1,
                    help="Sample every N frames for endpoint output. Use <=0 to let the app auto-compute stride from --endpoint-max.")
    ap.add_argument("--endpoint-max", type=int, default=300,
                    help="Max sampled endpoint frames per run. Use <=0 for unlimited.")
    ap.add_argument("--frictions", type=str, default=None,
                    help="Comma-separated friction values.")
    ap.add_argument("--init-velocity-sigma", type=float, default=None,
                    help="Legacy: set randomInit.vSigma (uniform mode).")
    ap.add_argument("--sigma-v", type=float, default=None,
                    help="Thermal randomInit: translational velocity scale. "
                         "Computes kBT = m * sigma_v^2 per AR. "
                         "Mutually exclusive with --init-velocity-sigma / --w-speed.")
    ap.add_argument("--w-speed", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--use-cuda", action="store_true")
    ap.add_argument("--delta", type=float, default=None)

    # NSC (hard contact) solver arguments
    ap.add_argument("--nsc", action="store_true",
                    help="Use NSC (impulse-based) contact solver instead of soft contact.")
    ap.add_argument("--nsc-iters", type=int, default=40)
    ap.add_argument("--nsc-beta", type=float, default=0.2)
    ap.add_argument("--nsc-cfm", type=float, default=0.0)
    ap.add_argument("--nsc-omega", type=float, default=1.0)
    ap.add_argument("--nsc-pos-iters", type=int, default=5)
    ap.add_argument("--nsc-pos-psor", type=int, default=50)
    ap.add_argument("--combine-frictions", action="store_true",
                    help="Deprecated alias; combined-friction mode is now the default.")
    ap.add_argument("--separate-frictions", action="store_true",
                    help="Submit one job per friction value (old behavior).")
    ap.add_argument("--bundle-all", action="store_true",
                    help="Bundle all filtered entries into one SLURM job.")
    ap.add_argument("--time", type=str, default=None,
                    help="SLURM time limit (e.g. '2-00:00:00').")
    ap.add_argument("--filter-n", type=int, nargs="+", default=None)
    ap.add_argument("--filter-ar", type=int, nargs="+", default=None)
    ap.add_argument("--filter-alpha", type=int, nargs="+", default=None,
                    help="Alias of --filter-ar.")
    ap.add_argument("--filter-id", type=str, nargs="+", default=None)
    ap.add_argument("--filter-metric", type=str, nargs="+", default=None,
                    choices=["MinFSA", "MaxFSA", "MinFTA", "MaxFTA"])
    ap.add_argument("--local", action="store_true",
                    help="Run simulations locally (no SLURM). Executes each run directly.")
    ap.add_argument("--local-workers", type=int, default=1,
                    help="Number of concurrent subprocesses to use with --local.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root_dir = find_root_dir()

    entries = load_extreme_rods_csv(args.extreme_rods_csv)
    if not entries:
        raise SystemExit(f"No rows in {args.extreme_rods_csv}")

    if args.filter_n:
        entries = [e for e in entries if e["N"] in args.filter_n]
    filter_ar = args.filter_ar
    if args.filter_alpha:
        filter_ar = (filter_ar or []) + args.filter_alpha
    if filter_ar:
        allowed_ar = set(filter_ar)
        entries = [e for e in entries if e["AR"] in allowed_ar]
    if args.filter_id:
        entries = [e for e in entries if e["id"] in args.filter_id]
    if args.filter_metric:
        entries = [e for e in entries if e["metric"] in args.filter_metric]
    if not entries:
        raise SystemExit("No entries after filtering.")

    combined_mode = (not args.separate_frictions) or args.combine_frictions
    print(f"Processing {len(entries)} rows combine={combined_mode} bundle={args.bundle_all}")

    # Validate thermal vs legacy velocity args
    if args.sigma_v is not None and (args.init_velocity_sigma is not None or args.w_speed != 0.2):
        raise SystemExit("--sigma-v is mutually exclusive with --init-velocity-sigma / --w-speed")

    if args.scene is not None:
        scene_src = args.scene
    elif args.nsc:
        scene_src = root_dir / "assets" / "scenes" / "default_entangled_nsc.json"
        if not scene_src.exists():
            scene_src = root_dir / "assets" / "scenes" / "default_entangled.json"
    else:
        scene_src = root_dir / "assets" / "scenes" / "default_entangled.json"
    if not scene_src.exists():
        raise SystemExit(f"Scene not found: {scene_src}")

    # Build NSC args dict if using hard contacts
    nsc_args = None
    if args.nsc:
        nsc_args = {
            "iters": args.nsc_iters, "beta": args.nsc_beta,
            "cfm": args.nsc_cfm, "omega": args.nsc_omega,
            "pos_iters": args.nsc_pos_iters, "pos_psor": args.nsc_pos_psor,
        }

    if args.sigma_v is not None:
        print(f"Thermal mode: sigma_v = {args.sigma_v} (kBT computed per-AR from rod mass)")
    if args.nsc:
        print(f"NSC mode: iters={args.nsc_iters} beta={args.nsc_beta} pos_iters={args.nsc_pos_iters}")

    binary_src = (root_dir / "build_cuda" / "rigidbody_viewer_3d" if args.use_cuda
                  else root_dir / "build-headless" / "rigidbody_viewer_3d")
    ensure_executable(binary_src)

    if args.use_cuda:
        slurm = SlurmCfg(partition="gpu", time="0-00:30:00", mem_gb=4,
                         ntasks=1, cpus=1, nodes=1,
                         module_line="module load gcc/13.2.0-fasrc01\n"
                                     "module load cuda/12.9.1-fasrc01")
    else:
        slurm = SlurmCfg(cpus=max(1, args.threads))

    if args.time is not None:
        slurm.time = args.time
    elif args.bundle_all:
        slurm.time = "3-00:00:00"

    friction_values: list = [None]
    if args.frictions is not None:
        friction_values = [float(f) for f in args.frictions.split(",") if f.strip()]
        print(f"Frictions: {friction_values}")

    endpoint_max = None if args.endpoint_max <= 0 else args.endpoint_max

    runs_root = args.runs_root / args.job_name
    runs_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), runs_root / Path(__file__).name)

    timestamp = now_ts()
    base_scene = json.loads(scene_src.read_text())
    submitted = skipped = failed = missing = 0
    local_tasks: List[LocalRunTask] = []

    if args.bundle_all:
        resolved = []
        for e in entries:
            N, ar, seed = e["N"], e["AR"], e["seed"]
            x_path = resolve_input_path(e, args.input_root)
            if x_path is None:
                print(f"WARNING: missing input N{N}/{seed}/x_relaxed_AR{ar}.txt")
                missing += 1
                continue
            resolved.append({**e, "x_path": x_path})

        if not resolved:
            raise SystemExit(f"No resolvable entries. Missing: {missing}")

        dt_val = args.dt if args.dt is not None else base_scene["physics"]["dt"]
        run_dir = runs_root / (
            f"{timestamp}_bundle_N{min(e['N'] for e in resolved)}-{max(e['N'] for e in resolved)}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")

        unique_Ns = sorted({e["N"] for e in resolved})
        unique_ARs = sorted({e["AR"] for e in resolved})
        for N in unique_Ns:
            for ar_val in unique_ARs:
                for f in friction_values:
                    scene_data = make_scene(base_scene, N, f, args.w_speed,
                                            args.init_velocity_sigma,
                                            args.use_cuda, args.delta, args.dt,
                                            use_nsc=args.nsc, sigma_v=args.sigma_v,
                                            ar=ar_val)
                    fname = f"scene_N{N}_AR{ar_val}_mu{f}.json" if f is not None else f"scene_N{N}_AR{ar_val}_mu_default.json"
                    (run_dir / fname).write_text(json.dumps(scene_data, indent=2))

        jname = safe_name(f"{args.job_name}_bundle")
        sb = build_sbatch_bundle(
            slurm, jname, args.use_cuda, resolved,
            friction_values, args.frames, dt_val,
            args.endpoint_stride, endpoint_max,
            args.stop_ke_threshold, args.stop_ke_min_steps,
            args.stop_slide_vel_threshold, args.stop_slide_vel_min_steps,
            nsc_args=nsc_args,
        )
        (run_dir / "Sbatch.sh").write_text(sb)

        print(f"Bundle dir: {run_dir}")
        print(
            f"  Entries: {len(resolved)} Missing: {missing} "
            f"Frictions: {len(friction_values)} Total runs: {len(resolved) * len(friction_values)}"
        )

        if not args.dry_run:
            if args.local:
                print(f"Queueing bundle locally in {run_dir} ...")
                for re_entry in resolved:
                    _N, _ar = re_entry["N"], re_entry["AR"]
                    _free_rod, _x_path = re_entry["free_rod"], re_entry["x_path"]
                    _seed_id, _metric = re_entry["id"], re_entry["metric"]
                    for fv in friction_values:
                        mu_tag = mu_file_tag(fv)
                        out_file = f"endpoints_N{_N}_AR{_ar}_{_seed_id}_{_metric}_rod{_free_rod}_mu{mu_tag}.csv"
                        scene_f = f"scene_N{_N}_AR{_ar}_mu{fv}.json" if fv is not None else f"scene_N{_N}_AR{_ar}_mu_default.json"
                        cmd = _build_sim_args(
                            scene_f, str(_x_path), args.frames, dt_val,
                            args.threads, _free_rod, out_file,
                            args.endpoint_stride, endpoint_max,
                            args.stop_ke_threshold, args.stop_ke_min_steps,
                            args.stop_slide_vel_threshold, args.stop_slide_vel_min_steps,
                            nsc_args=nsc_args,
                        )
                        local_tasks.append(LocalRunTask(
                            label=f"bundle N={_N} AR={_ar} {_seed_id} {_metric} rod={_free_rod} mu={fv}",
                            cmd=cmd,
                            cwd=run_dir,
                        ))
            else:
                r = subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir)
                if r.returncode != 0:
                    print("WARNING: sbatch failed.")
                else:
                    print("Submitted bundle job.")
        else:
            print(f"Dry run: {run_dir / 'Sbatch.sh'}")
        if args.local and not args.dry_run:
            queued = len(local_tasks)
            print(f"Running {queued} local subprocess task(s) with workers={max(1, args.local_workers)}")
            local_submitted, local_failed = run_local_tasks(local_tasks, args.local_workers)
            submitted += local_submitted
            failed += local_failed
        print(
            f"\nSubmitted {submitted}  Skipped {skipped} (done)  "
            f"Missing {missing} (no input)  Failed {failed}"
        )
        return

    for e in entries:
        N, ar, seed, seed_id = e["N"], e["AR"], e["seed"], e["id"]
        free_rod, metric = e["free_rod"], e["metric"]

        x_path = resolve_input_path(e, args.input_root)
        if x_path is None:
            print(f"WARNING: missing input N{N}/{seed}/x_relaxed_AR{ar}.txt")
            missing += 1
            continue

        dt_val = args.dt if args.dt is not None else base_scene["physics"]["dt"]

        if combined_mode:
            run_name = safe_name(f"{timestamp}_N{N}_{seed_id}_AR{ar}_{metric}_rod{free_rod}")
            run_dir = runs_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            # Skip if output for the last friction already exists.
            last_tag = mu_file_tag(friction_values[-1])
            sentinel = run_dir / f"free_rod_endpoints_mu{last_tag}.csv"
            if sentinel.exists():
                print(f"Skipping {run_name} (endpoint outputs exist)")
                skipped += 1
                continue

            shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")
            sym = run_dir / "x_relaxed.txt"
            if sym.exists():
                sym.unlink()
            sym.symlink_to(x_path)

            for f in friction_values:
                scene_data = make_scene(base_scene, N, f, args.w_speed,
                                        args.init_velocity_sigma,
                                        args.use_cuda, args.delta, args.dt,
                                        use_nsc=args.nsc, sigma_v=args.sigma_v,
                                        ar=ar)
                fname = f"scene_mu{f}.json" if f is not None else "scene_mu_default.json"
                (run_dir / fname).write_text(json.dumps(scene_data, indent=2))

            jname = safe_name(f"{args.job_name}_N{N}_AR{ar}_{metric}")
            sb = build_sbatch_combined(
                slurm, jname, args.use_cuda,
                N, ar, seed, metric, free_rod,
                friction_values, args.frames, dt_val,
                args.endpoint_stride, endpoint_max,
                args.stop_ke_threshold, args.stop_ke_min_steps,
                args.stop_slide_vel_threshold, args.stop_slide_vel_min_steps,
                nsc_args=nsc_args,
            )
            (run_dir / "Sbatch.sh").write_text(sb)

            if not args.dry_run:
                if args.local:
                    print(f"Queueing {run_name} locally...")
                    for fv in friction_values:
                        mu_tag = mu_file_tag(fv)
                        out_file = f"free_rod_endpoints_mu{mu_tag}.csv"
                        scene_f = f"scene_mu{fv}.json" if fv is not None else "scene_mu_default.json"
                        cmd = _build_sim_args(
                            scene_f, "x_relaxed.txt", args.frames, dt_val,
                            args.threads, free_rod, out_file,
                            args.endpoint_stride, endpoint_max,
                            args.stop_ke_threshold, args.stop_ke_min_steps,
                            args.stop_slide_vel_threshold, args.stop_slide_vel_min_steps,
                            nsc_args=nsc_args,
                        )
                        local_tasks.append(LocalRunTask(
                            label=f"{run_name} mu={fv}",
                            cmd=cmd,
                            cwd=run_dir,
                        ))
                else:
                    print(f"Submitting {run_name}...")
                    r = subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir)
                    if r.returncode != 0:
                        print(f"  WARNING: sbatch failed for {run_name}.")
                        failed += 1
                    else:
                        submitted += 1
            else:
                print(f"Dry run: {run_dir}")

        else:
            for friction in friction_values:
                mu_tag = f"_mu{friction}" if friction is not None else ""
                run_name = safe_name(
                    f"{timestamp}_N{N}_{seed_id}_AR{ar}_{metric}_rod{free_rod}{mu_tag}"
                )
                run_dir = runs_root / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                if (run_dir / "free_rod_endpoints.csv").exists():
                    print(f"Skipping {run_name} (free_rod_endpoints.csv exists)")
                    skipped += 1
                    continue

                shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")
                scene_data = make_scene(base_scene, N, friction, args.w_speed,
                                        args.init_velocity_sigma,
                                        args.use_cuda, args.delta, args.dt,
                                        use_nsc=args.nsc, sigma_v=args.sigma_v,
                                        ar=ar)
                (run_dir / "scene.json").write_text(json.dumps(scene_data, indent=2))

                sym = run_dir / "x_relaxed.txt"
                if sym.exists():
                    sym.unlink()
                sym.symlink_to(x_path)

                mu_str = f"mu={friction}" if friction is not None else "mu=scene"
                jname = safe_name(f"{args.job_name}_N{N}_AR{ar}_{metric}_{mu_str}")
                sb = build_sbatch_separate(
                    slurm, jname, args.use_cuda,
                    N, ar, seed, metric, free_rod, friction,
                    args.frames, dt_val,
                    args.endpoint_stride, endpoint_max,
                    args.stop_ke_threshold, args.stop_ke_min_steps,
                    args.stop_slide_vel_threshold, args.stop_slide_vel_min_steps,
                    nsc_args=nsc_args,
                )
                (run_dir / "Sbatch.sh").write_text(sb)

                if not args.dry_run:
                    if args.local:
                        print(f"Queueing {run_name} locally...")
                        cmd = _build_sim_args(
                            "scene.json", "x_relaxed.txt", args.frames, dt_val,
                            args.threads, free_rod, "free_rod_endpoints.csv",
                            args.endpoint_stride, endpoint_max,
                            args.stop_ke_threshold, args.stop_ke_min_steps,
                            args.stop_slide_vel_threshold, args.stop_slide_vel_min_steps,
                            nsc_args=nsc_args,
                        )
                        local_tasks.append(LocalRunTask(
                            label=run_name,
                            cmd=cmd,
                            cwd=run_dir,
                        ))
                    else:
                        print(f"Submitting {run_name}...")
                        r = subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir)
                        if r.returncode != 0:
                            print(f"  WARNING: sbatch failed for {run_name}.")
                            failed += 1
                        else:
                            submitted += 1
                else:
                    print(f"Dry run: {run_dir}")

    if args.local and not args.dry_run:
        queued = len(local_tasks)
        print(f"Running {queued} local subprocess task(s) with workers={max(1, args.local_workers)}")
        local_submitted, local_failed = run_local_tasks(local_tasks, args.local_workers)
        submitted += local_submitted
        failed += local_failed

    print(
        f"\nSubmitted {submitted}  Skipped {skipped} (done)  "
        f"Missing {missing} (no input)  Failed {failed}"
    )


if __name__ == "__main__":
    main()
