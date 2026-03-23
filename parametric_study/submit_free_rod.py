#!/usr/bin/env python3
"""submit_free_rod.py

Submit SLURM jobs for single-free-rod perturbation experiments.

Two modes depending on --combine-frictions:

  Normal (one job per friction):
    run_dir/  rigidbody_viewer_3d, scene.json, x_relaxed.txt, Sbatch.sh
    outputs:  free_rod.csv (filtered), perrod.csv deleted after filtering

  Combined (one job, all frictions in sequence):
    run_dir/  rigidbody_viewer_3d, scene_mu{f}.json x frictions, x_relaxed.txt, Sbatch.sh
    outputs:  free_rod.csv with prepended 'mu' column, all frictions stacked

Input is extreme_rods_summary.csv (columns: N,AR,ID,Metric,RodIndex,Value,FilePath).
ID uses underscores (278_868_121); folder uses commas (278,868,121).
Input files: <input-root>/N{N}/{ID with commas}/x_relaxed_AR{AR}.txt
"""

import argparse
import csv
import json
import os
import re
import shutil
import stat
import subprocess

from datetime import datetime
from pathlib import Path
from typing import List, Optional


DEFAULT_RUNS_ROOT = Path(
    "/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs"
)
DEFAULT_INPUT_BASE = Path(
    "/n/home01/yjung/Github/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading"
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


def make_scene(base: dict, N: int, friction: Optional[float],
               w_speed: float, v_sigma: Optional[float],
               use_cuda: bool, delta: Optional[float], dt: Optional[float]) -> dict:
    d = json.loads(json.dumps(base))
    d.setdefault("scene", {}).setdefault("populate", {})["count"] = N
    ri = d["scene"].setdefault("randomInit",
                               {"enabled": True, "vSigma": 0.0, "wSpeed": 0.01, "seed": 42})
    ri["enabled"] = True
    ri["wSpeed"] = w_speed
    if v_sigma is not None:
        ri["vSigma"] = v_sigma
    if friction is not None:
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
                 ntasks=1, cpus=8, nodes=1,
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
                "N":        int(row["N"]),
                "AR":       int(row["AR"]),
                "id":       row["ID"],
                "seed":     row["ID"].replace("_", ","),
                "metric":   row["Metric"],
                "free_rod": int(row["RodIndex"]),
                "value":    float(row["Value"]),
            })
    return entries


# ---------------------------------------------------------------------------
# Sbatch builders
# ---------------------------------------------------------------------------

def _filter_py_separate(free_rod: int) -> str:
    """Filter perrod.csv to free_rod rows only. Writes to stdout."""
    return "\n".join([
        "import sys",
        f"free = {free_rod}",
        "with open('perrod.csv') as f:",
        "    for line in f:",
        "        if line.startswith('#') or line.startswith('frame,'):",
        "            sys.stdout.write(line)",
        "        else:",
        "            cols = line.split(',')",
        f"            if len(cols) > 1 and cols[1].strip() == '{free_rod}':",
        "                sys.stdout.write(line)",
    ])


def _filter_py_combined(free_rod: int) -> str:
    """Filter perrod_tmp.csv, prepend mu column. Args: mu is_first.
    Called as: python3 - "$MU" "$FIRST" <<PYEOF >> free_rod.csv
    """
    return "\n".join([
        "import sys",
        f"free = {free_rod}",
        "mu    = sys.argv[1]",
        "first = sys.argv[2] == '1'",
        "with open('perrod_tmp.csv') as f:",
        "    for line in f:",
        "        if line.startswith('#'):",
        "            if first: sys.stdout.write(line)",
        "        elif line.startswith('frame,'):",
        "            if first: sys.stdout.write('mu,' + line)",
        "        else:",
        "            cols = line.split(',')",
        f"            if len(cols) > 1 and cols[1].strip() == '{free_rod}':",
        "                sys.stdout.write(mu + ',' + line)",
    ])


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


def build_sbatch_separate(slurm, job_name, use_cuda, N, ar, seed, metric,
                          free_rod, friction, frames, dt_val, w_speed, perrod_stride):
    mu_str = f"mu={friction}" if friction is not None else "mu=scene"
    sim_cmd = " ".join([
        "./rigidbody_viewer_3d --headless",
        "--scene scene.json --init-csv x_relaxed.txt",
        f"--steps {frames} --dt {dt_val} --threads {slurm.cpus}",
        f"--fix-every-except {free_rod}",
        "--perrod perrod.csv",
        f"--perrod-stride {perrod_stride}",
    ])
    fp = _filter_py_separate(free_rod)
    return f"""{_header(slurm, job_name, use_cuda)}

echo "Free-rod: N={N} AR={ar} seed={seed} metric={metric} rod={free_rod} {mu_str} frames={frames}"

{sim_cmd}

python3 - <<'PYEOF' > free_rod.csv
{fp}
PYEOF
rm -f perrod.csv
echo "Rows: $(grep -c '^[0-9]' free_rod.csv || true)"
echo "Job complete."
"""


def _filter_py_bundle() -> str:
    """Filter perrod_tmp.csv, prepend full metadata. Args: N AR seed_id metric free_rod mu is_first."""
    return "\n".join([
        "import sys",
        "N, AR, seed_id, metric, free_rod_s, mu, first_s = sys.argv[1:]",
        "free_rod = int(free_rod_s)",
        "first = first_s == '1'",
        "with open('perrod_tmp.csv') as f:",
        "    for line in f:",
        "        if line.startswith('#'):",
        "            pass",
        "        elif line.startswith('frame,'):",
        "            if first: sys.stdout.write('N,AR,seed_id,metric,rod,mu,' + line)",
        "        else:",
        "            cols = line.split(',')",
        "            if len(cols) > 1 and cols[1].strip() == str(free_rod):",
        "                sys.stdout.write(f'{N},{AR},{seed_id},{metric},{free_rod},{mu},' + line)",
    ])


def build_sbatch_bundle(slurm, job_name, use_cuda, resolved_entries,
                        friction_values, frames, dt_val, w_speed, perrod_stride):
    """One SLURM job that runs every entry × every friction sequentially."""
    mu_list = " ".join(str(f) for f in friction_values)
    fp = _filter_py_bundle()
    sim_cmd = " ".join([
        "./rigidbody_viewer_3d --headless",
        '--scene "scene_N${N}_mu${MU}.json"',
        '--init-csv "$X_PATH"',
        f"--steps {frames} --dt {dt_val} --threads {slurm.cpus}",
        '--fix-every-except "$FREE_ROD"',
        "--perrod perrod_tmp.csv",
        f"--perrod-stride {perrod_stride}",
    ])
    inner = (
        f"    for MU in {mu_list}; do\n"
        f'        echo "[$(date +%H:%M:%S)] N=$N AR=$AR $SEED_ID $METRIC rod=$FREE_ROD mu=$MU"\n'
        f"        {sim_cmd}\n"
        f"        python3 - \"$N\" \"$AR\" \"$SEED_ID\" \"$METRIC\" \"$FREE_ROD\" \"$MU\" \"$FIRST\" <<'PYEOF' >> free_rod_all.csv\n"
        f"{fp}\n"
        f"PYEOF\n"
        f"        rm -f perrod_tmp.csv\n"
        f"        FIRST=0\n"
        f"    done"
    )
    calls = "\n".join(
        f'run_one {e["N"]} {e["AR"]} {e["id"]} {e["metric"]} {e["free_rod"]} "{e["x_path"]}"'
        for e in resolved_entries
    )
    n_runs = len(resolved_entries) * len(friction_values)
    return f"""{_header(slurm, job_name, use_cuda)}

echo "Bundle: {len(resolved_entries)} entries × {len(friction_values)} frictions = {n_runs} runs"
echo "Frames={frames}  dt={dt_val}  wSpeed={w_speed}  threads={slurm.cpus}"

FIRST=1

run_one() {{
    local N=$1 AR=$2 SEED_ID=$3 METRIC=$4 FREE_ROD=$5 X_PATH=$6
{inner}
}}

{calls}

echo "Total rows: $(grep -c '^[0-9]' free_rod_all.csv || true)"
echo "Bundle complete."
"""


def build_sbatch_combined(slurm, job_name, use_cuda, N, ar, seed, metric,
                          free_rod, friction_values, frames, dt_val, w_speed, perrod_stride):
    mu_list = " ".join(str(f) for f in friction_values)
    # scene file name per mu: scene_mu0.0.json etc.
    sim_cmd = " ".join([
        "./rigidbody_viewer_3d --headless",
        '--scene "scene_mu${MU}.json"',
        "--init-csv x_relaxed.txt",
        f"--steps {frames} --dt {dt_val} --threads {slurm.cpus}",
        f"--fix-every-except {free_rod}",
        "--perrod perrod_tmp.csv",
        f"--perrod-stride {perrod_stride}",
    ])
    fp = _filter_py_combined(free_rod)
    return f"""{_header(slurm, job_name, use_cuda)}

echo "Free-rod combined: N={N} AR={ar} seed={seed} metric={metric} rod={free_rod}"
echo "Frictions: {mu_list}  frames={frames} dt={dt_val} wSpeed={w_speed}"

FIRST=1
for MU in {mu_list}; do
    echo "  --- mu=$MU ---"
    {sim_cmd}

    python3 - "$MU" "$FIRST" <<'PYEOF' >> free_rod.csv
{fp}
PYEOF
    rm -f perrod_tmp.csv
    FIRST=0
done
echo "Total rows: $(grep -c '^[0-9]' free_rod.csv || true)"
echo "Job complete."
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Submit single-free-rod SLURM jobs from extreme_rods_summary.csv."
    )
    ap.add_argument("--extreme-rods-csv", type=Path, required=True)
    ap.add_argument("--input-root",  type=Path, nargs="+", default=[DEFAULT_INPUT_BASE])
    ap.add_argument("--job-name",    type=str,  default="free_rod")
    ap.add_argument("--scene",       type=Path, default=None)
    ap.add_argument("--runs-root",   type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--frames",      type=int,  default=300)
    ap.add_argument("--dt",          type=float, default=None)
    ap.add_argument("--perrod-stride", type=int, default=1)
    ap.add_argument("--frictions",   type=str,  default=None,
                    help="Comma-separated friction values.")
    ap.add_argument("--init-velocity-sigma", type=float, default=None)
    ap.add_argument("--w-speed",     type=float, default=0.2)
    ap.add_argument("--threads",     type=int,  default=8)
    ap.add_argument("--use-cuda",    action="store_true")
    ap.add_argument("--delta",       type=float, default=None)
    ap.add_argument("--combine-frictions", action="store_true",
                    help="Run all frictions in one SLURM job; output single free_rod.csv "
                         "with 'mu' column prepended. Recommended for small N.")
    ap.add_argument("--bundle-all", action="store_true",
                    help="Bundle ALL filtered entries into one SLURM job; output single "
                         "free_rod_all.csv with full metadata (N,AR,seed_id,metric,rod,mu,...).")
    ap.add_argument("--time",        type=str,  default=None,
                    help="SLURM time limit (e.g. '2-00:00:00'). "
                         "Default: 0-01:00:00 normally, 3-00:00:00 for --bundle-all.")
    ap.add_argument("--filter-n",      type=int, nargs="+", default=None)
    ap.add_argument("--filter-ar",     type=int, nargs="+", default=None)
    ap.add_argument("--filter-id",     type=str, nargs="+", default=None)
    ap.add_argument("--filter-metric", type=str, nargs="+", default=None,
                    choices=["MinFSA", "MaxFSA", "MinFTA", "MaxFTA"])
    ap.add_argument("--dry-run",     action="store_true")
    args = ap.parse_args()

    root_dir = find_root_dir()

    entries = load_extreme_rods_csv(args.extreme_rods_csv)
    if not entries:
        raise SystemExit(f"No rows in {args.extreme_rods_csv}")

    if args.filter_n:
        entries = [e for e in entries if e["N"] in args.filter_n]
    if args.filter_ar:
        entries = [e for e in entries if e["AR"] in args.filter_ar]
    if args.filter_id:
        entries = [e for e in entries if e["id"] in args.filter_id]
    if args.filter_metric:
        entries = [e for e in entries if e["metric"] in args.filter_metric]
    if not entries:
        raise SystemExit("No entries after filtering.")
    print(f"Processing {len(entries)} rows  combine={args.combine_frictions}")

    scene_src = args.scene or root_dir / "assets" / "scenes" / "default_entangled.json"
    if not scene_src.exists():
        raise SystemExit(f"Scene not found: {scene_src}")

    binary_src = (root_dir / "build_cuda" / "rigidbody_viewer_3d" if args.use_cuda
                  else root_dir / "build_head" / "rigidbody_viewer_3d")
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

    runs_root = args.runs_root / args.job_name
    runs_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), runs_root / Path(__file__).name)

    timestamp = now_ts()
    base_scene = json.loads(scene_src.read_text())
    submitted = skipped = failed = missing = 0

    # ── Bundle-all mode: one SLURM job for every entry ────────────────────────
    if args.bundle_all:
        # Resolve x_path for each entry up front
        resolved = []
        for e in entries:
            N, ar, seed = e["N"], e["AR"], e["seed"]
            x_path = None
            for root in args.input_root:
                candidate = root / f"N{N}" / seed / f"x_relaxed_AR{ar}.txt"
                if candidate.exists():
                    x_path = candidate
                    break
            if x_path is None:
                print(f"WARNING: not found in any input root: N{N}/{seed}/x_relaxed_AR{ar}.txt")
                missing += 1
                continue
            resolved.append({**e, "x_path": x_path})

        if not resolved:
            raise SystemExit(f"No resolvable entries. {missing} missing.")

        dt_val = args.dt if args.dt is not None else base_scene["physics"]["dt"]
        run_dir = runs_root / f"{timestamp}_bundle_N{min(e['N'] for e in resolved)}-{max(e['N'] for e in resolved)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if (run_dir / "free_rod_all.csv").exists():
            print(f"Skipping bundle (free_rod_all.csv exists): {run_dir}")
            return

        shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")

        # Scene files: one per (N, mu)
        unique_Ns = sorted({e["N"] for e in resolved})
        for N in unique_Ns:
            for f in friction_values:
                scene_data = make_scene(base_scene, N, f, args.w_speed,
                                        args.init_velocity_sigma,
                                        args.use_cuda, args.delta, args.dt)
                fname = f"scene_N{N}_mu{f}.json" if f is not None else f"scene_N{N}_mu_default.json"
                (run_dir / fname).write_text(json.dumps(scene_data, indent=2))

        jname = safe_name(f"{args.job_name}_bundle")
        sb = build_sbatch_bundle(slurm, jname, args.use_cuda, resolved,
                                 friction_values, args.frames, dt_val,
                                 args.w_speed, args.perrod_stride)
        (run_dir / "Sbatch.sh").write_text(sb)

        print(f"Bundle dir: {run_dir}")
        print(f"  Entries: {len(resolved)}  Missing: {missing}  "
              f"Frictions: {len(friction_values)}  Total runs: {len(resolved)*len(friction_values)}")
        if not args.dry_run:
            r = subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir)
            if r.returncode != 0:
                print("WARNING: sbatch failed.")
            else:
                print("Submitted bundle job.")
        else:
            print(f"Dry run: {run_dir / 'Sbatch.sh'}")
        return
    # ─────────────────────────────────────────────────────────────────────────

    for e in entries:
        N, ar, seed, seed_id = e["N"], e["AR"], e["seed"], e["id"]
        free_rod, metric = e["free_rod"], e["metric"]

        x_path = None
        for root in args.input_root:
            candidate = root / f"N{N}" / seed / f"x_relaxed_AR{ar}.txt"
            if candidate.exists():
                x_path = candidate
                break
        if x_path is None:
            print(f"WARNING: not found in any input root: N{N}/{seed}/x_relaxed_AR{ar}.txt")
            missing += 1
            continue

        dt_val = args.dt if args.dt is not None else base_scene["physics"]["dt"]

        if args.combine_frictions:
            # One job, all frictions, single stacked free_rod.csv
            run_name = safe_name(f"{timestamp}_N{N}_{seed_id}_AR{ar}_{metric}_rod{free_rod}")
            run_dir = runs_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            if (run_dir / "free_rod.csv").exists():
                print(f"Skipping {run_name} (free_rod.csv exists)")
                skipped += 1
                continue

            shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")
            sym = run_dir / "x_relaxed.txt"
            if sym.exists():
                sym.unlink()
            sym.symlink_to(x_path)

            # One scene file per friction value
            for f in friction_values:
                scene_data = make_scene(base_scene, N, f, args.w_speed,
                                        args.init_velocity_sigma,
                                        args.use_cuda, args.delta, args.dt)
                fname = f"scene_mu{f}.json" if f is not None else "scene_mu_default.json"
                (run_dir / fname).write_text(json.dumps(scene_data, indent=2))

            jname = safe_name(f"{args.job_name}_N{N}_AR{ar}_{metric}")
            sb = build_sbatch_combined(slurm, jname, args.use_cuda,
                                       N, ar, seed, metric, free_rod,
                                       friction_values, args.frames, dt_val,
                                       args.w_speed, args.perrod_stride)
            (run_dir / "Sbatch.sh").write_text(sb)

            if not args.dry_run:
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
            # One job per friction value
            for friction in friction_values:
                mu_tag = f"_mu{friction}" if friction is not None else ""
                run_name = safe_name(
                    f"{timestamp}_N{N}_{seed_id}_AR{ar}_{metric}_rod{free_rod}{mu_tag}"
                )
                run_dir = runs_root / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                if (run_dir / "free_rod.csv").exists():
                    print(f"Skipping {run_name} (free_rod.csv exists)")
                    skipped += 1
                    continue

                shutil.copy2(binary_src, run_dir / "rigidbody_viewer_3d")
                scene_data = make_scene(base_scene, N, friction, args.w_speed,
                                        args.init_velocity_sigma,
                                        args.use_cuda, args.delta, args.dt)
                (run_dir / "scene.json").write_text(json.dumps(scene_data, indent=2))

                sym = run_dir / "x_relaxed.txt"
                if sym.exists():
                    sym.unlink()
                sym.symlink_to(x_path)

                mu_str = f"mu={friction}" if friction is not None else "mu=scene"
                jname = safe_name(f"{args.job_name}_N{N}_AR{ar}_{metric}_{mu_str}")
                sb = build_sbatch_separate(slurm, jname, args.use_cuda,
                                           N, ar, seed, metric, free_rod, friction,
                                           args.frames, dt_val, args.w_speed,
                                           args.perrod_stride)
                (run_dir / "Sbatch.sh").write_text(sb)

                if not args.dry_run:
                    print(f"Submitting {run_name}...")
                    r = subprocess.run(["sbatch", "Sbatch.sh"], cwd=run_dir)
                    if r.returncode != 0:
                        print(f"  WARNING: sbatch failed for {run_name}.")
                        failed += 1
                    else:
                        submitted += 1
                else:
                    print(f"Dry run: {run_dir}")

    print(
        f"\nSubmitted {submitted}  Skipped {skipped} (done)  "
        f"Missing {missing} (no input)  Failed {failed} (sbatch error)"
    )


if __name__ == "__main__":
    main()
