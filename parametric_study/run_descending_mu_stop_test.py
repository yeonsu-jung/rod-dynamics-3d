#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from submit_free_rod import (
    DEFAULT_INPUT_BASE,
    DEFAULT_RUNS_ROOT,
    ensure_executable,
    find_root_dir,
    load_extreme_rods_csv,
    make_scene,
    safe_name,
)


@dataclass(frozen=True)
class Entry:
    N: int
    AR: int
    id: str
    seed: str
    metric: str
    free_rod: int
    value: float
    x_path: Path


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def build_mu_values(mu_start: float, mu_stop: float, mu_step: float) -> list[float]:
    if mu_step <= 0:
        raise ValueError("mu_step must be > 0")
    values = []
    current = mu_start
    while current >= mu_stop - 1e-12:
        values.append(round(current, 10))
        current -= mu_step
    return values


def get_mu_values(args: argparse.Namespace) -> list[float]:
    """Return the list of mu values to run, honouring --mu-values if provided."""
    if args.mu_values:
        return sorted(args.mu_values, reverse=True)
    return build_mu_values(args.mu_start, args.mu_stop, args.mu_step)


def parse_args() -> argparse.Namespace:
    root_dir = find_root_dir()
    parser = argparse.ArgumentParser(
        description="Run descending-mu local free-rod tests in parallel and keep stopped cases."
    )
    parser.add_argument("--extreme-rods-csv", type=Path, default=root_dir / "extreme_rods_summary.csv")
    parser.add_argument("--input-root", type=Path, nargs="+", default=[DEFAULT_INPUT_BASE])
    parser.add_argument("--scene", type=Path, default=root_dir / "assets" / "scenes" / "default_entangled.json")
    parser.add_argument("--binary", type=Path, default=root_dir / "build_head" / "rigidbody_viewer_3d")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--job-name", type=str, default="free_rod_desc_mu_test")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--ar", type=int, default=None)
    parser.add_argument("--metrics", type=str, nargs="+", default=None)
    parser.add_argument("--ids", type=str, nargs="+", default=None)
    parser.add_argument("--mu-start", type=float, default=1.0)
    parser.add_argument("--mu-stop", type=float, default=0.0)
    parser.add_argument("--mu-step", type=float, default=0.1)
    parser.add_argument("--mu-values", type=float, nargs="+", default=None,
                        help="Explicit list of mu values to run (overrides --mu-start/stop/step).")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sim-threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--w-speed", type=float, default=0.2)
    parser.add_argument("--init-velocity-sigma", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--stop-slide-vel-threshold", type=float, default=1e-3)
    parser.add_argument("--stop-slide-vel-min-steps", type=int, default=1000)
    parser.add_argument("--endpoint-max", type=int, default=300)
    parser.add_argument("--stop-after-first-nonstop", action="store_true", default=True)
    parser.add_argument("--keep-all-cases", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Resume or continue into an existing run directory instead of creating a new timestamped one.")
    return parser.parse_args()


def resolve_entries(args: argparse.Namespace) -> list[Entry]:
    rows = load_extreme_rods_csv(args.extreme_rods_csv)
    if args.n is not None:
        rows = [r for r in rows if r["N"] == args.n]
    if args.ar is not None:
        rows = [r for r in rows if r["AR"] == args.ar]
    if args.metrics:
        allowed_metrics = set(args.metrics)
        rows = [r for r in rows if r["metric"] in allowed_metrics]
    if args.ids:
        allowed_ids = set(args.ids)
        rows = [r for r in rows if r["id"] in allowed_ids]
    resolved: list[Entry] = []
    for row in rows:
        x_path = None
        for root in args.input_root:
            candidate = root / f"N{row['N']}" / row["seed"] / f"x_relaxed_AR{row['AR']}.txt"
            if candidate.exists():
                x_path = candidate
                break
        if x_path is None:
            continue
        resolved.append(
            Entry(
                N=row["N"],
                AR=row["AR"],
                id=row["id"],
                seed=row["seed"],
                metric=row["metric"],
                free_rod=row["free_rod"],
                value=row["value"],
                x_path=x_path,
            )
        )
    return resolved


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _try_resume_entry(entry_dir: Path, entry: Entry, args: argparse.Namespace, mu_values: list[float]) -> list[dict] | None:
    """Return reconstructed result rows from existing logs if the entry is fully done, else None."""
    results: list[dict] = []
    for mu in mu_values:
        mu_tag = str(mu).replace("-", "m").replace(".", "p")
        log_path = entry_dir / f"run_mu{mu_tag}.log"
        output_csv = entry_dir / f"endpoints_mu{mu_tag}.csv"

        if not log_path.exists():
            # This mu was not run; expected only if the previous mu triggered stop_after_first_nonstop
            if results and not results[-1]["stopped"]:
                return results
            return None  # unexpected gap — re-run

        log_text = log_path.read_text()
        if "Headless run complete." not in log_text:
            return None  # incomplete log

        stopped = "[Headless] Early stop" in log_text
        stop_reason = ""
        if "|v.dot(axis)|" in log_text:
            stop_reason = "slide_velocity"
        elif " KE=" in log_text and stopped:
            stop_reason = "kinetic_energy"

        frames_completed = None
        for line in log_text.splitlines():
            if line.startswith("Headless run complete. Frames="):
                frames_completed = int(line.rsplit("=", 1)[1])
                break

        results.append({
            "N": entry.N, "AR": entry.AR, "ID": entry.id, "Metric": entry.metric,
            "RodIndex": entry.free_rod, "Value": entry.value,
            "mu": mu, "stopped": int(stopped), "stop_reason": stop_reason,
            "frames_completed": frames_completed if frames_completed is not None else -1,
            "returncode": 0,
            "run_dir": str(entry_dir),
            "endpoint_csv": str(output_csv),
            "log_path": str(log_path),
        })

        if args.stop_after_first_nonstop and not stopped:
            return results

    return results  # all mu values complete


def run_entry(entry: Entry, args: argparse.Namespace, run_root: Path, base_scene: dict) -> list[dict]:
    entry_dir = run_root / safe_name(f"N{entry.N}_AR{entry.AR}_{entry.id}_{entry.metric}_rod{entry.free_rod}")

    if entry_dir.exists():
        resumed = _try_resume_entry(entry_dir, entry, args, get_mu_values(args))
        if resumed is not None:
            return resumed

    entry_dir.mkdir(parents=True, exist_ok=True)
    link_path = entry_dir / "x_relaxed.txt"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(entry.x_path)

    results: list[dict] = []
    for mu in get_mu_values(args):
        mu_tag = str(mu).replace("-", "m").replace(".", "p")
        scene_path = entry_dir / f"scene_mu{mu_tag}.json"
        output_csv = entry_dir / f"endpoints_mu{mu_tag}.csv"
        log_path = entry_dir / f"run_mu{mu_tag}.log"

        scene_data = make_scene(
            base_scene,
            entry.N,
            mu,
            args.w_speed,
            args.init_velocity_sigma,
            False,
            args.delta,
            args.dt,
        )
        scene_path.write_text(json.dumps(scene_data, indent=2))

        cmd = [
            str(args.binary),
            "--headless",
            "--scene",
            str(scene_path),
            "--init-csv",
            str(link_path),
            "--steps",
            str(args.steps),
            "--dt",
            str(args.dt),
            "--threads",
            str(args.sim_threads),
            "--fix-every-except",
            str(entry.free_rod),
            "--test-rod-endpoints",
            str(output_csv),
            "--test-rod-endpoints-max",
            str(args.endpoint_max),
            "--stop-slide-vel-threshold",
            str(args.stop_slide_vel_threshold),
            "--stop-slide-vel-min-steps",
            str(args.stop_slide_vel_min_steps),
        ]

        proc = subprocess.run(
            cmd,
            cwd=entry_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log_path.write_text(proc.stdout)

        stopped = "[Headless] Early stop" in proc.stdout
        stop_reason = ""
        if "|v.dot(axis)|" in proc.stdout:
            stop_reason = "slide_velocity"
        elif " KE=" in proc.stdout and stopped:
            stop_reason = "kinetic_energy"

        frames_completed = None
        for line in proc.stdout.splitlines():
            if line.startswith("Headless run complete. Frames="):
                frames_completed = int(line.rsplit("=", 1)[1])

        row = {
            "N": entry.N,
            "AR": entry.AR,
            "ID": entry.id,
            "Metric": entry.metric,
            "RodIndex": entry.free_rod,
            "Value": entry.value,
            "mu": mu,
            "stopped": int(stopped),
            "stop_reason": stop_reason,
            "frames_completed": frames_completed if frames_completed is not None else -1,
            "returncode": proc.returncode,
            "run_dir": str(entry_dir),
            "endpoint_csv": str(output_csv),
            "log_path": str(log_path),
        }
        results.append(row)

        if args.stop_after_first_nonstop and not stopped:
            break
    return results


def main() -> None:
    args = parse_args()
    ensure_executable(args.binary)
    if not args.scene.exists():
        raise SystemExit(f"Scene not found: {args.scene}")

    entries = resolve_entries(args)
    if not entries:
        raise SystemExit("No resolvable entries for requested N/AR.")

    n_tag = "ALL" if args.n is None else str(args.n)
    ar_tag = "ALL" if args.ar is None else str(args.ar)
    if args.run_dir is not None:
        root_dir = args.run_dir
        root_dir.mkdir(parents=True, exist_ok=True)
    else:
        root_dir = args.runs_root / args.job_name / f"{now_ts()}_N{n_tag}_AR{ar_tag}"
        root_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), root_dir / Path(__file__).name)

    base_scene = json.loads(args.scene.read_text())

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_entry, entry, args, root_dir, base_scene): entry for entry in entries}
        for future in as_completed(futures):
            entry = futures[future]
            rows = future.result()
            all_rows.extend(rows)
            stopped_rows = [r for r in rows if r["stopped"]]
            print(
                f"[done] {entry.id} {entry.metric} rod={entry.free_rod} "
                f"tested={len(rows)} stopped={len(stopped_rows)}"
            )

    all_rows.sort(key=lambda r: (r["ID"], r["Metric"], -r["mu"]))
    stopped_rows = [r for r in all_rows if r["stopped"]]

    if args.keep_all_cases:
        write_csv(root_dir / "all_cases.csv", all_rows)
    write_csv(root_dir / "stopped_cases.csv", stopped_rows)

    summary = {
        "run_root": str(root_dir),
        "entries": len(entries),
        "tested_cases": len(all_rows),
        "stopped_cases": len(stopped_rows),
        "mu_values": get_mu_values(args),
        "workers": args.workers,
        "sim_threads": args.sim_threads,
        "steps": args.steps,
        "dt": args.dt,
        "stop_slide_vel_threshold": args.stop_slide_vel_threshold,
        "stop_slide_vel_min_steps": args.stop_slide_vel_min_steps,
    }
    (root_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Run root: {root_dir}")
    print(f"Entries: {len(entries)}")
    print(f"Tested cases: {len(all_rows)}")
    print(f"Stopped cases: {len(stopped_rows)}")
    print(f"Stopped CSV: {root_dir / 'stopped_cases.csv'}")
    if args.keep_all_cases:
        print(f"All cases CSV: {root_dir / 'all_cases.csv'}")


if __name__ == "__main__":
    main()