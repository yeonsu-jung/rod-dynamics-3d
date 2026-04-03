#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


RUN_RE = re.compile(
    r"^(?P<timestamp>\d{8}-\d{6})_(?P<seed>.+)_AR(?P<ar>\d+)_Friction(?P<friction>[-+0-9.]+)(?:_SigV(?P<sigv>[-+0-9.]+))?(?:_SigW(?P<sigw>[-+0-9.]+))?$"
)
N_RE = re.compile(r"_N(?P<n>\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect final entanglement values from completed run directories into one CSV."
    )
    parser.add_argument(
        "--run-root",
        action="append",
        required=True,
        help="Run root directory to scan. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    return parser.parse_args()


def parse_run_name(run_dir: Path) -> dict[str, object]:
    match = RUN_RE.match(run_dir.name)
    if not match:
        return {
            "timestamp": "",
            "seed": run_dir.parent.name,
            "ar": "",
            "friction": "",
            "sigma_v": "",
            "sigma_w": "",
        }
    groups = match.groupdict()
    return {
        "timestamp": groups.get("timestamp") or "",
        "seed": groups.get("seed") or "",
        "ar": int(groups["ar"]) if groups.get("ar") else "",
        "friction": float(groups["friction"]) if groups.get("friction") else "",
        "sigma_v": float(groups["sigv"]) if groups.get("sigv") else "",
        "sigma_w": float(groups["sigw"]) if groups.get("sigw") else "",
    }


def read_scene_metadata(scene_path: Path) -> dict[str, object]:
    if not scene_path.exists():
        return {
            "n_rods": "",
            "dt": "",
            "gravity": "",
            "lin_damp": "",
            "ang_damp": "",
            "nsc_enabled": "",
            "nsc_mu": "",
            "nsc_velocity_iters": "",
            "nsc_beta": "",
            "nsc_cfm": "",
            "nsc_omega": "",
            "random_init_mode": "",
            "random_init_sigma_v": "",
            "random_init_sigma_w": "",
            "random_init_kbt": "",
        }

    data = json.loads(scene_path.read_text())
    physics = data.get("physics", {})
    nsc = physics.get("nsc", {})
    random_init = data.get("scene", {}).get("randomInit", {})
    gravity = physics.get("gravity", "")
    return {
        "n_rods": data.get("scene", {}).get("populate", {}).get("count", ""),
        "dt": physics.get("dt", ""),
        "gravity": ",".join(str(x) for x in gravity) if isinstance(gravity, list) else gravity,
        "lin_damp": physics.get("lin_damp", ""),
        "ang_damp": physics.get("ang_damp", ""),
        "nsc_enabled": nsc.get("enabled", ""),
        "nsc_mu": nsc.get("mu", ""),
        "nsc_velocity_iters": nsc.get("velocity_iters", ""),
        "nsc_beta": nsc.get("beta", ""),
        "nsc_cfm": nsc.get("cfm", ""),
        "nsc_omega": nsc.get("omega", ""),
        "random_init_mode": random_init.get("mode", ""),
        "random_init_sigma_v": random_init.get("vSigma", ""),
        "random_init_sigma_w": random_init.get("wSigma", ""),
        "random_init_kbt": random_init.get("kBT", ""),
    }


def read_final_row(output_csv: Path) -> dict[str, object] | None:
    last_row = None
    with output_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            last_row = row
    if last_row is None:
        return None
    return {
        "final_frame": int(last_row["frame"]) if last_row.get("frame") else "",
        "final_contacts": int(last_row["contacts"]) if last_row.get("contacts") else "",
        "final_ke": float(last_row["KE"]) if last_row.get("KE") else "",
        "final_max_overlap": float(last_row["max_overlap"]) if last_row.get("max_overlap") else "",
        "final_gyration_sq": float(last_row["gyration_sq"]) if last_row.get("gyration_sq") else "",
        "final_reldisp_sq": float(last_row["reldisp_sq"]) if last_row.get("reldisp_sq") else "",
        "final_ent_sum": float(last_row["ent_sum"]) if last_row.get("ent_sum") else "",
        "final_ent_pairs": int(last_row["ent_pairs"]) if last_row.get("ent_pairs") else "",
    }


def infer_submission_mode(root: Path) -> str:
    name = root.name
    if "array" in name:
        return "array"
    if "single" in name:
        return "single"
    return "unknown"


def infer_n_from_root(root: Path) -> object:
    match = N_RE.search(root.name + "_")
    return int(match.group("n")) if match else ""


def collect_rows(run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    submission_mode = infer_submission_mode(run_root)
    root_n = infer_n_from_root(run_root)
    for output_csv in sorted(run_root.rglob("output.csv")):
        run_dir = output_csv.parent
        final_row = read_final_row(output_csv)
        if final_row is None:
            continue
        name_meta = parse_run_name(run_dir)
        scene_meta = read_scene_metadata(run_dir / "scene.json")
        rows.append(
            {
                "submission_mode": submission_mode,
                "run_root": str(run_root),
                "run_dir": str(run_dir),
                "output_csv": str(output_csv),
                "n_rods": scene_meta["n_rods"] or root_n,
                "timestamp": name_meta["timestamp"],
                "seed": name_meta["seed"],
                "ar": name_meta["ar"],
                "friction": name_meta["friction"],
                "sigma_v": name_meta["sigma_v"] or scene_meta["random_init_sigma_v"],
                "sigma_w": name_meta["sigma_w"] or scene_meta["random_init_sigma_w"],
                **scene_meta,
                **final_row,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, object]] = []
    for root_str in args.run_root:
        root = Path(root_str)
        all_rows.extend(collect_rows(root))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "submission_mode",
        "run_root",
        "run_dir",
        "output_csv",
        "n_rods",
        "timestamp",
        "seed",
        "ar",
        "friction",
        "sigma_v",
        "sigma_w",
        "dt",
        "gravity",
        "lin_damp",
        "ang_damp",
        "nsc_enabled",
        "nsc_mu",
        "nsc_velocity_iters",
        "nsc_beta",
        "nsc_cfm",
        "nsc_omega",
        "random_init_mode",
        "random_init_sigma_v",
        "random_init_sigma_w",
        "random_init_kbt",
        "final_frame",
        "final_contacts",
        "final_ke",
        "final_max_overlap",
        "final_gyration_sq",
        "final_reldisp_sq",
        "final_ent_sum",
        "final_ent_pairs",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()