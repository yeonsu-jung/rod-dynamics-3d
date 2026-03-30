#!/usr/bin/env python3
"""Scan free-rod endpoint runs, report health, and generate summary plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_RE = re.compile(
    r"^\d{8}-\d{6}_N(?P<N>\d+)_(?P<seed>.+)_AR(?P<AR>\d+)_(?P<metric>MinFSA|MaxFSA|MinFTA|MaxFTA)_rod(?P<rod>\d+)$"
)
CSV_RE = re.compile(r"^free_rod_endpoints_mu(?P<tag>.+)\.csv$")
DEFAULT_MUS = [0.0, 0.1, 0.2, 0.4, 1.0]
METRIC_ORDER = ["MinFSA", "MaxFSA", "MinFTA", "MaxFTA"]
COLORS = {
    "MinFSA": "#0f766e",
    "MaxFSA": "#dc2626",
    "MinFTA": "#2563eb",
    "MaxFTA": "#d97706",
}


@dataclass(frozen=True)
class RunMeta:
    path: Path
    run_name: str
    N: int
    AR: int
    seed_id: str
    metric: str
    rod: int


def mu_to_tag(mu: float) -> str:
    return str(mu).replace("-", "m").replace(".", "p")


def tag_to_mu(tag: str) -> float:
    return float(tag.replace("m", "-").replace("p", "."))


def parse_run_dir(path: Path) -> RunMeta | None:
    match = RUN_RE.match(path.name)
    if not match:
        return None
    groups = match.groupdict()
    return RunMeta(
        path=path,
        run_name=path.name,
        N=int(groups["N"]),
        AR=int(groups["AR"]),
        seed_id=groups["seed"],
        metric=groups["metric"],
        rod=int(groups["rod"]),
    )


def load_endpoint_csv(csv_path: Path) -> dict[str, np.ndarray]:
    frames = []
    times = []
    centers = []
    orient = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x0 = float(row["x0"])
            y0 = float(row["y0"])
            z0 = float(row["z0"])
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            z1 = float(row["z1"])
            frames.append(int(row["frame"]))
            times.append(float(row["time"]))
            center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5])
            centers.append(center)
            axis = np.array([x1 - x0, y1 - y0, z1 - z0])
            norm = float(np.linalg.norm(axis))
            orient.append(axis / norm if norm > 0 else np.array([0.0, 0.0, 0.0]))

    frame_arr = np.asarray(frames, dtype=int)
    time_arr = np.asarray(times, dtype=float)
    center_arr = np.asarray(centers, dtype=float)
    orient_arr = np.asarray(orient, dtype=float)
    disp = np.linalg.norm(center_arr - center_arr[0], axis=1)
    steps = np.linalg.norm(np.diff(center_arr, axis=0, prepend=center_arr[[0]]), axis=1)
    path = np.cumsum(steps)
    dot = np.clip(np.sum(orient_arr * orient_arr[0], axis=1), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(dot))
    return {
        "frame": frame_arr,
        "time": time_arr,
        "disp": disp,
        "path": path,
        "angle_deg": angle_deg,
    }


def compute_stats(series_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.vstack(series_list)
    return stacked.mean(axis=0), stacked.std(axis=0)


def make_health_summary(run_root: Path, expected_mus: list[float], metas: list[RunMeta]) -> dict:
    per_mu = Counter()
    complete_runs = 0
    missing_by_mu = Counter()
    run_rows: list[dict] = []

    for meta in metas:
        available = set()
        for csv_path in meta.path.glob("free_rod_endpoints_mu*.csv"):
            match = CSV_RE.match(csv_path.name)
            if match:
                mu = tag_to_mu(match.group("tag"))
                available.add(mu)
                per_mu[mu] += 1
        missing = [mu for mu in expected_mus if mu not in available]
        if not missing:
            complete_runs += 1
        for mu in missing:
            missing_by_mu[mu] += 1
        run_rows.append(
            {
                "run_name": meta.run_name,
                "N": meta.N,
                "AR": meta.AR,
                "seed_id": meta.seed_id,
                "metric": meta.metric,
                "rod": meta.rod,
                "available_mu": sorted(available),
                "missing_mu": missing,
                "complete": not missing,
            }
        )

    return {
        "run_root": str(run_root),
        "total_run_dirs": len(metas),
        "expected_mus": expected_mus,
        "complete_runs": complete_runs,
        "incomplete_runs": len(metas) - complete_runs,
        "per_mu_file_counts": {str(mu): per_mu.get(mu, 0) for mu in expected_mus},
        "missing_by_mu": {str(mu): missing_by_mu.get(mu, 0) for mu in expected_mus},
        "runs": run_rows,
    }


def filter_metas(
    metas: list[RunMeta],
    filter_n: set[int] | None,
    filter_ar: set[int] | None,
    filter_metric: set[str] | None,
) -> list[RunMeta]:
    result = []
    for meta in metas:
        if filter_n and meta.N not in filter_n:
            continue
        if filter_ar and meta.AR not in filter_ar:
            continue
        if filter_metric and meta.metric not in filter_metric:
            continue
        result.append(meta)
    return result


def plot_mean_metric_grid(
    trajectories: dict[tuple[float, str], list[dict[str, np.ndarray]]],
    metric_key: str,
    out_path: Path,
) -> None:
    mus = sorted({mu for mu, _ in trajectories})
    fig, axes = plt.subplots(1, len(mus), figsize=(4.0 * max(1, len(mus)), 4.2), sharey=True)
    if len(mus) == 1:
        axes = [axes]

    for ax, mu in zip(axes, mus):
        for metric in METRIC_ORDER:
            runs = trajectories.get((mu, metric), [])
            if not runs:
                continue
            mean, std = compute_stats([run[metric_key] for run in runs])
            time = runs[0]["time"]
            ax.plot(time, mean, color=COLORS[metric], label=metric, linewidth=1.8)
            ax.fill_between(time, mean - std, mean + std, color=COLORS[metric], alpha=0.18)
        ax.set_title(f"mu = {mu:g}")
        ax.set_xlabel("time")
        ax.grid(True, alpha=0.3)

    ylabel = {
        "disp": "center displacement",
        "path": "cumulative path length",
        "angle_deg": "orientation change (deg)",
    }[metric_key]
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_final_boxplot(
    trajectories: dict[tuple[float, str], list[dict[str, np.ndarray]]],
    metric_key: str,
    out_path: Path,
) -> None:
    mus = sorted({mu for mu, _ in trajectories})
    fig, axes = plt.subplots(1, len(mus), figsize=(4.2 * max(1, len(mus)), 4.4), sharey=True)
    if len(mus) == 1:
        axes = [axes]

    for ax, mu in zip(axes, mus):
        data = []
        labels = []
        colors = []
        for metric in METRIC_ORDER:
            runs = trajectories.get((mu, metric), [])
            if not runs:
                continue
            data.append([run[metric_key][-1] for run in runs])
            labels.append(metric)
            colors.append(COLORS[metric])
        if not data:
            continue
        box = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(f"mu = {mu:g}")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)

    ylabel = {
        "disp": "final center displacement",
        "path": "final cumulative path length",
        "angle_deg": "final orientation change (deg)",
    }[metric_key]
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/n/holylabs/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs"),
    )
    parser.add_argument("--job-name", default="free_rod_sweep")
    parser.add_argument("--out-dir", type=Path, default=Path("plots_free_rod_endpoints"))
    parser.add_argument("--filter-n", nargs="*", type=int, default=None)
    parser.add_argument("--filter-ar", nargs="*", type=int, default=None)
    parser.add_argument("--filter-metric", nargs="*", default=None)
    parser.add_argument("--only-complete", action="store_true", default=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    run_root = args.runs_root / args.job_name
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metas = [meta for path in sorted(run_root.iterdir()) if path.is_dir() if (meta := parse_run_dir(path))]
    summary = make_health_summary(run_root, DEFAULT_MUS, metas)
    (args.out_dir / "health_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Run root: {run_root}")
    print(f"Total run dirs: {summary['total_run_dirs']}")
    print(f"Complete runs: {summary['complete_runs']}")
    print(f"Incomplete runs: {summary['incomplete_runs']}")
    print(f"Per-mu file counts: {summary['per_mu_file_counts']}")

    filtered = filter_metas(
        metas,
        set(args.filter_n) if args.filter_n else None,
        set(args.filter_ar) if args.filter_ar else None,
        set(args.filter_metric) if args.filter_metric else None,
    )

    if args.only_complete and not args.allow_incomplete:
        complete_names = {row["run_name"] for row in summary["runs"] if row["complete"]}
        filtered = [meta for meta in filtered if meta.run_name in complete_names]

    trajectories: dict[tuple[float, str], list[dict[str, np.ndarray]]] = defaultdict(list)
    loaded = 0
    for meta in filtered:
        for mu in DEFAULT_MUS:
            csv_path = meta.path / f"free_rod_endpoints_mu{mu_to_tag(mu)}.csv"
            if not csv_path.exists():
                continue
            data = load_endpoint_csv(csv_path)
            trajectories[(mu, meta.metric)].append(data)
            loaded += 1

    print(f"Loaded endpoint CSVs: {loaded}")
    if not trajectories:
        print("No endpoint CSV data available for plotting yet.")
        return

    plot_mean_metric_grid(trajectories, "disp", args.out_dir / "mean_displacement_vs_time.png")
    plot_mean_metric_grid(trajectories, "path", args.out_dir / "mean_path_vs_time.png")
    plot_mean_metric_grid(trajectories, "angle_deg", args.out_dir / "mean_orientation_change_vs_time.png")
    plot_final_boxplot(trajectories, "disp", args.out_dir / "final_displacement_boxplot.png")
    plot_final_boxplot(trajectories, "path", args.out_dir / "final_path_boxplot.png")
    plot_final_boxplot(trajectories, "angle_deg", args.out_dir / "final_orientation_change_boxplot.png")
    print(f"Saved plots under {args.out_dir}")


if __name__ == "__main__":
    main()