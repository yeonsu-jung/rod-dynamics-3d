from __future__ import annotations

import json
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Sweep soft-contact mu values and overlay averaged all-pair velocity norms."
    )
    parser.add_argument(
        "--exe",
        default="/Users/yeonsu/GitHub/rod-dynamics-3d/build/rigidbody_viewer_3d",
    )
    parser.add_argument(
        "--scene",
        default="/Users/yeonsu/GitHub/rod-dynamics-3d/assets/scenes/default_entangled.json",
    )
    parser.add_argument(
        "--init-csv",
        default="/Users/yeonsu/GitHub/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading/N200/945,12,381/x_relaxed_AR200.txt",
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument(
        "--mu-values",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.4, 1.0],
    )
    parser.add_argument("--workdir", default="mu_sweep")
    parser.add_argument("--output", default="mu_sweep_overlay.png")
    parser.add_argument("--log-velocity-output", default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser


def write_scene(template_path: Path, out_path: Path, mu: float) -> None:
    scene = json.loads(template_path.read_text())
    scene.setdefault("physics", {}).setdefault("soft_contact", {})["mu"] = mu
    out_path.write_text(json.dumps(scene, indent=4))


def run_case(
    exe: Path,
    scene_path: Path,
    init_csv_path: Path,
    case_dir: Path,
    steps: int,
    start: int,
    end: int,
    stride: int,
) -> Path:
    summary_csv = case_dir / "pair_velocity_summary_early.csv"
    aggregate_stats_csv = case_dir / "pair_aggregate_stats.csv"
    pair_csv = case_dir / "pair_distance_early.csv"
    contact_csv = case_dir / "pair_contact_velocity_early.csv"
    output_csv = case_dir / "output.csv"

    cmd = [
        str(exe),
        "--headless",
        "--scene",
        str(scene_path),
        "--init-csv",
        str(init_csv_path),
        "--steps",
        str(steps),
        "--early-pair-diagnostics",
        "--early-pair-start",
        str(start),
        "--early-pair-end",
        str(end),
        "--early-pair-stride",
        str(stride),
        "--early-pair-contact-csv",
        str(contact_csv),
        "--early-pair-distance-csv",
        str(pair_csv),
        "--early-pair-velocity-summary-csv",
        str(summary_csv),
        "--output",
        str(output_csv),
        "--csv-stride",
        "1000",
    ]
    subprocess.run(cmd, check=True)

    pair_df = pd.read_csv(
        pair_csv,
        usecols=["frame", "distance_metric", "v_rel_speed", "v_n", "v_t"],
    )
    pair_df["abs_v_n"] = pair_df["v_n"].abs()
    aggregate_stats = (
        pair_df.groupby("frame")
        .agg(
            mean_v_rel=("v_rel_speed", "mean"),
            median_v_rel=("v_rel_speed", "median"),
            max_v_rel=("v_rel_speed", "max"),
            mean_abs_v_n=("abs_v_n", "mean"),
            median_abs_v_n=("abs_v_n", "median"),
            max_abs_v_n=("abs_v_n", "max"),
            mean_v_t=("v_t", "mean"),
            median_v_t=("v_t", "median"),
            max_v_t=("v_t", "max"),
            mean_distance=("distance_metric", "mean"),
            median_distance=("distance_metric", "median"),
            max_distance=("distance_metric", "max"),
        )
        .reset_index()
    )
    aggregate_stats.to_csv(aggregate_stats_csv, index=False)

    if contact_csv.exists():
        contact_csv.unlink()
    if pair_csv.exists():
        pair_csv.unlink()
    if output_csv.exists():
        output_csv.unlink()

    return aggregate_stats_csv


def plot_six_panel_overlay(
    mu_to_stats: dict[float, Path], output_path: Path, log_velocity: bool = False
) -> None:
    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(15, 9))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    stat_styles = {
        "mean": "-",
        "median": "--",
        "max": ":",
    }

    velocity_specs = [
        ("v_rel", "|v_rel|"),
        ("abs_v_n", "|v_n|"),
        ("v_t", "|v_t|"),
    ]
    distance_specs = [
        ("mean_distance", "mean pair distance"),
        ("median_distance", "median pair distance"),
        ("max_distance", "max pair distance"),
    ]

    for color_idx, (mu, csv_path) in enumerate(mu_to_stats.items()):
        df = pd.read_csv(csv_path)
        color = colors[color_idx % len(colors)]
        mu_label = f"mu={mu:g}"

        for ax, (suffix, title) in zip(axes[0], velocity_specs):
            for stat_name, line_style in stat_styles.items():
                column = f"{stat_name}_{suffix}"
                label = f"{mu_label} {stat_name}"
                ax.plot(
                    df["frame"],
                    df[column],
                    color=color,
                    linestyle=line_style,
                    label=label,
                )
            ax.set_title(title)
            ax.set_ylabel("speed")
            if log_velocity:
                ax.set_yscale("log")

        for ax, (column, title) in zip(axes[1], distance_specs):
            ax.plot(df["frame"], df[column], color=color, linestyle="-", label=mu_label)
            ax.set_title(title)
            ax.set_ylabel("distance")

    for ax in axes[1]:
        ax.set_xlabel("frame")

    for ax in axes[0]:
        ax.legend(fontsize=8, ncol=2)
    for ax in axes[1]:
        ax.legend(fontsize=8)

    title = "All-pair velocity and distance statistics vs soft-contact mu"
    if log_velocity:
        title += " (log velocity scale)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    exe = Path(args.exe)
    scene = Path(args.scene)
    init_csv = Path(args.init_csv)
    workdir = Path(args.workdir)
    if workdir.exists() and not args.reuse_existing:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    mu_to_stats: dict[float, Path] = {}
    for mu in args.mu_values:
        mu_label = str(mu).replace(".", "p")
        case_dir = workdir / f"mu_{mu_label}"
        aggregate_stats_csv = case_dir / "pair_aggregate_stats.csv"
        if args.reuse_existing and aggregate_stats_csv.exists():
            mu_to_stats[mu] = aggregate_stats_csv
            continue

        case_dir.mkdir(parents=True, exist_ok=True)
        scene_path = case_dir / "scene.json"
        write_scene(scene, scene_path, mu)
        aggregate_stats_csv = run_case(
            exe=exe,
            scene_path=scene_path,
            init_csv_path=init_csv,
            case_dir=case_dir,
            steps=args.steps,
            start=args.start,
            end=args.end,
            stride=args.stride,
        )
        mu_to_stats[mu] = aggregate_stats_csv

    plot_six_panel_overlay(mu_to_stats, Path(args.output))
    if args.log_velocity_output:
        plot_six_panel_overlay(
            mu_to_stats, Path(args.log_velocity_output), log_velocity=True
        )


if __name__ == "__main__":
    main()