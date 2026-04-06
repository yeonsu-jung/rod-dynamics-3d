from __future__ import annotations

import json
import math
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
    parser.add_argument("--dt", type=float, default=1e-7)
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
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-velocity-output", default=None)
    parser.add_argument("--loglog-distance-output", default=None)
    parser.add_argument("--collision-output", default=None)
    parser.add_argument(
        "--free-volume-exe",
        default="/Users/yeonsu/GitHub/rod-free-volume/build/rod_free_volume",
    )
    parser.add_argument("--free-volume-output", default=None)
    parser.add_argument("--free-volume-samples", type=int, default=360)
    parser.add_argument("--free-volume-bisection-steps", type=int, default=16)
    parser.add_argument("--free-volume-theta-coarse", type=int, default=48)
    parser.add_argument("--free-volume-threads", type=int, default=None)
    parser.add_argument("--free-volume-verbose", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    return parser


def format_float_tag(value: float) -> str:
    mantissa, exponent = f"{value:.15e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = str(int(exponent))
    return f"{mantissa}e{exponent}"


def resolve_output_path(path_str: str | None, workdir: Path) -> Path | None:
    if path_str is None:
        return None
    output_path = Path(path_str)
    if not output_path.is_absolute() and output_path.parent == Path("."):
        output_path = workdir / output_path.name
    return output_path


def write_scene(template_path: Path, out_path: Path, mu: float, use_nsc: bool = False) -> None:
    scene = json.loads(template_path.read_text())
    physics = scene.setdefault("physics", {})
    if use_nsc:
        physics.setdefault("nsc", {})["enabled"] = True
    else:
        physics.setdefault("soft_contact", {})["mu"] = mu
    out_path.write_text(json.dumps(scene, indent=4))


def compute_sample_count(start: int, end: int, stride: int) -> int:
    if end < start:
        return 1
    return ((end - start) // max(1, stride)) + 1


def quaternion_to_axis_y(quat: list[float]) -> tuple[float, float, float]:
    w, x, y, z = quat
    axis_x = 2.0 * (x * y - z * w)
    axis_y = 1.0 - 2.0 * (x * x + z * z)
    axis_z = 2.0 * (y * z + x * w)
    norm = math.sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
    if norm == 0.0:
        return (0.0, 1.0, 0.0)
    return (axis_x / norm, axis_y / norm, axis_z / norm)


def write_free_volume_input(snapshot: dict, out_path: Path) -> None:
    capsules = [body for body in snapshot["bodies"] if body.get("shape") == "capsule"]
    if not capsules:
        raise ValueError("Snapshot does not contain any capsule bodies")

    radius = float(capsules[0]["radius"])
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# diameter = {2.0 * radius:.12g}\n")
        for body in capsules:
            axis = quaternion_to_axis_y(body["quat"])
            half_height = float(body["halfHeight"])
            pos_x, pos_y, pos_z = (float(value) for value in body["pos"])
            dx = axis[0] * half_height
            dy = axis[1] * half_height
            dz = axis[2] * half_height
            handle.write(
                f"{pos_x - dx:.12g} {pos_y - dy:.12g} {pos_z - dz:.12g} "
                f"{pos_x + dx:.12g} {pos_y + dy:.12g} {pos_z + dz:.12g}\n"
            )


def compute_free_volume_stats(
    free_volume_exe: Path,
    snapshots_path: Path,
    case_dir: Path,
    samples: int,
    bisection_steps: int,
    theta_coarse: int,
    threads: int | None,
    verbose: bool,
) -> Path:
    free_volume_csv = case_dir / "free_volume_aggregate_stats.csv"
    scratch_dir = case_dir / "free_volume_snapshots"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    try:
        with snapshots_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                snapshot = json.loads(stripped)
                frame = int(snapshot["frame"])
                frame_input = scratch_dir / f"frame_{frame:07d}.txt"
                frame_output = scratch_dir / f"frame_{frame:07d}_free_volume.csv"
                write_free_volume_input(snapshot, frame_input)

                cmd = [
                    str(free_volume_exe),
                    "--output",
                    str(frame_output),
                    "--samples",
                    str(samples),
                    "--bisection-steps",
                    str(bisection_steps),
                    "--theta-coarse",
                    str(theta_coarse),
                ]
                if threads is not None:
                    cmd.extend(["--threads", str(threads)])
                if verbose:
                    cmd.append("--verbose")
                cmd.append(str(frame_input))
                subprocess.run(cmd, check=True)

                frame_df = pd.read_csv(frame_output)
                rows.append(
                    {
                        "frame": frame,
                        "mean_free_translation_area": frame_df["free_translation_area"].mean(),
                        "median_free_translation_area": frame_df["free_translation_area"].median(),
                        "max_free_translation_area": frame_df["free_translation_area"].max(),
                        "mean_free_solid_angle": frame_df["free_solid_angle"].mean(),
                        "median_free_solid_angle": frame_df["free_solid_angle"].median(),
                        "max_free_solid_angle": frame_df["free_solid_angle"].max(),
                    }
                )
                if frame_input.exists():
                    frame_input.unlink()
                if frame_output.exists():
                    frame_output.unlink()
    finally:
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)

    free_volume_df = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
    free_volume_df.to_csv(free_volume_csv, index=False)
    return free_volume_csv


def run_case(
    exe: Path,
    scene_path: Path,
    init_csv_path: Path,
    case_dir: Path,
    dt: float,
    steps: int,
    start: int,
    end: int,
    stride: int,
    free_volume_exe: Path | None,
    free_volume_samples: int,
    free_volume_bisection_steps: int,
    free_volume_theta_coarse: int,
    free_volume_threads: int | None,
    free_volume_verbose: bool,
    threads: int,
    use_nsc: bool = False,
    nsc_mu: float | None = None,
) -> Path:
    summary_csv = case_dir / "pair_velocity_summary_early.csv"
    aggregate_stats_csv = case_dir / "pair_aggregate_stats.csv"
    pair_csv = case_dir / "pair_distance_early.csv"
    contact_csv = case_dir / "pair_contact_velocity_early.csv"
    output_csv = case_dir / "output.csv"
    snapshots_path = case_dir / "snapshots.ndjson"

    cmd = [
        str(exe),
        "--headless",
        "--scene",
        str(scene_path),
        "--init-csv",
        str(init_csv_path),
        "--dt",
        str(dt),
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
        "--snap-stride",
        str(stride),
        "--snap-frames",
        str(compute_sample_count(start, end, stride)),
        "--snap-start",
        str(start),
        "--snap-path",
        str(snapshots_path),
        "--csv-stride",
        "1000",
        "--threads",
        str(threads)
    ]
    if use_nsc:
        cmd.append("--nsc")
        if nsc_mu is not None:
            cmd.extend(["--nsc-mu", str(nsc_mu)])
    subprocess.run(cmd, check=True)

    pair_df = pd.read_csv(
        pair_csv,
        usecols=["frame", "distance_metric", "v_rel_speed", "v_n", "v_t"],
    )
    collision_counts = pd.read_csv(contact_csv, usecols=["frame"])
    collision_counts = (
        collision_counts.groupby("frame").size().rename("collision_count").reset_index()
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
    aggregate_stats = aggregate_stats.merge(collision_counts, on="frame", how="left")
    aggregate_stats["collision_count"] = aggregate_stats["collision_count"].fillna(0).astype(int)

    if free_volume_exe is not None:
        free_volume_csv = compute_free_volume_stats(
            free_volume_exe=free_volume_exe,
            snapshots_path=snapshots_path,
            case_dir=case_dir,
            samples=free_volume_samples,
            bisection_steps=free_volume_bisection_steps,
            theta_coarse=free_volume_theta_coarse,
            threads=free_volume_threads,
            verbose=free_volume_verbose,
        )
        free_volume_df = pd.read_csv(free_volume_csv)
        aggregate_stats = aggregate_stats.merge(free_volume_df, on="frame", how="left")

    aggregate_stats.to_csv(aggregate_stats_csv, index=False)

    if contact_csv.exists():
        contact_csv.unlink()
    if pair_csv.exists():
        pair_csv.unlink()
    if output_csv.exists():
        output_csv.unlink()
    if snapshots_path.exists():
        snapshots_path.unlink()

    return aggregate_stats_csv


def plot_six_panel_overlay(
    mu_to_stats: dict[float, Path], output_path: Path, log_velocity: bool = False,
    loglog_distance: bool = False,
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
            if loglog_distance:
                ax.set_xscale("log")
                ax.set_yscale("log")

    for ax in axes[1]:
        ax.set_xlabel("frame")

    for ax in axes[0]:
        ax.legend(fontsize=8, ncol=2)
    for ax in axes[1]:
        ax.legend(fontsize=8)

    title = "All-pair velocity and distance statistics vs soft-contact mu"
    if log_velocity:
        title += " (log velocity scale)"
    if loglog_distance:
        title += " (log-log distance scale)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def plot_collision_overlay(mu_to_stats: dict[float, Path], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for mu, csv_path in mu_to_stats.items():
        df = pd.read_csv(csv_path)
        ax.plot(df["frame"], df["collision_count"], label=f"mu={mu:g}")

    ax.set_xlabel("frame")
    ax.set_ylabel("collision count")
    ax.set_title("Collision counts vs soft-contact mu")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def plot_free_volume_overlay(mu_to_stats: dict[float, Path], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(15, 8))
    specs = [
        ("mean_free_translation_area", "mean free translation area"),
        ("median_free_translation_area", "median free translation area"),
        ("max_free_translation_area", "max free translation area"),
        ("mean_free_solid_angle", "mean free solid angle"),
        ("median_free_solid_angle", "median free solid angle"),
        ("max_free_solid_angle", "max free solid angle"),
    ]

    for mu, csv_path in mu_to_stats.items():
        df = pd.read_csv(csv_path)
        if any(column not in df.columns for column, _ in specs):
            continue
        for ax, (column, title) in zip(axes.flat, specs):
            ax.plot(df["frame"], df[column], label=f"mu={mu:g}")
            ax.set_title(title)
            ax.set_ylabel(column)

    for ax in axes[1]:
        ax.set_xlabel("frame")
    for ax in axes.flat:
        ax.legend(fontsize=8)

    fig.suptitle("Free-volume statistics vs soft-contact mu")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    exe = Path(args.exe)
    scene = Path(args.scene)
    init_csv = Path(args.init_csv)
    dt_tag = format_float_tag(args.dt)
    workdir = Path(args.workdir) if args.workdir else Path(f"mu_sweep_dt{dt_tag}")
    if workdir.exists() and not args.reuse_existing:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    default_output = workdir / f"compare_pair_velocity_averages_dt{dt_tag}.png"
    output_path = resolve_output_path(args.output, workdir)
    log_velocity_output_path = resolve_output_path(
        args.log_velocity_output,
        workdir,
    )
    loglog_distance_output_path = resolve_output_path(
        args.loglog_distance_output,
        workdir,
    )
    collision_output_path = resolve_output_path(
        args.collision_output,
        workdir,
    )
    free_volume_output_path = resolve_output_path(
        args.free_volume_output,
        workdir,
    )
    free_volume_exe = Path(args.free_volume_exe) if args.free_volume_exe else None
    if free_volume_exe is not None and not free_volume_exe.exists():
        print(f"[free-volume] executable not found, skipping: {free_volume_exe}")
        free_volume_exe = None

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
            dt=args.dt,
            steps=args.steps,
            start=args.start,
            end=args.end,
            stride=args.stride,
            free_volume_exe=free_volume_exe,
            free_volume_samples=args.free_volume_samples,
            free_volume_bisection_steps=args.free_volume_bisection_steps,
            free_volume_theta_coarse=args.free_volume_theta_coarse,
            free_volume_threads=args.free_volume_threads,
            free_volume_verbose=args.free_volume_verbose,
            thread=args.threads
        )
        mu_to_stats[mu] = aggregate_stats_csv

    plot_six_panel_overlay(mu_to_stats, output_path or default_output)
    if log_velocity_output_path:
        plot_six_panel_overlay(
            mu_to_stats, log_velocity_output_path, log_velocity=True
        )
    if loglog_distance_output_path:
        plot_six_panel_overlay(
            mu_to_stats, loglog_distance_output_path, loglog_distance=True
        )
    if collision_output_path:
        plot_collision_overlay(mu_to_stats, collision_output_path)
    if free_volume_output_path:
        plot_free_volume_overlay(mu_to_stats, free_volume_output_path)


if __name__ == "__main__":
    main()