#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def axis_y_from_quat(qw: float, qx: float, qy: float, qz: float) -> tuple[float, float, float]:
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return (
        2.0 * (xy - wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz + wx),
    )


def load_perrod(path: Path, rod_id: int, dt: float) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df = df[df["rod"] == rod_id].copy()
    if df.empty:
        raise ValueError(f"No rod={rod_id} rows in {path}")

    axes = df.apply(
        lambda row: axis_y_from_quat(row["qw"], row["qx"], row["qy"], row["qz"]),
        axis=1,
        result_type="expand",
    )
    axes.columns = ["ax", "ay", "az"]
    df = pd.concat([df.reset_index(drop=True), axes], axis=1)
    df["tangent_vel"] = (
        df["vx"] * df["ax"] + df["vy"] * df["ay"] + df["vz"] * df["az"]
    ).abs()
    df["time"] = df["frame"] * dt
    return df


def first_stop_row(df: pd.DataFrame, threshold: float) -> pd.Series:
    hits = df[df["tangent_vel"] < threshold]
    if hits.empty:
        return df.iloc[-1]
    return hits.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze reptation stopping from postprocessed tangential velocity"
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing perrod_*.csv files")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Tangential-velocity stop threshold")
    parser.add_argument("--dt", type=float, default=1e-4, help="Simulation timestep used for the runs")
    parser.add_argument("--rod-id", type=int, default=0, help="Tracked rod id")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows: list[dict[str, float | str]] = []
    for path in sorted(input_dir.glob("perrod_*.csv")):
        tag = path.stem[len("perrod_"):]
        df = load_perrod(path, args.rod_id, args.dt)
        stop = first_stop_row(df, args.threshold)
        rows.append(
            {
                "tag": tag,
                "stop_frame": int(stop["frame"]),
                "stop_time": float(stop["time"]),
                "stop_py": float(stop["py"]),
                "stop_tangent_vel": float(stop["tangent_vel"]),
                "final_frame": int(df.iloc[-1]["frame"]),
                "final_time": float(df.iloc[-1]["time"]),
                "final_py": float(df.iloc[-1]["py"]),
                "final_tangent_vel": float(df.iloc[-1]["tangent_vel"]),
                "min_tangent_vel": float(df["tangent_vel"].min()),
                "max_tangent_vel": float(df["tangent_vel"].max()),
            }
        )

    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()