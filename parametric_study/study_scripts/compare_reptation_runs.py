#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def extract_tag_value(tag: str, prefix: str) -> float:
    for part in tag.split("_"):
        if part.startswith(prefix):
            return float(part[len(prefix) :])
    raise ValueError(f"Missing {prefix!r} in tag {tag!r}")


def summarize(dataset_name: str, run_dir: Path, mode: str, input_name: str) -> pd.DataFrame:
    df = pd.read_csv(run_dir / input_name)
    df["gap"] = df["tag"].map(lambda tag: extract_tag_value(tag, "gap"))
    df["mu"] = df["tag"].map(lambda tag: extract_tag_value(tag, "mu"))
    if "resolved" in df.columns:
        df["resolved"] = df["resolved"].astype(bool)
    else:
        df["resolved"] = df["stop_time"] < (df["final_time"] - 1e-9)
    df["sliding_length"] = df["stop_py"].abs()
    grouped = df.groupby(["gap", "mu"], as_index=False).agg(
        resolved=("resolved", "sum"),
        median_stop_time=("stop_time", "median"),
        median_sliding=("sliding_length", "median"),
    )
    grouped.insert(0, "mode", mode)
    grouped.insert(0, "dataset", dataset_name)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reptation run summaries across datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("NAME", "DIR"),
        required=True,
        help="Dataset name and run directory containing tangent stop summaries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path. Prints CSV to stdout if omitted.",
    )
    args = parser.parse_args()

    frames = []
    for dataset_name, directory in args.dataset:
        run_dir = Path(directory)
        frames.append(summarize(dataset_name, run_dir, "first", "tangent_stop_summary.csv"))
        frames.append(
            summarize(
                dataset_name,
                run_dir,
                "sustained",
                "tangent_stop_summary_sustained_w10.csv",
            )
        )

    out = pd.concat(frames, ignore_index=True)
    if args.output is not None:
        out.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")
    else:
        print(out.to_csv(index=False), end="")


if __name__ == "__main__":
    main()