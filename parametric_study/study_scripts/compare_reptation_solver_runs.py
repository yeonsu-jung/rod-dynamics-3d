#!/usr/bin/env python3

from pathlib import Path
import argparse

import pandas as pd


def load_summary(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "tangent_stop_summary.csv")
    parts = df["tag"].str.split("_")
    df["gap"] = parts.apply(lambda tokens: float(next(item[3:] for item in tokens if item.startswith("gap"))))
    df["mu"] = parts.apply(lambda tokens: float(next(item[2:] for item in tokens if item.startswith("mu"))))
    if "resolved" in df.columns:
        df["resolved"] = df["resolved"].astype(bool)
    else:
        df["resolved"] = df["stop_time"] < (df["final_time"] - 1e-9)
    df["sliding_length"] = df["stop_py"].abs()
    return df.groupby(["gap", "mu"], as_index=False).agg(
        resolved_count=("resolved", "sum"),
        median_stop_time=("stop_time", "median"),
        median_sliding_length=("sliding_length", "median"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare soft-contact and NSC reptation first-stop summaries")
    parser.add_argument("--nsc-dir", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True,
                        help="Pair of dataset label and NSC result directory")
    parser.add_argument("--soft-dir", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True,
                        help="Pair of dataset label and soft-contact result directory")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    nsc_map = {label: Path(path) for label, path in args.nsc_dir}
    soft_map = {label: Path(path) for label, path in args.soft_dir}
    labels = sorted(set(nsc_map) & set(soft_map))
    if not labels:
        raise SystemExit("No overlapping labels between --nsc-dir and --soft-dir")

    rows = []
    for label in labels:
        nsc = load_summary(nsc_map[label])
        soft = load_summary(soft_map[label])
        merged = nsc.merge(soft, on=["gap", "mu"], suffixes=("_nsc", "_soft"))
        merged.insert(0, "label", label)
        merged["resolved_count_soft_minus_nsc"] = merged["resolved_count_soft"] - merged["resolved_count_nsc"]
        merged["median_stop_time_soft_minus_nsc"] = merged["median_stop_time_soft"] - merged["median_stop_time_nsc"]
        merged["median_sliding_soft_minus_nsc"] = merged["median_sliding_length_soft"] - merged["median_sliding_length_nsc"]
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out_path)
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()