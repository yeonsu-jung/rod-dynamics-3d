from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_contact_average(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["contact_speed"] = np.sqrt(
        df["v_rel_x"] ** 2 + df["v_rel_y"] ** 2 + df["v_rel_z"] ** 2
    )
    grouped = (
        df.groupby("frame")
        .agg(
            contact_mean_speed=("contact_speed", "mean"),
            contact_mean_v_n=("v_n", "mean"),
            contact_mean_abs_v_n=("v_n", lambda s: np.abs(s).mean()),
            contact_mean_v_t=("v_t", "mean"),
            contact_count=("body_a", "count"),
        )
        .reset_index()
    )
    return grouped


def load_pair_average(path: Path, cutoff: float | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if cutoff is not None:
      df = df[df["signed_gap"] <= cutoff].copy()
    df["abs_v_n"] = np.abs(df["v_n"])
    grouped = (
        df.groupby("frame")
        .agg(
            pair_mean_speed=("v_rel_speed", "mean"),
            pair_mean_v_n=("v_n", "mean"),
            pair_mean_abs_v_n=("abs_v_n", "mean"),
            pair_mean_v_t=("v_t", "mean"),
            pair_count=("body_a", "count"),
        )
        .reset_index()
    )
    return grouped


def main() -> None:
    parser = ArgumentParser(description="Compare contact-only and all-pair averaged relative velocities.")
    parser.add_argument("--contact-csv", default="pair_contact_velocity_early.csv")
    parser.add_argument("--pair-csv", default="pair_distance_early.csv")
    parser.add_argument("--gap-cutoff", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    contact = load_contact_average(Path(args.contact_csv))
    pair = load_pair_average(Path(args.pair_csv), args.gap_cutoff)
    merged = contact.merge(pair, on="frame", how="outer").sort_values("frame")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

    axes[0].plot(merged["frame"], merged["contact_mean_speed"], label="contact mean |v_rel|")
    axes[0].plot(merged["frame"], merged["pair_mean_speed"], label="all-pair mean |v_rel|")
    axes[0].plot(merged["frame"], merged["contact_mean_v_t"], label="contact mean v_t", linestyle="--")
    axes[0].plot(merged["frame"], merged["pair_mean_v_t"], label="all-pair mean v_t", linestyle=":")
    axes[0].set_ylabel("speed")
    axes[0].legend()

    axes[1].plot(merged["frame"], merged["contact_mean_abs_v_n"], label="contact mean |v_n|")
    axes[1].plot(merged["frame"], merged["pair_mean_abs_v_n"], label="all-pair mean |v_n|")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("normal speed")
    axes[1].legend()

    cutoff_label = "all pairs" if args.gap_cutoff is None else f"signed gap <= {args.gap_cutoff}"
    fig.suptitle(f"Contact vs {cutoff_label} averaged relative velocities")
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()