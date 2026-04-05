from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = ArgumentParser(description="Plot all-pair closest-point velocity summaries.")
    parser.add_argument("--summary-csv", default="pair_velocity_summary_early.csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    df = pd.read_csv(Path(args.summary_csv))

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

    axes[0].plot(df["frame"], df["mean_v_rel_speed"], label="mean |v_rel|")
    axes[0].plot(df["frame"], df["mean_v_t"], label="mean v_t")
    axes[0].set_ylabel("speed")
    axes[0].legend()

    axes[1].plot(df["frame"], df["mean_v_n"], label="mean v_n")
    axes[1].plot(df["frame"], df["mean_abs_v_n"], label="mean |v_n|")
    axes[1].set_ylabel("normal")
    axes[1].legend()

    axes[2].plot(df["frame"], df["mean_signed_gap"], label="mean signed gap")
    axes[2].plot(df["frame"], df["mean_distance_metric"], label="mean distance")
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("gap / distance")
    axes[2].legend()

    fig.suptitle("All-pair closest-point velocity summary")
    fig.tight_layout()
    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()

