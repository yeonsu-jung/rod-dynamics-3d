#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_tag(tag: str) -> tuple[float, int]:
    parts = tag.split("_")
    mu = None
    trial = None
    for part in parts:
        if part.startswith("mu"):
            mu = float(part[2:])
        elif part.startswith("t") and part[1:].isdigit():
            trial = int(part[1:])
    if mu is None or trial is None:
        raise ValueError(f"Could not parse mu/trial from tag: {tag}")
    return mu, trial


def main() -> None:
    input_path = Path("results/reptation_ar200_thermal_sv0p1_sw0p2_tangent_post_full/tangent_stop_summary.csv")
    output_path = Path("results/reptation_ar200_thermal_sv0p1_sw0p2_tangent_post_full/tangent_stop_time_vs_mu.png")

    df = pd.read_csv(input_path)
    parsed = df["tag"].map(parse_tag)
    df["mu"] = parsed.map(lambda item: item[0])
    df["trial"] = parsed.map(lambda item: item[1])

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for trial in sorted(df["trial"].unique()):
        sub = df[df["trial"] == trial].sort_values("mu")
        ax.plot(sub["mu"], sub["stop_time"], marker="o", linewidth=1.6, label=f"t{trial}")

    ax.set_xlabel("mu")
    ax.set_ylabel("Tangential-velocity stop time")
    ax.set_title("Postprocessed stop time from tangential velocity")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()