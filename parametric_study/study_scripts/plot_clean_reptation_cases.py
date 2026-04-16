#!/usr/bin/env python3

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_trial_numbers(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    ordered["trial"] = ordered.groupby(["gap", "mu"]).cumcount()
    return ordered


def stable_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["final_KE"].fillna(np.inf) < 1e-4
    ) & (
        df["net_displacement"].fillna(np.inf).abs() < 2.0
    ) & (
        df["wall_hits"].fillna(np.inf) < 5e4
    )


def format_gap(gap: float) -> str:
    return f"{gap:.3f}".rstrip("0").rstrip(".")


def build_replay_commands(out_dir: Path, worst_trials: pd.DataFrame) -> str:
    lines = []
    for row in worst_trials.itertuples(index=False):
        scene = out_dir / f"scene_AR200_gap{format_gap(row.gap)}_mu{row.mu:.1f}_t{int(row.trial)}.json"
        lines.extend([
            "./build/rigidbody_viewer_3d \\",
            f"  --scene {scene.as_posix()} \\",
            "  --nsc \\",
            "  --nsc-iters 40 \\",
            "  --nsc-beta 0.2 \\",
            "  --nsc-pos-iters 5 \\",
            "  --nsc-pos-psor 50 \\",
            "  --steps 200000",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    out_dir = Path("results/reptation_ar200_thermal_sv0p1_sw0p2")
    combined_path = out_dir / "combined.csv"
    df = pd.read_csv(combined_path)
    df = infer_trial_numbers(df)
    df["stable_trial"] = stable_mask(df)
    df["abs_slide"] = df["net_displacement"].abs()

    unstable = df[np.isclose(df["gap"], 0.001) & np.isclose(df["mu"], 1.0)].copy()
    unstable["nan_ke"] = unstable["final_KE"].isna().astype(int)
    unstable["rank_ke"] = unstable["final_KE"].fillna(-1.0)
    unstable["rank_disp"] = unstable["net_displacement"].fillna(-1.0).abs()
    worst_trials = unstable.sort_values(["nan_ke", "rank_ke", "rank_disp"], ascending=[False, False, False]).head(8)
    replay_text = build_replay_commands(out_dir, worst_trials[["gap", "mu", "trial"]])
    replay_path = out_dir / "worst_gap0p001_mu1p0_render_commands.txt"
    replay_path.write_text(replay_text)

    clean = (
        df.groupby(["gap", "mu"], as_index=False)
        .agg(
            stable_fraction=("stable_trial", "mean"),
            clean_count=("stable_trial", "sum"),
        )
    )
    clean = clean[clean["stable_fraction"] >= 0.7].copy()

    clean_points = df[df["stable_trial"]].merge(clean[["gap", "mu"]], on=["gap", "mu"], how="inner")
    plot_data = (
        clean_points.groupby(["gap", "mu"], as_index=False)
        .agg(
            mean_abs_slide=("abs_slide", "mean"),
            std_abs_slide=("abs_slide", "std"),
            n=("abs_slide", "size"),
        )
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    mu_values = sorted(plot_data["mu"].unique())
    colors = plt.cm.cividis(np.linspace(0.15, 0.9, len(mu_values)))
    for color, mu in zip(colors, mu_values):
        sub = plot_data[plot_data["mu"] == mu].sort_values("gap")
        ax.errorbar(
            sub["gap"],
            sub["mean_abs_slide"],
            yerr=sub["std_abs_slide"].fillna(0.0),
            marker="o",
            linewidth=1.5,
            capsize=3,
            color=color,
            label=f"mu={mu:g}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Gap")
    ax.set_ylabel("Sliding length |net displacement|")
    ax.set_title("Clean-looking reptation cases")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    plot_path = out_dir / "clean_sliding_length_vs_gap_mu.png"
    fig.savefig(plot_path, dpi=200)

    summary_path = out_dir / "clean_sliding_length_summary.csv"
    plot_data.to_csv(summary_path, index=False)

    print("Worst replay commands:")
    print(replay_text)
    print(f"Plot written to: {plot_path}")
    print(f"Summary written to: {summary_path}")
    print(f"Replay commands written to: {replay_path}")


if __name__ == "__main__":
    main()