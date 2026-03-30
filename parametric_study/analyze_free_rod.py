#!/usr/bin/env python3
"""analyze_free_rod.py

Compute and plot cumulative sliding length over time for free-rod perturbation
trajectories, grouped by metric type (MinFSA / MaxFSA / MinFTA / MaxFTA).

Sliding length = cumulative arc length of center-of-mass trajectory:
  L(t) = sum_{i=1}^{t} |r(i) - r(i-1)|

Handles two data sources:
  --bundle-csv   free_rod_all.csv  (cols: N,AR,seed_id,metric,rod,mu,frame,rod,px,...)
  --runs-dir     runs root, scans for per-entry free_rod.csv files
                 (cols: mu,frame,rod,px,...; metadata parsed from directory name)

Both can be supplied together; data is merged.

Usage examples:
  python3 parametric_study/analyze_free_rod.py \\
      --bundle-csv /path/to/free_rod_all.csv \\
      --out-dir plots/

  python3 parametric_study/analyze_free_rod.py \\
      --runs-dir /path/to/runs  --job-name free_rod_sweep \\
      --filter-n 500 1000 --out-dir plots/

Note: the full bundle CSV (~1.5M rows) requires ~2 GB RAM.
Run on a compute node or use --filter-n/--filter-ar/--max-rows to subset.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────────────

METRICS = ["MinFSA", "MaxFSA", "MinFTA", "MaxFTA"]
COLORS = {
    "MinFSA": "#1f77b4",
    "MaxFSA": "#ff7f0e",
    "MinFTA": "#2ca02c",
    "MaxFTA": "#d62728",
}
LABELS = {
    "MinFSA": "Min rotation (MinFSA)",
    "MaxFSA": "Max rotation (MaxFSA)",
    "MinFTA": "Min translation (MinFTA)",
    "MaxFTA": "Max translation (MaxFTA)",
}

GROUP_COLS = ["N", "AR", "seed_id", "metric", "rod", "mu"]
_KEEP = GROUP_COLS + ["frame", "px", "py", "pz"]

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_bundle_csv(path: Path, max_rows: int = None) -> pd.DataFrame:
    """Load free_rod_all.csv produced by --bundle-all mode.

    Header: N,AR,seed_id,metric,rod,mu,frame,rod,px,py,pz,...
    Pandas auto-renames the duplicate 'rod' column to 'rod.1' — we drop it.
    """
    df = pd.read_csv(path, comment="#", low_memory=False, nrows=max_rows)
    df = df.drop(columns=["rod.1"], errors="ignore")
    df["mu"] = df["mu"].astype(float)
    df["rod"] = df["rod"].astype(int)
    return df[_KEEP]


_DIR_RE = re.compile(
    r"^\d{8}-\d{6}_N(\d+)_(.+?)_AR(\d+)_(Min|Max)(FSA|FTA)_rod(\d+)$"
)

def load_per_entry_csvs(runs_dir: Path, job_name: str = "free_rod_sweep") -> pd.DataFrame:
    """Scan runs_dir/job_name for per-entry free_rod.csv files.

    Directory name pattern: <ts>_N{N}_{seed_id}_AR{ar}_(Min|Max)(FSA|FTA)_rod{rod}
    CSV columns: mu,frame,rod,px,py,pz,...
    """
    job_dir = runs_dir / job_name
    if not job_dir.exists():
        print(f"WARNING: runs dir not found: {job_dir}")
        return pd.DataFrame()

    frames = []
    skipped = 0
    for csv_path in sorted(job_dir.glob("*/free_rod.csv")):
        m = _DIR_RE.match(csv_path.parent.name)
        if not m:
            skipped += 1
            continue
        N, seed_id, AR, minmax, fsa_fta, rod = m.groups()
        metric = minmax + fsa_fta
        try:
            df = pd.read_csv(csv_path, comment="#", low_memory=False)
        except Exception as exc:
            print(f"WARNING: could not read {csv_path}: {exc}")
            skipped += 1
            continue
        if df.empty or "px" not in df.columns:
            skipped += 1
            continue
        df["N"]       = int(N)
        df["AR"]      = int(AR)
        df["seed_id"] = seed_id
        df["metric"]  = metric
        df["rod"]     = int(rod)
        df["mu"]      = df["mu"].astype(float)
        frames.append(df[_KEEP])

    print(f"  Loaded {len(frames)} per-entry CSVs  ({skipped} skipped/missing)")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Sliding length ─────────────────────────────────────────────────────────────

def compute_sliding_length(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative arc-length of CoM trajectory per trajectory group.

    Adds column 'slide' (cumulative displacement, same units as px/py/pz).
    """
    df = df.sort_values(GROUP_COLS + ["frame"]).copy()
    grp = df.groupby(GROUP_COLS, sort=False)

    # diff within each group; first row per group → NaN → 0
    dx = grp["px"].diff().fillna(0.0)
    dy = grp["py"].diff().fillna(0.0)
    dz = grp["pz"].diff().fillna(0.0)

    df["_step"] = np.sqrt(dx**2 + dy**2 + dz**2)
    df["slide"] = df.groupby(GROUP_COLS, sort=False)["_step"].cumsum()
    return df.drop(columns=["_step"])


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_sliding_over_time(df: pd.DataFrame, out_path: Path,
                           filter_n=None, filter_ar=None) -> None:
    """Mean ± std sliding length vs frame, one panel per mu, curves by metric."""
    sub = df.copy()
    if filter_n:
        sub = sub[sub["N"].isin(filter_n)]
    if filter_ar:
        sub = sub[sub["AR"].isin(filter_ar)]
    if sub.empty:
        print("No data after filtering for sliding-over-time plot.")
        return

    mus = sorted(sub["mu"].unique())
    fig, axes = plt.subplots(1, len(mus), figsize=(3.5 * len(mus), 4), sharey=True)
    if len(mus) == 1:
        axes = [axes]

    for ax, mu in zip(axes, mus):
        mdf = sub[sub["mu"] == mu]
        for metric in METRICS:
            g = mdf[mdf["metric"] == metric]
            if g.empty:
                continue
            stats = g.groupby("frame")["slide"].agg(["mean", "std"])
            frames = stats.index.values
            mean = stats["mean"].values
            std  = stats["std"].fillna(0).values
            ax.plot(frames, mean, color=COLORS[metric], lw=1.5, label=LABELS[metric])
            ax.fill_between(frames, mean - std, mean + std,
                            color=COLORS[metric], alpha=0.15)
        ax.set_title(f"μ = {mu}", fontsize=9)
        ax.set_xlabel("Frame")
        ax.grid(True, lw=0.4, alpha=0.5)

    axes[0].set_ylabel("Cumulative sliding length (L)")
    axes[-1].legend(fontsize=7, loc="upper left")

    parts = []
    if filter_n:
        parts.append(f"N ∈ {filter_n}")
    if filter_ar:
        parts.append(f"AR ∈ {filter_ar}")
    title = "Free-rod sliding length over time"
    if parts:
        title += "  —  " + ", ".join(parts)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_final_slide_distributions(df: pd.DataFrame, out_path: Path) -> None:
    """Violin plot of final sliding length per metric for each mu."""
    max_frame = df["frame"].max()
    final = df[df["frame"] == max_frame]
    if final.empty:
        print("No data at max frame for distribution plot.")
        return

    mus = sorted(final["mu"].unique())
    fig, axes = plt.subplots(1, len(mus), figsize=(3.5 * len(mus), 4), sharey=True)
    if len(mus) == 1:
        axes = [axes]

    for ax, mu in zip(axes, mus):
        mdf = final[final["mu"] == mu]
        data = [mdf[mdf["metric"] == m]["slide"].dropna().values for m in METRICS]
        parts = ax.violinplot([d if len(d) > 0 else [0] for d in data],
                              positions=range(len(METRICS)), showmedians=True)
        for pc, m in zip(parts["bodies"], METRICS):
            pc.set_facecolor(COLORS[m])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels(METRICS, fontsize=8, rotation=20, ha="right")
        ax.set_title(f"μ = {mu}", fontsize=9)
        ax.grid(True, axis="y", lw=0.4, alpha=0.5)

    axes[0].set_ylabel(f"Sliding length at frame {max_frame}")
    fig.suptitle("Final sliding length distribution by metric", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_sliding_by_N(df: pd.DataFrame, out_path: Path, mu: float = 0.1) -> None:
    """Mean sliding length vs frame, one panel per N, curves by metric, for a single mu."""
    sub = df[df["mu"] == mu]
    if sub.empty:
        return
    Ns = sorted(sub["N"].unique())
    ncols = min(4, len(Ns))
    nrows = (len(Ns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows),
                             sharey=False, squeeze=False)

    for idx, N in enumerate(Ns):
        ax = axes[idx // ncols][idx % ncols]
        ndf = sub[sub["N"] == N]
        for metric in METRICS:
            g = ndf[ndf["metric"] == metric]
            if g.empty:
                continue
            stats = g.groupby("frame")["slide"].agg(["mean", "std"])
            frames = stats.index.values
            mean = stats["mean"].values
            std  = stats["std"].fillna(0).values
            ax.plot(frames, mean, color=COLORS[metric], lw=1.5, label=metric)
            ax.fill_between(frames, mean - std, mean + std,
                            color=COLORS[metric], alpha=0.15)
        ax.set_title(f"N = {N}", fontsize=9)
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylabel("Sliding L", fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.5)

    # Remove unused axes
    for idx in range(len(Ns), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Sliding length by N  (μ = {mu})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-csv", type=Path, default=None,
                    help="free_rod_all.csv from --bundle-all mode.")
    ap.add_argument("--runs-dir",   type=Path, default=None,
                    help="Runs root directory containing per-entry free_rod.csv files.")
    ap.add_argument("--job-name",   type=str,  default="free_rod_sweep")
    ap.add_argument("--out-dir",    type=Path, default=Path("plots_free_rod"),
                    help="Output directory for plots (default: plots_free_rod/).")
    ap.add_argument("--max-rows",   type=int,  default=None,
                    help="Limit rows read from bundle CSV (for testing on login nodes).")
    ap.add_argument("--filter-n",   type=int,  nargs="+", default=None,
                    help="Restrict plots to these N values.")
    ap.add_argument("--filter-ar",  type=int,  nargs="+", default=None,
                    help="Restrict plots to these AR values.")
    ap.add_argument("--by-n-mu",    type=float, default=0.1,
                    help="Friction value used for the per-N breakdown plot (default 0.1).")
    args = ap.parse_args()

    dfs = []

    if args.bundle_csv:
        if args.bundle_csv.exists():
            print(f"Loading bundle CSV: {args.bundle_csv}")
            dfs.append(load_bundle_csv(args.bundle_csv, max_rows=args.max_rows))
            print(f"  {len(dfs[-1]):,} rows")
        else:
            print(f"WARNING: bundle CSV not found: {args.bundle_csv}")

    if args.runs_dir:
        print(f"Loading per-entry CSVs: {args.runs_dir}")
        pe = load_per_entry_csvs(args.runs_dir, args.job_name)
        if not pe.empty:
            dfs.append(pe)
            print(f"  {len(pe):,} rows")

    if not dfs:
        raise SystemExit("No data loaded. Supply --bundle-csv and/or --runs-dir.")

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    print(f"\nTotal rows: {len(df):,}")
    print(f"N values:   {sorted(df['N'].unique())}")
    print(f"AR values:  {sorted(df['AR'].unique())}")
    print(f"Metrics:    {sorted(df['metric'].unique())}")
    print(f"Frictions:  {sorted(df['mu'].unique())}")

    print("\nComputing sliding lengths...")
    df = compute_sliding_length(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    plot_sliding_over_time(
        df,
        args.out_dir / "sliding_over_time.png",
        filter_n=args.filter_n,
        filter_ar=args.filter_ar,
    )
    plot_final_slide_distributions(
        df,
        args.out_dir / "sliding_final_dist.png",
    )
    plot_sliding_by_N(
        df,
        args.out_dir / f"sliding_by_N_mu{args.by_n_mu}.png",
        mu=args.by_n_mu,
    )
    print("Done.")


if __name__ == "__main__":
    main()
