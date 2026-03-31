#!/usr/bin/env python3
"""analyze_free_rod_msd.py

Analyze free-rod endpoint trajectories from submit_free_rod.py local runs.

Computes per-rod:
  - Center-of-mass MSD(t)
  - MSD decomposed into parallel (along initial rod axis) and perpendicular components
  - Rod orientation angular MSD

Produces plots:
  1. MSD vs time, averaged over seeds, one panel per N, lines = AR, subplots = friction
  2. MSD vs time, averaged over seeds, one panel per AR, lines = friction
  3. Long-time diffusion coefficient D vs AR for each N (from MSD slope)
  4. D_parallel / D_perp ratio vs AR
  5. D vs N for each AR
  6. Angular MSD vs time
"""

import argparse
import re
import sys
from pathlib import Path

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

DIR_RE = re.compile(
    r"^\d{8}-\d{6}_N(\d+)_(\d+_\d+_\d+)_AR(\d+)_(\w+)_rod(\d+)$"
)

MU_FILE_RE = re.compile(r"free_rod_endpoints_mu([0-9pm]+)\.csv")


def parse_mu_tag(tag: str) -> float:
    return float(tag.replace("p", ".").replace("m", "-"))


def load_endpoints(csv_path: Path) -> dict:
    """Load endpoint CSV.  Returns dict with arrays: time, cm, axis."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1, ndmin=2)
    if data.size == 0:
        return None
    t = data[:, 1]
    p0 = data[:, 3:6]
    p1 = data[:, 6:9]
    cm = 0.5 * (p0 + p1)
    axis = p1 - p0
    axis_len = np.linalg.norm(axis, axis=1, keepdims=True)
    axis_len[axis_len == 0] = 1.0
    axis = axis / axis_len
    return {"time": t, "cm": cm, "axis": axis}


def compute_msd(data: dict):
    """Compute total, parallel, and perpendicular MSD relative to t=0."""
    cm = data["cm"]
    axis0 = data["axis"][0]
    dr = cm - cm[0]
    dr_par = np.sum(dr * axis0, axis=1)
    dr_perp = dr - np.outer(dr_par, axis0)

    msd_total = np.sum(dr ** 2, axis=1)
    msd_par = dr_par ** 2
    msd_perp = np.sum(dr_perp ** 2, axis=1)
    return msd_total, msd_par, msd_perp


def compute_angular_msd(data: dict):
    """Angular MSD: <theta(t)^2> where theta = arccos(|n(t) . n(0)|)."""
    axes = data["axis"]
    n0 = axes[0]
    cos_theta = np.clip(np.abs(np.sum(axes * n0, axis=1)), 0, 1)
    theta = np.arccos(cos_theta)
    return theta ** 2


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_all(runs_root: Path, timestamp_prefix: str = None):
    """Walk run directories and collect MSD data.

    Returns: dict keyed by (N, AR, mu) -> list of per-realization dicts.
    """
    results = defaultdict(list)
    run_dirs = sorted(runs_root.iterdir())
    n_loaded = 0

    for d in run_dirs:
        if not d.is_dir():
            continue
        m = DIR_RE.match(d.name)
        if m is None:
            continue
        if timestamp_prefix and not d.name.startswith(timestamp_prefix):
            continue

        N = int(m.group(1))
        seed_id = m.group(2)
        AR = int(m.group(3))
        metric = m.group(4)
        rod = int(m.group(5))

        for csv_file in d.glob("free_rod_endpoints_mu*.csv"):
            mm = MU_FILE_RE.match(csv_file.name)
            if mm is None:
                continue
            mu = parse_mu_tag(mm.group(1))

            data = load_endpoints(csv_file)
            if data is None:
                continue

            msd_total, msd_par, msd_perp = compute_msd(data)
            angular_msd = compute_angular_msd(data)

            results[(N, AR, mu)].append({
                "time": data["time"],
                "msd": msd_total,
                "msd_par": msd_par,
                "msd_perp": msd_perp,
                "angular_msd": angular_msd,
                "seed": seed_id,
                "metric": metric,
            })
            n_loaded += 1
            if n_loaded % 500 == 0:
                print(f"  loaded {n_loaded} trajectories...")

    print(f"  loaded {n_loaded} trajectories total")
    return results


def average_msd(entries):
    """Average MSD arrays across entries, interpolating to the longest time grid."""
    if not entries:
        return None
    # Use the most common array length as the reference time grid
    lengths = [len(e["time"]) for e in entries]
    target_len = max(set(lengths), key=lengths.count)
    # Pick the first entry with that length as reference
    ref = next(e for e in entries if len(e["time"]) == target_len)
    t = ref["time"]

    # Only average entries with matching time grid length
    matching = [e for e in entries if len(e["time"]) == target_len]
    n = len(matching)
    msd = np.mean([e["msd"] for e in matching], axis=0)
    msd_par = np.mean([e["msd_par"] for e in matching], axis=0)
    msd_perp = np.mean([e["msd_perp"] for e in matching], axis=0)
    angular = np.mean([e["angular_msd"] for e in matching], axis=0)
    return {"time": t, "msd": msd, "msd_par": msd_par, "msd_perp": msd_perp,
            "angular_msd": angular, "n_samples": n}


def fit_diffusion(t, msd, fit_frac=(0.5, 0.9)):
    """Fit D from MSD = 6*D*t in the late-time window (3D)."""
    n = len(t)
    i0 = int(n * fit_frac[0])
    i1 = int(n * fit_frac[1])
    if i1 <= i0 + 2:
        return np.nan
    t_fit = t[i0:i1]
    msd_fit = msd[i0:i1]
    if t_fit[-1] - t_fit[0] < 1e-12:
        return np.nan
    slope, _ = np.polyfit(t_fit, msd_fit, 1)
    return slope / 6.0


def fit_diffusion_par(t, msd_par, fit_frac=(0.5, 0.9)):
    """D_par from MSD_par = 2*D_par*t (1D)."""
    n = len(t)
    i0 = int(n * fit_frac[0])
    i1 = int(n * fit_frac[1])
    if i1 <= i0 + 2:
        return np.nan
    slope, _ = np.polyfit(t[i0:i1], msd_par[i0:i1], 1)
    return slope / 2.0


def fit_diffusion_perp(t, msd_perp, fit_frac=(0.5, 0.9)):
    """D_perp from MSD_perp = 4*D_perp*t (2D)."""
    n = len(t)
    i0 = int(n * fit_frac[0])
    i1 = int(n * fit_frac[1])
    if i1 <= i0 + 2:
        return np.nan
    slope, _ = np.polyfit(t[i0:i1], msd_perp[i0:i1], 1)
    return slope / 4.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS_MU = {0.0: "C0", 0.1: "C1", 0.2: "C2", 0.4: "C3", 1.0: "C4"}


def plot_msd_by_N(avg, all_Ns, all_ARs, all_mus, pdf):
    """One figure per N: subplots = friction, lines = AR."""
    cmap = plt.cm.viridis
    ar_colors = {ar: cmap(i / max(1, len(all_ARs) - 1)) for i, ar in enumerate(sorted(all_ARs))}

    for N in sorted(all_Ns):
        fig, axes = plt.subplots(1, len(all_mus), figsize=(4 * len(all_mus), 4),
                                 sharex=True, sharey=True, squeeze=False)
        fig.suptitle(f"N = {N}  —  MSD of free rod center-of-mass", fontsize=13)
        for j, mu in enumerate(sorted(all_mus)):
            ax = axes[0, j]
            ax.set_title(f"$\\mu$ = {mu}")
            for AR in sorted(all_ARs):
                key = (N, AR, mu)
                if key not in avg:
                    continue
                d = avg[key]
                ax.loglog(d["time"][1:], d["msd"][1:],
                          color=ar_colors[AR], label=f"AR={AR}", linewidth=0.8)
            if j == 0:
                ax.set_ylabel("MSD")
            ax.set_xlabel("time")
            t_ref = np.geomspace(1e-2, 1e2, 50)
            ax.plot(t_ref, 1e-4 * t_ref, "--", color="gray", linewidth=0.5, label="$\\sim t$")
        axes[0, 0].legend(fontsize=5, ncol=2, loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot_msd_by_AR(avg, all_Ns, all_ARs, all_mus, pdf):
    """One figure per AR: subplots = N, lines = friction."""
    for AR in sorted(all_ARs):
        Ns_with_data = sorted([N for N in all_Ns if any((N, AR, mu) in avg for mu in all_mus)])
        if not Ns_with_data:
            continue
        ncols = min(len(Ns_with_data), 4)
        nrows = (len(Ns_with_data) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                                 sharex=True, sharey=True, squeeze=False)
        fig.suptitle(f"AR = {AR}  —  MSD by friction", fontsize=13)
        for idx, N in enumerate(Ns_with_data):
            ax = axes[idx // ncols, idx % ncols]
            ax.set_title(f"N = {N}", fontsize=10)
            for mu in sorted(all_mus):
                key = (N, AR, mu)
                if key not in avg:
                    continue
                d = avg[key]
                color = COLORS_MU.get(mu, "gray")
                ax.loglog(d["time"][1:], d["msd"][1:],
                          color=color, label=f"$\\mu$={mu}", linewidth=0.8)
            if idx % ncols == 0:
                ax.set_ylabel("MSD")
            ax.set_xlabel("time")
        for idx in range(len(Ns_with_data), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)
        axes[0, 0].legend(fontsize=6, loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot_msd_decomposed(avg, all_Ns, all_ARs, all_mus, pdf):
    """One figure per (N, mu): lines = AR, comparing MSD_par vs MSD_perp."""
    cmap = plt.cm.viridis
    ar_colors = {ar: cmap(i / max(1, len(all_ARs) - 1)) for i, ar in enumerate(sorted(all_ARs))}

    for N in sorted(all_Ns):
        for mu in sorted(all_mus):
            ARs_here = sorted([AR for AR in all_ARs if (N, AR, mu) in avg])
            if not ARs_here:
                continue
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
            fig.suptitle(f"N={N}, $\\mu$={mu}  —  MSD decomposition", fontsize=12)
            ax1.set_title("Parallel (along initial axis)")
            ax2.set_title("Perpendicular")
            for AR in ARs_here:
                d = avg[(N, AR, mu)]
                c = ar_colors[AR]
                ax1.loglog(d["time"][1:], d["msd_par"][1:], color=c,
                           label=f"AR={AR}", linewidth=0.8)
                ax2.loglog(d["time"][1:], d["msd_perp"][1:], color=c,
                           label=f"AR={AR}", linewidth=0.8)
            ax1.set_ylabel("MSD$_\\parallel$")
            ax2.set_ylabel("MSD$_\\perp$")
            for ax in (ax1, ax2):
                ax.set_xlabel("time")
            ax1.legend(fontsize=5, ncol=2, loc="upper left")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def plot_diffusion_vs_AR(avg, all_Ns, all_ARs, all_mus, pdf):
    """D, D_par, D_perp, D_par/D_perp vs AR for each N."""
    for N in sorted(all_Ns):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"N = {N}  —  Diffusion coefficients vs AR", fontsize=13)

        for mu in sorted(all_mus):
            ARs = sorted([AR for AR in all_ARs if (N, AR, mu) in avg])
            if not ARs:
                continue
            D_vals, Dpar_vals, Dperp_vals, ratio_vals = [], [], [], []
            for AR in ARs:
                d = avg[(N, AR, mu)]
                D = fit_diffusion(d["time"], d["msd"])
                Dp = fit_diffusion_par(d["time"], d["msd_par"])
                Dq = fit_diffusion_perp(d["time"], d["msd_perp"])
                D_vals.append(D)
                Dpar_vals.append(Dp)
                Dperp_vals.append(Dq)
                ratio_vals.append(Dp / Dq if Dq > 0 else np.nan)
            color = COLORS_MU.get(mu, "gray")
            axes[0, 0].plot(ARs, D_vals, "o-", color=color, label=f"$\\mu$={mu}",
                            markersize=3, linewidth=1)
            axes[0, 1].plot(ARs, Dpar_vals, "o-", color=color, markersize=3, linewidth=1)
            axes[1, 0].plot(ARs, Dperp_vals, "o-", color=color, markersize=3, linewidth=1)
            axes[1, 1].plot(ARs, ratio_vals, "o-", color=color, markersize=3, linewidth=1)

        axes[0, 0].set_title("D (total)")
        axes[0, 1].set_title("$D_\\parallel$")
        axes[1, 0].set_title("$D_\\perp$")
        axes[1, 1].set_title("$D_\\parallel / D_\\perp$")
        for ax in axes.flat:
            ax.set_xlabel("AR")
            ax.set_xscale("log")
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            ax.set_yscale("log")
        axes[1, 1].axhline(2, color="gray", linestyle="--", linewidth=0.5, label="free-rod (2)")
        axes[0, 0].legend(fontsize=6)
        axes[1, 1].legend(fontsize=6)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot_angular_msd(avg, all_Ns, all_ARs, all_mus, pdf):
    """Angular MSD vs time: one figure per N."""
    cmap = plt.cm.viridis
    ar_colors = {ar: cmap(i / max(1, len(all_ARs) - 1)) for i, ar in enumerate(sorted(all_ARs))}

    for N in sorted(all_Ns):
        mus_here = sorted([mu for mu in all_mus if any((N, AR, mu) in avg for AR in all_ARs)])
        if not mus_here:
            continue
        fig, axes = plt.subplots(1, len(mus_here), figsize=(4 * len(mus_here), 4),
                                 sharex=True, sharey=True, squeeze=False)
        fig.suptitle(f"N = {N}  —  Angular MSD", fontsize=13)
        for j, mu in enumerate(mus_here):
            ax = axes[0, j]
            ax.set_title(f"$\\mu$ = {mu}")
            for AR in sorted(all_ARs):
                key = (N, AR, mu)
                if key not in avg:
                    continue
                d = avg[key]
                ax.loglog(d["time"][1:], d["angular_msd"][1:],
                          color=ar_colors[AR], label=f"AR={AR}", linewidth=0.8)
            if j == 0:
                ax.set_ylabel("$\\langle \\theta^2 \\rangle$")
            ax.set_xlabel("time")
        axes[0, 0].legend(fontsize=5, ncol=2, loc="upper left")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def plot_D_vs_N(avg, all_Ns, all_ARs, all_mus, pdf):
    """D vs N for each AR — shows confinement effect."""
    for AR in sorted(all_ARs):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(f"AR = {AR}  —  D vs N")
        for mu in sorted(all_mus):
            Ns = sorted([N for N in all_Ns if (N, AR, mu) in avg])
            if not Ns:
                continue
            Ds = [fit_diffusion(avg[(N, AR, mu)]["time"], avg[(N, AR, mu)]["msd"]) for N in Ns]
            color = COLORS_MU.get(mu, "gray")
            ax.plot(Ns, Ds, "o-", color=color, label=f"$\\mu$={mu}", markersize=4, linewidth=1)
        ax.set_xlabel("N")
        ax.set_ylabel("D")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze free-rod MSD from endpoint runs.")
    ap.add_argument("--runs-root", type=Path,
                    default=Path(__file__).resolve().parent.parent / "runs" / "free_rod")
    ap.add_argument("--timestamp", type=str, default=None,
                    help="Only process directories starting with this timestamp prefix.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output PDF path.")
    ap.add_argument("--csv-out", type=Path, default=None,
                    help="Write summary CSV of diffusion coefficients.")
    args = ap.parse_args()

    output_pdf = args.output or (args.runs_root / "free_rod_msd_analysis.pdf")
    csv_out = args.csv_out or (args.runs_root / "free_rod_diffusion.csv")

    print(f"Scanning {args.runs_root} ...")
    results = collect_all(args.runs_root, args.timestamp)
    print(f"Collected data for {len(results)} (N, AR, mu) combinations")

    if not results:
        print("No data found.")
        sys.exit(1)

    avg = {}
    for key, entries in results.items():
        a = average_msd(entries)
        if a is not None:
            avg[key] = a

    all_Ns = sorted({k[0] for k in avg})
    all_ARs = sorted({k[1] for k in avg})
    all_mus = sorted({k[2] for k in avg})

    print(f"N values: {all_Ns}")
    print(f"AR values: {all_ARs}")
    print(f"Friction values: {all_mus}")
    print(f"Generating plots -> {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        plot_msd_by_N(avg, all_Ns, all_ARs, all_mus, pdf)
        plot_msd_by_AR(avg, all_Ns, all_ARs, all_mus, pdf)
        plot_msd_decomposed(avg, all_Ns, all_ARs, all_mus, pdf)
        plot_diffusion_vs_AR(avg, all_Ns, all_ARs, all_mus, pdf)
        plot_D_vs_N(avg, all_Ns, all_ARs, all_mus, pdf)
        plot_angular_msd(avg, all_Ns, all_ARs, all_mus, pdf)

    print(f"Saved {output_pdf}")

    with open(csv_out, "w") as f:
        f.write("N,AR,mu,D,D_par,D_perp,D_ratio,n_samples\n")
        for (N, AR, mu) in sorted(avg.keys()):
            d = avg[(N, AR, mu)]
            D = fit_diffusion(d["time"], d["msd"])
            Dp = fit_diffusion_par(d["time"], d["msd_par"])
            Dq = fit_diffusion_perp(d["time"], d["msd_perp"])
            ratio = Dp / Dq if Dq > 0 else float("nan")
            f.write(f"{N},{AR},{mu},{D:.6e},{Dp:.6e},{Dq:.6e},{ratio:.4f},{d['n_samples']}\n")
    print(f"Saved diffusion summary -> {csv_out}")


if __name__ == "__main__":
    main()
