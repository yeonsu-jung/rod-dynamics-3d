#!/usr/bin/env python3
"""
post_analyze_parametric_runs.py

Aggregate and analyze completed SLURM runs under /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs produced by
submit_parametric_runs.py. Summarizes kinetic energy decay per run and plots
aggregate trends across aspect ratios.

Features
- Scans runs/ for folders matching *RUN_rods*.
- Loads profile.csv from each run.
- Computes:
  * initial/final KE
  * exponential decay exponent (KE ~ a * exp(b * t))
  * power-law exponent (KE ~ t^{-p}) using last-fraction tail
- Saves a summary CSV and optional plots.

Dependencies
- pandas/numpy/matplotlib/scipy if available; otherwise uses built-in fallbacks.

Usage
  python3 post_analyze_parametric_runs.py \
      --runs-root /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs \
      --job-name my_job \
      --outdir ./analysis \
      --tail-frac 0.25 \
      --make-plots
"""
from __future__ import annotations
from pathlib import Path
import argparse, csv, math, os, re

# Optional imports
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None
try:
    import numpy as np  # type: ignore
except Exception:
    np = None
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None
try:
    import scipy.optimize as opt  # type: ignore
except Exception:
    opt = None

RUN_RE = re.compile(r"RUN_rods.*_AR(?P<alpha>[-+0-9.]+).*_N(?P<N>[-+0-9.]+).*_C(?P<C>[-+0-9.]+).*_L(?P<L>[-+0-9.]+)")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', type=Path, default=Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs"))
    ap.add_argument('--job-name', type=str, default='', help='Job name subdirectory under runs-root')
    ap.add_argument('--glob', default='*RUN_rods*', help='glob to select run folders')
    ap.add_argument('--outdir', type=Path, default=Path('analysis'))
    ap.add_argument('--summary-csv', type=Path, default=None, help='path to write summary CSV (defaults to <outdir>/summary.csv)')
    ap.add_argument('--tail-frac', type=float, default=0.25, help='fraction of tail samples for power-law fit')
    ap.add_argument('--make-plots', action='store_true')
    return ap.parse_args()


def discover_runs(runs_root: Path, patt: str):
    if not runs_root.exists():
        raise SystemExit(f"Runs root not found: {runs_root}")
    for d in sorted(runs_root.glob(patt)):
        if d.is_dir():
            yield d


def parse_run_name(name: str):
    m = RUN_RE.search(name)
    meta = {}
    if m:
        for k, v in m.groupdict().items():
            try:
                meta[k] = float(v) if ('.' in v) else int(v)
            except Exception:
                meta[k] = v
    return meta


def load_profile_csv(path: Path):
    if pd is not None:
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise RuntimeError(f"Failed reading {path}: {e}")
    # minimal fallback
    with open(path, newline='') as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows:
        raise RuntimeError("Empty CSV file")
    # convert to dict of lists
    cols = {}
    for k in rows[0].keys():
        cols[k] = []
    for r in rows:
        for k, v in r.items():
            cols[k].append(v)
    return cols


def series_to_np(df, col):
    if pd is not None:
        return df[col].to_numpy(dtype=float)
    else:
        return np.array([float(x) for x in df[col]], dtype=float) if np is not None else [float(x) for x in df[col]]


def fit_exponential(frames, ke):
    # Model: KE = a * exp(b * t)
    if np is None:
        return float('nan')
    t = np.array(frames, dtype=float)
    y = np.array(ke, dtype=float)
    # sanitize
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y = t[mask], y[mask]
    if t.size < 5:
        return float('nan')
    if opt is not None:
        try:
            def model(tt, aa, bb):
                return aa * np.exp(bb * tt)
            (aa, bb), _ = opt.curve_fit(model, t, y, p0=[y[0], -1e-3], maxfev=20000)
            return float(bb)
        except Exception:
            pass
    # fallback: linear fit on log(y)
    ly = np.log(y)
    A = np.vstack([t, np.ones_like(t)]).T
    # least squares
    try:
        slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
        return float(slope)
    except Exception:
        return float('nan')


def fit_powerlaw(frames, ke, tail_frac=0.25):
    # Model: KE ~ t^{-p} => log(KE) = log(c) - p log(t)
    if np is None:
        return float('nan')
    t = np.array(frames, dtype=float)
    y = np.array(ke, dtype=float)
    n = t.size
    if n < 10:
        return float('nan')
    start = max(1, int((1.0 - max(0.0, min(1.0, tail_frac))) * n))
    tt = t[start:]
    yy = y[start:]
    mask = (tt > 0) & np.isfinite(tt) & np.isfinite(yy) & (yy > 0)
    tt, yy = tt[mask], yy[mask]
    if tt.size < 5:
        return float('nan')
    lt = np.log(tt)
    ly = np.log(yy)
    A = np.vstack([lt, np.ones_like(lt)]).T
    try:
        slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
        p = -float(slope)
        return p
    except Exception:
        return float('nan')


def fit_exponential_params(frames, ke):
    """Return (a, b) for KE ~ a * exp(b * t)."""
    if np is None:
        return float('nan'), float('nan')
    t = np.array(frames, dtype=float)
    y = np.array(ke, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y = t[mask], y[mask]
    if t.size < 5:
        return float('nan'), float('nan')
    if opt is not None:
        try:
            def model(tt, aa, bb):
                return aa * np.exp(bb * tt)
            (aa, bb), _ = opt.curve_fit(model, t, y, p0=[y[0], -1e-3], maxfev=20000)
            return float(aa), float(bb)
        except Exception:
            pass
    # fallback: linear fit on log(y)
    try:
        ly = np.log(y)
        A = np.vstack([t, np.ones_like(t)]).T
        slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
        b = float(slope)
        a = float(np.exp(intercept))
        return a, b
    except Exception:
        return float('nan'), float('nan')


def fit_powerlaw_params(frames, ke, tail_frac=0.25):
    """Return (c, p) for KE ~ c * t^{-p}."""
    if np is None:
        return float('nan'), float('nan')
    t = np.array(frames, dtype=float)
    y = np.array(ke, dtype=float)
    n = t.size
    if n < 10:
        return float('nan'), float('nan')
    start = max(1, int((1.0 - max(0.0, min(1.0, tail_frac))) * n))
    tt = t[start:]
    yy = y[start:]
    mask = (tt > 0) & np.isfinite(tt) & np.isfinite(yy) & (yy > 0)
    tt, yy = tt[mask], yy[mask]
    if tt.size < 5:
        return float('nan'), float('nan')
    try:
        lt = np.log(tt)
        ly = np.log(yy)
        A = np.vstack([lt, np.ones_like(lt)]).T
        slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
        p = -float(slope)
        c = float(np.exp(intercept))
        return c, p
    except Exception:
        return float('nan'), float('nan')


def analyze_run(run_dir: Path, tail_frac: float):
    prof = run_dir / 'profile.csv'
    if not prof.exists():
        return None, None, None
    try:
        df = load_profile_csv(prof)
    except Exception as e:
        print(f"[post] Skipping run {run_dir.name}: invalid or empty profile.csv ({e})")
        return None, None, None
    frames = series_to_np(df, 'frame')
    ke = series_to_np(df, 'KE')
    meta = parse_run_name(run_dir.name)
    entry = {
        'run': run_dir.name,
        **meta,
        'initial_KE': float(ke[0]) if len(ke) else float('nan'),
        'final_KE': float(ke[-1]) if len(ke) else float('nan'),
        'exp_b': fit_exponential(frames, ke),
        'power_p': fit_powerlaw(frames, ke, tail_frac=tail_frac),
    }
    return entry, frames, ke


def save_summary(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    else:
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)


def make_plots(summary_rows, plot_data, outdir: Path):
    if plt is None or np is None:
        print('[post] matplotlib/numpy not available; skipping plots')
        return
    outdir.mkdir(parents=True, exist_ok=True)

    # Group by aspect ratio (alpha)
    by_ar = {}
    for r in summary_rows:
        ar = r.get('alpha')
        if ar is None: continue
        by_ar.setdefault(ar, []).append(r)

    # Exponential exponent vs AR
    ars = sorted([a for a in by_ar.keys() if isinstance(a, (int, float))])
    exp_b = [np.nanmean([x['exp_b'] for x in by_ar[a] if math.isfinite(x.get('exp_b', float('nan')))]) for a in ars]
    plt.figure()
    plt.plot(ars, exp_b, 'o-')
    plt.xlabel('Aspect Ratio (AR)')
    plt.ylabel('Exponential decay exponent b')
    plt.title('Exp decay exponent vs AR')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'exp_exponent_vs_AR.png')

    # Power-law exponent vs AR
    power_p = [np.nanmean([x['power_p'] for x in by_ar[a] if math.isfinite(x.get('power_p', float('nan')))]) for a in ars]
    plt.figure()
    plt.plot(ars, power_p, 's-')
    plt.xlabel('Aspect Ratio (AR)')
    plt.ylabel('Power-law exponent p')
    plt.title('Power-law exponent vs AR')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'power_exponent_vs_AR.png')

    # Overlaid KE vs t plot (per-run faint lines, average per AR, fitted curves)
    plt.figure(figsize=(10, 6))
    for ar in ars:
        pd_list = [pd for pd in plot_data if pd['alpha'] == ar and pd['frames'] is not None and pd['ke'] is not None]
        if not pd_list:
            continue
        # Determine common prefix length to safely average without interpolation
        lengths = [len(pd['frames']) for pd in pd_list]
        min_len = min(lengths)
        if min_len < 5:
            # too short to be meaningful
            continue
        # Use the common prefix of frames from the first entry
        frames_ref = np.array(pd_list[0]['frames'][:min_len], dtype=float)
        # Truncate each run to min_len and stack
        ke_stack = np.vstack([np.array(pd['ke'][:min_len], dtype=float) for pd in pd_list])
        # plot individual runs faintly
        for row in ke_stack:
            plt.plot(frames_ref, row, color='gray', alpha=0.25)
        # average curve
        avg_ke = np.mean(ke_stack, axis=0)
        plt.plot(frames_ref, avg_ke, label=f'AR={ar}', linewidth=2)
        # fitted exponential on averaged curve
        a, b = fit_exponential_params(frames_ref, avg_ke)
        if np.isfinite(a) and np.isfinite(b):
            try:
                plt.plot(frames_ref, a * np.exp(b * frames_ref), '--', color='C0', alpha=0.9)
            except Exception:
                pass
        # fitted power-law on averaged curve (skip t<=0)
        c, p = fit_powerlaw_params(frames_ref, avg_ke, tail_frac=0.25)
        if np.isfinite(c) and np.isfinite(p):
            try:
                t_plot = frames_ref.copy()
                # avoid zero in power law evaluation
                t_plot = np.where(t_plot <= 0, np.nan, t_plot)
                plt.plot(t_plot, c * (t_plot ** (-p)), ':', color='C1', alpha=0.9)
            except Exception:
                pass
    plt.ylim([1e-5, 1.2 * np.max([pd['ke'][0] for pd in plot_data if pd['ke'] is not None])])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frame (t)')
    plt.ylabel('Kinetic Energy (KE)')
    plt.title('Overlaid KE vs t (individual runs faint, average per AR + fits)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / 'overlaid_ke_vs_t.png')


def main():
    args = parse_args()
    runs_root = args.runs_root
    if args.job_name:
        runs_root = runs_root / args.job_name
    runs = list(discover_runs(runs_root, args.glob))
    if not runs:
        raise SystemExit(f"No run folders found in {args.runs_root} with glob '{args.glob}'")

    rows = []
    plot_data = []
    for rd in runs:
        res, frames, ke = analyze_run(rd, tail_frac=args.tail_frac)
        if res is None:
            print(f"[post] Missing profile.csv in: {rd}")
            continue
        rows.append(res)
        plot_data.append({'frames': frames, 'ke': ke, 'alpha': res.get('alpha')})

    outdir = args.outdir
    out_csv = args.summary_csv or (outdir / 'summary.csv')
    save_summary(rows, out_csv)
    print(f"[post] Wrote summary: {out_csv}")

    if args.make_plots:
        make_plots(rows, plot_data, outdir)
        print(f"[post] Wrote plots to: {outdir}")

if __name__ == '__main__':
    main()
