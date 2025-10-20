#!/usr/bin/env python3
"""
post_analyze_friction_effects.py

Analyze the effect of friction coefficient on KE decay for a fixed aspect ratio (default: alpha=100).

- Scans runs under a given --runs-root and --job-name subdirectory
- Filters runs where parsed alpha in the run name equals --alpha
- Loads each run's profile.csv robustly (skips empty/invalid/partial files)
- Attempts to read friction coefficient from the run's scene.json using heuristics or an optional --json-key path
- Computes KE metrics per run: initial/final KE, exponential decay exponent, power-law exponent
- Writes a summary CSV and plots of exponents vs. friction, and KE overlays grouped by friction

Usage:
  python3 post_analyze_friction_effects.py \
    --job-name TEST \
    --runs-root /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs \
    --alpha 100 \
    --outdir analysis_friction_TEST \
    --make-plots

Notes:
- Friction extraction looks for keys like 'friction', 'mu', 'frictionCoefficient' in scene.json. You can override with --json-key "scene/physics/solver/friction".
- Plots and summary tolerate in-progress runs with growing CSVs.
"""
from __future__ import annotations
from pathlib import Path
import argparse, csv, json, math, re

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
    ap = argparse.ArgumentParser(description="Analyze friction effect on KE decay for a fixed aspect ratio")
    ap.add_argument('--runs-root', type=Path, default=Path("/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs"))
    ap.add_argument('--job-name', type=str, default='', help='Job name subdirectory under runs-root')
    ap.add_argument('--glob', default='*RUN_rods*', help='glob to select run folders')
    ap.add_argument('--alpha', type=float, default=100.0, help='Aspect ratio to filter (e.g., 100)')
    ap.add_argument('--outdir', type=Path, default=Path('analysis_friction'))
    ap.add_argument('--summary-csv', type=Path, default=None, help='path to write summary CSV (defaults to <outdir>/summary_alpha<alpha>.csv)')
    ap.add_argument('--tail-frac', type=float, default=0.25, help='fraction of tail samples for power-law fit')
    ap.add_argument('--json-key', type=str, default='', help='Optional slash-delimited key path into scene.json to read friction (e.g., scene/physics/solver/friction)')
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
    cols = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k, v in r.items():
            cols[k].append(v)
    return cols


def series_to_np(df, col):
    if pd is not None:
        return df[col].to_numpy(dtype=float)
    else:
        return np.array([float(x) for x in df[col]], dtype=float) if np is not None else [float(x) for x in df[col]]


def maybe_series_to_np(df, col):
    if pd is not None:
        if hasattr(df, 'columns') and col in df.columns:
            try:
                return df[col].to_numpy(dtype=float)
            except Exception:
                return None
        return None
    else:
        if isinstance(df, dict) and col in df:
            try:
                return np.array([float(x) for x in df[col]], dtype=float) if np is not None else [float(x) for x in df[col]]
            except Exception:
                return None
        return None


def fit_exponential(frames, ke):
    if np is None:
        return float('nan')
    t = np.array(frames, dtype=float)
    y = np.array(ke, dtype=float)
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
    try:
        ly = np.log(y)
        A = np.vstack([t, np.ones_like(t)]).T
        slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
        return float(slope)
    except Exception:
        return float('nan')


def fit_powerlaw(frames, ke, tail_frac=0.25):
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
    try:
        lt = np.log(tt)
        ly = np.log(yy)
        A = np.vstack([lt, np.ones_like(lt)]).T
        slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
        p = -float(slope)
        return p
    except Exception:
        return float('nan')


# --- friction extraction from scene.json ---

def get_by_path(obj, path: str):
    cur = obj
    for p in path.split('/'):
        if p == '':
            continue
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def find_friction_heuristic(obj):
    """Depth-first search for the 'mu' key.
    Returns the numeric value if found, else nan.
    """
    if isinstance(obj, dict):
        if 'mu' in obj and isinstance(obj['mu'], (int, float)):
            return float(obj['mu'])
        for v in obj.values():
            val = find_friction_heuristic(v)
            if isinstance(val, float):
                return val
    if isinstance(obj, list):
        for it in obj:
            val = find_friction_heuristic(it)
            if isinstance(val, float):
                return val
    return float('nan')


# def extract_mu(run_dir: Path, json_key_path: str = '') -> float:
#     scene = run_dir / 'scene.json'
#     if not scene.exists():
#         return float('nan')
#     try:
#         with open(scene, 'r') as f:
#             data = json.load(f)
#     except Exception:
#         return float('nan')
#     if json_key_path:
#         val = get_by_path(data, json_key_path)
#         if isinstance(val, (int, float)):
#             return float(val)
#         else:
#             return float('nan')
#     return find_friction_heuristic(data)


def extract_mu(run_dir: Path, json_key_path: str = '') -> float:
    """Extracts the 'friction' value from the JSON file."""
    json_path = run_dir / 'scene.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    try:
        friction_value = data["scene"]["bodies"][0]["friction"]
        return friction_value
    except (KeyError, IndexError) as e:
        raise ValueError(f"Could not find 'friction' key in the JSON: {e}")


# --- per-run analysis ---

def analyze_run(run_dir: Path, tail_frac: float, json_key_path: str):
    prof = run_dir / 'profile.csv'
    if not prof.exists():
        return None
    try:
        df = load_profile_csv(prof)
    except Exception as e:
        print(f"[fric] Skipping {run_dir.name}: invalid or empty profile.csv ({e})")
        return None
    try:
        frames = series_to_np(df, 'frame')
        ke = series_to_np(df, 'KE')
    except Exception as e:
        print(f"[fric] Skipping {run_dir.name}: missing columns ({e})")
        return None
    if len(frames) < 5 or len(ke) < 5:
        return None
    meta = parse_run_name(run_dir.name)
    mu = extract_mu(run_dir, json_key_path)
    
    
    return {
        'run': run_dir.name,
        'alpha': meta.get('alpha', None),
        'mu': mu,
        'initial_KE': float(ke[0]),
        'final_KE': float(ke[-1]),
        'exp_b': fit_exponential(frames, ke),
        'power_p': fit_powerlaw(frames, ke, tail_frac=tail_frac),
        'frames': frames,
        'ke': ke,
    }


def save_summary(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if pd is not None:
        pd.DataFrame([{k: v for k, v in r.items() if k not in ('frames','ke')} for r in rows]).to_csv(out_csv, index=False)
    else:
        if not rows:
            return
        keys = [k for k in rows[0].keys() if k not in ('frames','ke')]
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})


def make_plots(rows, outdir: Path, alpha: float):
    if plt is None or np is None:
        print('[fric] matplotlib/numpy not available; skipping plots')
        return
    outdir.mkdir(parents=True, exist_ok=True)

    # Filter rows with valid mu
    rows_mu = [r for r in rows if isinstance(r.get('mu'), (int,float)) and math.isfinite(r['mu'])]
    if not rows_mu:
        print('[fric] No runs with identifiable friction coefficient; skipping plots')
        return

    # exp_b vs mu
    mus = np.array([r['mu'] for r in rows_mu], dtype=float)
    exp_b = np.array([r['exp_b'] for r in rows_mu], dtype=float)
    plt.figure()
    plt.scatter(mus, exp_b, c='tab:blue')
    plt.xlabel('Friction coefficient (mu)')
    plt.ylabel('Exponential decay exponent b')
    plt.title(f'KE exp decay vs friction (alpha={alpha:g})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f'exp_b_vs_mu_alpha{int(alpha)}.png')

    # power_p vs mu
    power_p = np.array([r['power_p'] for r in rows_mu], dtype=float)
    plt.figure()
    plt.scatter(mus, power_p, c='tab:orange')
    plt.xlabel('Friction coefficient (mu)')
    plt.ylabel('Power-law exponent p')
    plt.title(f'KE power-law vs friction (alpha={alpha:g})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f'power_p_vs_mu_alpha{int(alpha)}.png')
    # KE overlays grouped by mu (average over runs with same mu, common prefix)
    # Build groups
    groups = {}
    for r in rows_mu:
        groups.setdefault(float(r['mu']), []).append(r)
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('viridis')
    mus_sorted = sorted(groups.keys())

    # Map each distinct mu to a distinct color; if only one mu, pick middle of cmap
    mu_colors = {}
    mcount = len(mus_sorted)
    if mcount == 0:
        return
    # use a diverging colormap, centered on the median mu so low/high mu get opposing colors
    cmap_div = plt.get_cmap('RdYlBu')
    mn = mus_sorted[0]
    mx = mus_sorted[-1]
    try:
        center = float(np.median(mus_sorted))
    except Exception:
        center = 0.5 * (mn + mx)
    span = max(center - mn, mx - center)
    if span == 0:
        for mu in mus_sorted:
            mu_colors[mu] = cmap_div(0.5)
    else:
        for mu in mus_sorted:
            norm = 0.5 + (mu - center) / (2.0 * span)  # center -> 0.5, extremes -> ~0 or 1
            norm = max(0.0, min(1.0, norm))
            mu_colors[mu] = cmap_div(norm)

    for k_idx, mu in enumerate(mus_sorted):
        grp = groups[mu]
        lens = [len(x['frames']) for x in grp]
        L = min(lens)
        if L < 5:
            continue
        t = np.array(grp[0]['frames'][:L], dtype=float)
        stack = np.vstack([np.array(x['ke'][:L], dtype=float) for x in grp])
        color = mu_colors[mu]
        # plot individual runs with the group's color (lighter) only if multiple runs
        if len(grp) > 1:
            for row in stack:
                # plt.plot(t, row, color=color, alpha=0.25)
                plt.plot(t, row)
        # plot the group average with the same color (stronger)
        avg = np.mean(stack, axis=0)
        # plt.plot(t, avg, color=color, linewidth=2, label=f'mu={mu:g}')
        plt.plot(t, avg, linewidth=2, label=f'mu={mu:g}')

    plt.xlabel('Frame (t)')
    plt.ylabel('Kinetic Energy (KE)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'KE vs t by friction (alpha={alpha:g})')
    plt.legend(title='mu', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f'ke_overlay_by_mu_loglog_alpha{int(alpha)}.png')

    plt.xscale('linear')
    plt.yscale('linear')
    plt.savefig(outdir / f'ke_overlay_by_mu_alpha{int(alpha)}.png')

    # semilogy
    plt.xscale('linear')
    plt.yscale('log')
    plt.savefig(outdir / f'ke_overlay_by_mu_semilogy_alpha{int(alpha)}.png')

    # semilogx
    plt.xscale('log')
    plt.yscale('linear')
    plt.savefig(outdir / f'ke_overlay_by_mu_semilogx_alpha{int(alpha)}.png')


def main():
    args = parse_args()
    runs_root = args.runs_root
    if args.job_name:
        runs_root = runs_root / args.job_name
    runs = list(discover_runs(runs_root, args.glob))
    if not runs:
        raise SystemExit(f"No run folders found in {args.runs_root} with glob '{args.glob}'")

    target_alpha = args.alpha
    rows = []
    for rd in runs:
        meta = parse_run_name(rd.name)
        a = meta.get('alpha', None)
        if a is None:
            continue
        try:
            a_val = float(a)
        except Exception:
            continue
        if abs(a_val - target_alpha) > 1e-6:
            continue
        res = analyze_run(rd, tail_frac=args.tail_frac, json_key_path=args.json_key)
        if res is None:
            continue
        rows.append(res)

    if not rows:
        raise SystemExit(f"No valid runs found for alpha={target_alpha}")

    outdir = args.outdir
    out_csv = args.summary_csv or (outdir / f'summary_alpha{int(target_alpha)}.csv')
    save_summary(rows, out_csv)
    print(f"[fric] Wrote summary: {out_csv}")

    if args.make_plots:
        make_plots(rows, outdir, target_alpha)
        print(f"[fric] Wrote plots to: {outdir}")


if __name__ == '__main__':
    main()
