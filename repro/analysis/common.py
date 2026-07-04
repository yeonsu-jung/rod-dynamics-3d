"""Shared helpers for the reproduction analysis scripts (numpy-only)."""
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
RUNS = REPO / "runs"
FIGS = REPO / "figures"
MANIFEST = REPO / "repro" / "manifest.csv"
PACKINGS = REPO / "assets" / "packings_metadata.csv"

# Paper timescales (SI S3.6): t_u = d*/v0 with d* = 0.32 l, v0 = 0.1.
V0 = 0.1
D_STAR = 0.32
T_U = D_STAR / V0  # = 3.2 time units


def load_manifest(groups=None):
    rows = list(csv.DictReader(MANIFEST.open()))
    if groups:
        rows = [r for r in rows if r["group"] in groups]
    return rows


def load_packings():
    return {p["packing_id"]: p for p in csv.DictReader(PACKINGS.open())}


def run_dir(run_id):
    return RUNS / run_id


def is_done(run_id):
    import json
    mp = run_dir(run_id) / "meta.json"
    if not mp.exists():
        return False
    try:
        return json.loads(mp.read_text()).get("status") == "done"
    except Exception:
        return False


def load_profile(run_id, cols):
    """Read selected columns of runs/<id>/profile.csv into a dict of
    float arrays keyed by column name (plus 'frame')."""
    want = ["frame"] + [c for c in cols if c != "frame"]
    out = {c: [] for c in want}
    with (run_dir(run_id) / "profile.csv").open() as fh:
        for row in csv.DictReader(fh):
            for c in want:
                out[c].append(float(row[c]))
    return {c: np.asarray(v) for c, v in out.items()}


def ent_series(prof, ent_period, dt=1e-3):
    """(t, ebar) at the frames where entanglement was actually evaluated."""
    fr = prof["frame"].astype(int)
    mask = fr % int(ent_period) == 0
    fr, s, p = fr[mask], prof["ent_sum"][mask], prof["ent_pairs"][mask]
    # dedupe repeated frames (frame 0 can appear once only; keep first)
    _, idx = np.unique(fr, return_index=True)
    fr, s, p = fr[idx], s[idx], p[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        ebar = np.where(p > 0, s / p, np.nan)
    return fr * dt, ebar


def halving_time(t, series):
    """First time the series drops below half its initial value (linear
    interpolation between samples); nan if it never does."""
    s0 = series[0]
    below = np.where(series <= 0.5 * s0)[0]
    if not len(below) or s0 <= 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return t[0]
    f = (0.5 * s0 - series[i - 1]) / (series[i] - series[i - 1])
    return float(t[i - 1] + f * (t[i] - t[i - 1]))


def rolling_mean(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def paper_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.bbox": "tight",
        "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt
