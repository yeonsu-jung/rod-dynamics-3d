#!/usr/bin/env python3
"""
diagnose_ke.py

Usage:
  python diagnose_ke.py --csv profile_headless.csv [--perrod perrod.csv] [--abs-thresh 0.5] [--rel-thresh 0.05] [--top-rods 5] [--plot]

- Reports frames where total KE changes sharply and prints per-frame impulse sums (jn_sum/jt_sum)
- If per-rod CSV is provided, shows top rods contributing KE change at sampled frames
- Optional plotting requires matplotlib
"""
import argparse
import math
import sys
import csv

def try_import(name):
    try:
        mod = __import__(name)
        return mod
    except Exception:
        return None

pd = try_import('pandas')
np = try_import('numpy')
plt = try_import('matplotlib.pyplot')

def load_profile(path):
    if pd:
        df = pd.read_csv(path)
        return df
    # fallback simple csv parse
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # convert to dict-of-lists
    cols = {}
    for k in rows[0].keys():
        cols[k] = []
    for r in rows:
        for k,v in r.items():
            cols[k].append(v)
    return cols

def col_get(df, name, default=None):
    if pd:
        return df[name] if name in df.columns else None
    return df.get(name, None)

def as_float_list(series):
    if series is None: return None
    if pd:
        return series.astype(float).to_numpy()
    else:
        out=[]
        for v in series:
            try:
                out.append(float(v))
            except:
                out.append(float('nan'))
        return out

def load_perrod(path):
    if pd:
        pr = pd.read_csv(path)
        return pr
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='Profile CSV (per-frame) with KE column')
    ap.add_argument('--perrod', help='Per-rod CSV (sampled frames)')
    ap.add_argument('--abs-thresh', type=float, default=0.0,
                    help='Absolute KE increase threshold to report (J). Default 0 => use rel threshold only')
    ap.add_argument('--rel-thresh', type=float, default=0.05,
                    help='Relative KE increase threshold (fraction of prev KE). Default 0.05 (5%%)')
    ap.add_argument('--top-rods', type=int, default=5, help='Top rods to print from per-rod data')
    ap.add_argument('--plot', action='store_true', help='Show simple plots (matplotlib required)')
    args = ap.parse_args()

    print(f"[diag] Loading profile CSV: {args.csv}")
    try:
        df = load_profile(args.csv)
    except Exception as e:
        print("ERROR loading profile CSV:", e)
        sys.exit(2)

    # Extract frame and KE
    if pd:
        if 'frame' not in df.columns or 'KE' not in df.columns:
            print("ERROR: CSV must contain 'frame' and 'KE' columns")
            sys.exit(2)
        frames = df['frame'].astype(int).to_numpy()
        ke = df['KE'].astype(float).to_numpy()
        jn = df['jn_sum'].astype(float).to_numpy() if 'jn_sum' in df.columns else None
        jt = df['jt_sum'].astype(float).to_numpy() if 'jt_sum' in df.columns else None
        impc = df['impulse_count'].astype(int).to_numpy() if 'impulse_count' in df.columns else None
    else:
        frames = [int(x) for x in df['frame']]
        ke = [float(x) for x in df['KE']]
        jn = [float(x) for x in df.get('jn_sum', [0]*len(frames))]
        jt = [float(x) for x in df.get('jt_sum', [0]*len(frames))]
        impc = [int(float(x)) for x in df.get('impulse_count', [0]*len(frames))]

    # compute frame-to-frame KE diff
    ke_prev = ke[:-1]
    ke_next = ke[1:]
    dke = [kn - kp for kp,kn in zip(ke_prev, ke_next)]
    # Use absolute and relative thresholds
    abs_thresh = args.abs_thresh
    rel_thresh = args.rel_thresh

    candidates = []
    for i,delta in enumerate(dke):
        prev = ke_prev[i]
        rel = (abs(delta) / (abs(prev) + 1e-12))
        if (abs_thresh > 0 and delta > abs_thresh) or (rel > rel_thresh and delta > 1e-12):
            # record event at frame index i+1 (the frame where change observed)
            idx = i+1
            jn_v = jn[idx] if jn is not None else None
            jt_v = jt[idx] if jt is not None else None
            ic_v = impc[idx] if impc is not None else None
            candidates.append({
                'frame': int(frames[idx]),
                'dKE': float(delta),
                'KE': float(ke_next[i]),
                'jn_sum': float(jn_v) if jn_v is not None else None,
                'jt_sum': float(jt_v) if jt_v is not None else None,
                'impulse_count': int(ic_v) if ic_v is not None else None,
                'row_index': idx
            })

    if not candidates:
        print("[diag] No KE increase events found with thresholds abs=%.6g rel=%.3f" % (abs_thresh, rel_thresh))
    else:
        print("[diag] Found %d KE-increase events:" % len(candidates))
        for ev in candidates:
            print("  frame=%d  KE=%.6g  dKE=%.6g  jn_sum=%s  jt_sum=%s  impulses=%s" % (
                ev['frame'], ev['KE'], ev['dKE'],
                ('%.6g'%ev['jn_sum']) if ev['jn_sum'] is not None else 'N/A',
                ('%.6g'%ev['jt_sum']) if ev['jt_sum'] is not None else 'N/A',
                str(ev['impulse_count']) if ev['impulse_count'] is not None else 'N/A'
            ))

    # If per-rod CSV provided, inspect top rods at candidate frames
    if args.perrod:
        print("\n[diag] Loading per-rod CSV:", args.perrod)
        try:
            pr = load_perrod(args.perrod)
        except Exception as e:
            print("ERROR loading per-rod CSV:", e)
            pr = None
        if pr is not None:
            if pd:
                # group by frame for quick lookup
                grouped = pr.groupby('frame')
                for ev in candidates:
                    f = ev['frame']
                    if f not in grouped.groups:
                        print(f"[diag] per-rod sample for frame {f} not present (per-rod is sampled).")
                        continue
                    g = grouped.get_group(f)
                    # compute per-rod KE_total if present (or use KE_lin+KE_rot)
                    if 'KE_total' in g.columns:
                        g2 = g.sort_values('KE_total', ascending=False)
                        print(f"\nTop {args.top_rods} rods at sampled frame {f}:")
                        for i,row in g2.head(args.top_rods).iterrows():
                            print("  rod=%d  KE_total=%.6g  KE_lin=%.6g  KE_rot=%.6g  pos=(%.3g,%.3g,%.3g)  v=(%.3g,%.3g,%.3g)" % (
                                int(row['rod']), row.get('KE_total',0.0), row.get('KE_lin',0.0), row.get('KE_rot',0.0),
                                row.get('px',0.0), row.get('py',0.0), row.get('pz',0.0),
                                row.get('vx',0.0), row.get('vy',0.0), row.get('vz',0.0)
                            ))
                    else:
                        print(f"[diag] per-rod CSV missing KE_total column for frame {f}")
            else:
                # fallback plain csv rows list
                rows = pr
                byframe = {}
                for r in rows:
                    f = int(float(r['frame']))
                    byframe.setdefault(f, []).append(r)
                for ev in candidates:
                    f = ev['frame']
                    if f not in byframe:
                        print(f"[diag] per-rod sample for frame {f} not present.")
                        continue
                    rowsf = byframe[f]
                    # compute KE_total per row
                    def row_ke(r):
                        try:
                            return float(r.get('KE_total') or (float(r.get('KE_lin',0))+float(r.get('KE_rot',0))))
                        except:
                            return 0.0
                    rowsf.sort(key=row_ke, reverse=True)
                    print(f"\nTop {args.top_rods} rods at sampled frame {f}:")
                    for r in rowsf[:args.top_rods]:
                        print("  rod=%s  KE_total=%s  pos=(%s,%s,%s)  v=(%s,%s,%s)" % (
                            r.get('rod'), r.get('KE_total') or (r.get('KE_lin')+'+'+r.get('KE_rot', '0')),
                            r.get('px'), r.get('py'), r.get('pz'),
                            r.get('vx'), r.get('vy'), r.get('vz')
                        ))

    # optional plotting
    if args.plot and plt:
        print("[diag] plotting KE and impulse sums (requires display)")
        if pd:
            x = frames
            plt.figure(figsize=(10,6))
            plt.subplot(2,1,1)
            plt.plot(x, ke, label='KE')
            plt.ylabel('KE')
            for ev in candidates:
                plt.axvline(ev['frame'], color='r', alpha=0.3)
            plt.subplot(2,1,2)
            if jn is not None:
                plt.plot(x, jn, label='jn_sum')
            if jt is not None:
                plt.plot(x, jt, label='jt_sum')
            plt.legend()
            plt.xlabel('frame')
            plt.show()
        else:
            print("Plotting requires pandas/numpy integration; skipping.")

if __name__ == '__main__':
    main()