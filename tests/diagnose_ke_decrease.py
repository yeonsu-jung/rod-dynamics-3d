#!/usr/bin/env python3
"""
diagnose_ke_decrease.py

Usage:
  python diagnose_ke_decrease.py --csv profile_headless.csv [--perrod perrod.csv] [--abs 0.1] [--rel 0.01] [--top 8] [--plot]

What it does:
- Finds frames where total KE decreases (absolute or relative threshold).
- Prints top N decreases with jn_sum/jt_sum/impulse_count from the per-frame CSV.
- If a per-rod CSV is provided (sampled frames), shows top rods that lost KE between sampled frames surrounding the event.
- Optionally plots KE + impulses (requires matplotlib).

Notes:
- If per-rod sampling missed the exact frames, per-rod info may be approximate.
- If you want exact stage-level diagnostics (KE after integrate / after solve / after positional correction), I can add logging to the simulator to emit those checkpoints.
"""
import argparse, sys, csv, math

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except Exception:
    pd = None
    np = None
    plt = None

def load_profile(path):
    if pd:
        return pd.read_csv(path)
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

def load_perrod(path):
    if pd:
        return pd.read_csv(path)
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='per-frame CSV (must contain frame, KE)')
    p.add_argument('--perrod', help='per-rod sampled CSV (optional)')
    p.add_argument('--abs', type=float, default=0.0, help='absolute KE decrease threshold')
    p.add_argument('--rel', type=float, default=0.01, help='relative KE decrease threshold (fraction)')
    p.add_argument('--top', type=int, default=8, help='top N decreases to report')
    p.add_argument('--plot', action='store_true', help='plot KE & impulses (matplotlib)')
    args = p.parse_args()

    df = load_profile(args.csv)
    if pd:
        if 'frame' not in df.columns or 'KE' not in df.columns:
            print("CSV missing 'frame' or 'KE' columns", file=sys.stderr); sys.exit(2)
        frames = df['frame'].astype(int).to_numpy()
        KE = df['KE'].astype(float).to_numpy()
        jn = df['jn_sum'].astype(float).to_numpy() if 'jn_sum' in df.columns else None
        jt = df['jt_sum'].astype(float).to_numpy() if 'jt_sum' in df.columns else None
        ic = df['impulse_count'].astype(int).to_numpy() if 'impulse_count' in df.columns else None
    else:
        rows = df
        frames = [int(float(r['frame'])) for r in rows]
        KE = [float(r['KE']) for r in rows]
        jn = [float(r.get('jn_sum',0.0)) for r in rows]
        jt = [float(r.get('jt_sum',0.0)) for r in rows]
        ic = [int(float(r.get('impulse_count',0))) for r in rows]

    # diffs: KE_next - KE_prev -> negative means decrease
    deltas = []
    for i in range(1, len(KE)):
        d = KE[i] - KE[i-1]
        rel = abs(d) / (abs(KE[i-1]) + 1e-12)
        deltas.append((i, frames[i], d, rel, jn[i] if jn is not None else None, jt[i] if jt is not None else None, ic[i] if ic is not None else None))

    # Filter decreases
    decs = [t for t in deltas if t[2] < 0 and (abs(t[2]) >= args.abs or t[3] >= args.rel)]
    if not decs:
        print("No significant KE decreases found (abs=%.6g rel=%.3f)"%(args.abs, args.rel))
    else:
        decs.sort(key=lambda x: x[2])  # most negative first
        print("Top %d KE decreases:"%min(args.top, len(decs)))
        for idx, frame, dke, rel, jn_v, jt_v, ic_v in decs[:args.top]:
            print(" frame=%d  dKE=%.6g  rel=%.3g  KE_after=%.6g  jn_sum=%s  jt_sum=%s  impulses=%s" % (
                frame, dke, rel, KE[idx], ('%.6g'%jn_v) if jn_v is not None else 'N/A',
                ('%.6g'%jt_v) if jt_v is not None else 'N/A', str(ic_v) if ic_v is not None else 'N/A'
            ))

    # If per-rod present, show top rods losing KE nearby sampled frames
    if args.perrod:
        pr = load_perrod(args.perrod)
        if pd:
            grouped = pr.groupby('frame')
        else:
            byframe = {}
            for r in pr:
                byframe.setdefault(int(float(r['frame'])), []).append(r)

        def report_near(frame_idx, frame_number):
            # find nearest sampled frame <= frame_number (per-rod samples may be sparse)
            if pd:
                available = sorted(grouped.groups.keys())
            else:
                available = sorted(byframe.keys())
            import bisect
            i = bisect.bisect_right(available, frame_number)
            if i==0:
                return None
            sample_frame = available[i-1]
            # compute per-rod KE list
            if pd:
                g = grouped.get_group(sample_frame)
                if 'KE_total' in g.columns:
                    g2 = g.sort_values('KE_total', ascending=False)
                    rows = g2.head(10)
                    print(f"\nSampled frame {sample_frame} (closest <= {frame_number}): top rods by KE_total")
                    for _,row in rows.iterrows():
                        print("  rod=%d KE_total=%.6g pos=(%.3g,%.3g,%.3g) v=(%.3g,%.3g,%.3g)"%
                              (int(row['rod']), row.get('KE_total',0.0),
                               row.get('px',0), row.get('py',0), row.get('pz',0),
                               row.get('vx',0), row.get('vy',0), row.get('vz',0)))
                else:
                    print(f"per-rod sample {sample_frame} missing KE_total")
            else:
                rows = byframe[sample_frame]
                def ke_row(r):
                    try:
                        return float(r.get('KE_total') or (float(r.get('KE_lin',0))+float(r.get('KE_rot',0))))
                    except:
                        return 0.0
                rows.sort(key=ke_row, reverse=True)
                print(f"\nSampled frame {sample_frame} (closest <= {frame_number}): top rods by KE_total")
                for r in rows[:10]:
                    print("  rod=%s KE_total=%s pos=(%s,%s,%s) v=(%s,%s,%s)"%
                          (r.get('rod'), r.get('KE_total') or (r.get('KE_lin')+'+'+r.get('KE_rot','0')),
                           r.get('px'), r.get('py'), r.get('pz'), r.get('vx'), r.get('vy'), r.get('vz')))
            return sample_frame

        # show per-rod for top decreases
        if decs:
            print("\nPer-rod sampled summaries (closest sampled frame <= event frame):")
            for t in decs[:min(len(decs), args.top)]:
                _, frame_num, dke, _, _, _, _ = t
                report_near(_, frame_num)

    # optional plot
    if args.plot and plt and pd:
        x = frames
        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.plot(x, KE, label='KE')
        plt.ylabel('KE')
        for _,f,_,_,_,_,_ in decs[:args.top]:
            plt.axvline(f, color='r', alpha=0.3)
        plt.subplot(2,1,2)
        if jn is not None: plt.plot(x, jn, label='jn_sum')
        if jt is not None: plt.plot(x, jt, label='jt_sum')
        plt.legend()
        plt.xlabel('frame')
        plt.show()

if __name__ == '__main__':
    main()