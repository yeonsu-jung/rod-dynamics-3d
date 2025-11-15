#!/usr/bin/env python3
"""
20-rod hard-contact friction sweep.
- Keeps periodic box size fixed (default 0.9).
- Sweeps friction coefficients and preserves per-μ CSV/scene filenames.
- Aggregates energy metrics and plots KE vs μ (and drift% vs μ) if matplotlib is available.
"""
import json, argparse, numpy as np, pandas as pd, subprocess, time, os, math

DEF_DT_HARD = 1.0/600.0


def build_scene(path: str, box_size=0.90, rod_length=0.50, rod_diameter=0.02,
                density=1000.0, lin_speed=1.0, ang_speed=5.0, n_rods=20, seed=2025,
                friction=0.0, restitution=1.0,
                noise_enabled=False, noise_fsigma=0.0, noise_tau=0.0, noise_seed=0,
                contact_model="hard", soft_delta=0.005, soft_k=1000.0, soft_nu=1e-3,
                soft_enable_friction=True, soft_verbose=False,
                dt: float = None):
    rng = np.random.default_rng(seed)
    half = box_size/2.0
    bodies = []
    margin = rod_length*0.5
    for i in range(n_rods):
        pos = rng.uniform(-half+margin, half-margin, size=3)
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        v_dir = rng.normal(size=3); v_dir /= np.linalg.norm(v_dir)
        w_dir = rng.normal(size=3); w_dir /= np.linalg.norm(w_dir)
        v_lin = (lin_speed * v_dir).tolist()
        v_ang = (ang_speed * w_dir).tolist()
        bodies.append({
            "pos": pos.tolist(),
            "length": rod_length,
            "diameter": rod_diameter,
            "density": density,
            "rot_axis": axis.tolist(),
            "rot_deg": 0.0,
            "v_lin": v_lin,
            "v_ang": v_ang,
            "restitution": restitution,
            "friction": friction
        })
    scene = {
        "physics": {
            "gravity": [0.0,0.0,0.0],
            "dt": (dt if dt is not None else DEF_DT_HARD),
            "lin_damp": 0.0,
            "ang_damp": 0.0,
            "friction": 0.0,
            "restitution": restitution,
            "soft_contact": {
                "enabled": (contact_model == "soft"),
                "delta": soft_delta,
                "k_scaler": soft_k,
                "mu": friction,  # tie sweep friction to soft contact friction
                "nu": soft_nu,
                "enable_friction": soft_enable_friction,
                "verbose": soft_verbose
            },
            "solver": {
                "velIters": 100,
                "baumgarte": 0.0,
                "allowedPen": 0.0005,
                "splitImpulse": False,
                "splitOrient": False
            }
        },
        "scene": {
            "periodic": {
                "enabled": True,
                "min": [-half,-half,-half],
                "max": [half,half,half]
            },
            "bodies": bodies
        }
    }
    # Optional Gaussian random force/torque injection (applied every step)
    if noise_enabled:
        scene["scene"]["randomForce"] = {
            "enabled": True,
            "fSigma": float(noise_fsigma),  # translational force stddev
            "tauMag": float(noise_tau),     # rotational torque magnitude
            "seed": int(noise_seed)
        }
    with open(path,'w') as f: json.dump(scene,f,indent=2)
    return scene


def run_sim(scene_file: str, out_csv: str, steps: int, contact_model: str, dt: float, soft_pe_path: str = None):
    # Resolve simulator binary relative to this script to avoid CWD issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viewer_bin = os.path.abspath(os.path.join(script_dir, "..", "build", "rigidbody_viewer_3d"))
    if not os.path.exists(viewer_bin):
        raise FileNotFoundError(f"Simulator binary not found at {viewer_bin}. Please build it or adjust the path.")
    cmd = [viewer_bin,"--headless","--scene",scene_file,
        "--steps",str(steps),"--perrod",out_csv,"--perrod-max",str(steps),
        "--velIters","100","--dt",f"{dt}","--no-split-impulse","--no-split-orient"]
    # Soft contact potential energy logging (separate CSV) if requested
    if contact_model == 'soft' and soft_pe_path:
        cmd += ["--soft-pe", soft_pe_path]
    print("Running hard-contact:"," ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd,capture_output=True,text=True)
    dt_wall = time.time()-t0
    if proc.returncode != 0:
        print(proc.stderr)
        raise SystemExit(f"Simulation failed rc={proc.returncode}")
    print(f"Finished in {dt_wall:.2f}s wall time")


def aggregate_energy(csv_path: str, steady_frac: float = 0.2):
    df = pd.read_csv(csv_path)
    grp = df.groupby('frame')
    ke_total = grp['KE_total'].sum()
    has_lin = 'KE_lin' in df.columns
    has_rot = 'KE_rot' in df.columns
    ke_lin = grp['KE_lin'].sum() if has_lin else None
    ke_rot = grp['KE_rot'].sum() if has_rot else None

    ke0 = ke_total.iloc[0]; kef = ke_total.iloc[-1]
    drift_pct = (kef-ke0)/ke0*100.0 if ke0!=0 else float('nan')
    span_pct = (ke_total.max()-ke_total.min())/ke_total.mean()*100.0 if ke_total.mean()!=0 else float('nan')
    # Steady-state metrics over last X% of frames
    n = len(ke_total)
    s0 = max(0, int((1.0 - max(0.0, min(1.0, steady_frac))) * n))
    ke_tail = ke_total.iloc[s0:]
    lin_tail = (ke_lin.iloc[s0:] if ke_lin is not None else None)
    rot_tail = (ke_rot.iloc[s0:] if ke_rot is not None else None)
    return dict(
        initial=ke0,
        final=kef,
        drift=drift_pct,
        span=span_pct,
        mean=ke_total.mean(),
        frames=len(ke_total),
        series=ke_total.values,
        steady_mean=float(ke_tail.mean()),
        steady_std=float(ke_tail.std(ddof=0)),
        lin_initial=(ke_lin.iloc[0] if ke_lin is not None else float('nan')),
        lin_final=(ke_lin.iloc[-1] if ke_lin is not None else float('nan')),
        lin_mean=(ke_lin.mean() if ke_lin is not None else float('nan')),
        lin_steady_mean=(float(lin_tail.mean()) if lin_tail is not None else float('nan')),
        lin_steady_std=(float(lin_tail.std(ddof=0)) if lin_tail is not None else float('nan')),
        rot_initial=(ke_rot.iloc[0] if ke_rot is not None else float('nan')),
        rot_final=(ke_rot.iloc[-1] if ke_rot is not None else float('nan')),
        rot_mean=(ke_rot.mean() if ke_rot is not None else float('nan')),
        rot_steady_mean=(float(rot_tail.mean()) if rot_tail is not None else float('nan')),
        rot_steady_std=(float(rot_tail.std(ddof=0)) if rot_tail is not None else float('nan')),
    )


def parse_mu_list(s: str):
    if not s:
        return [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    vals = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok: continue
        vals.append(float(tok))
    return vals

def parse_float_list(s: str):
    if s is None:
        return []
    vals = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds',type=float,default=33.333,help='Runtime in seconds (default ~20k frames)')
    ap.add_argument('--frames',type=int,default=0,help='Override seconds with exact frame count (e.g., 20000 or 60000)')
    ap.add_argument('--dt', type=float, default=DEF_DT_HARD, help='Simulation time step (default 1/600 ≈ 0.001667). Use smaller (e.g., 1e-5) for stability experiments.')
    ap.add_argument('--box',type=float,default=0.90,help='Periodic box size; keep same per request')
    ap.add_argument('--length',type=float,default=0.50)
    ap.add_argument('--diameter',type=float,default=0.02)
    ap.add_argument('--lin-speed',type=float,default=1.0)
    ap.add_argument('--ang-speed',type=float,default=5.0)
    ap.add_argument('--seed',type=int,default=2025)
    ap.add_argument('--n',type=int,default=20)
    ap.add_argument('--mu-list',type=str,default='0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0')
    ap.add_argument('--outdir',type=str,default='.',help='Output directory for scenes/CSVs/plots')
    # Noise controls (optional): keep constant across mu sweep unless you vary via separate runs
    ap.add_argument('--noise', action='store_true', help='Enable Gaussian random force/torque each step')
    ap.add_argument('--noise-fsigma', type=float, default=0.0, help='Stddev for translational random force (N)')
    ap.add_argument('--noise-tau', type=float, default=0.0, help='Magnitude for random torque per step (N·m)')
    ap.add_argument('--noise-seed', type=int, default=0, help='Seed for noise RNG (0 => random)')
    ap.add_argument('--fsigma-list', type=str, default='', help='Comma-separated list of fSigma values to sweep (tauMag will match fSigma). If provided, overrides single --noise-fsigma/tau for runs.')
    ap.add_argument('--seed-list', type=str, default='', help='Comma-separated list of RNG seeds for multi-seed runs (applies to noise).')
    ap.add_argument('--steady-frac', type=float, default=0.2, help='Fraction of tail used to compute steady-state means/std (default 0.2)')
    ap.add_argument('--post-only', action='store_true', help='Skip simulations; just load existing summary CSV in outdir and produce derived plots (ratio, etc).')
    # Contact model selection & soft parameters
    ap.add_argument('--contact-model', type=str, choices=['hard','soft'], default='hard', help='Choose contact model: hard impulse or soft penalty.')
    ap.add_argument('--soft-delta', type=float, default=0.005, help='Soft contact transition width.')
    ap.add_argument('--soft-k', type=float, default=1000.0, help='Soft contact stiffness scaler.')
    ap.add_argument('--soft-nu', type=float, default=1e-3, help='Soft contact sticking velocity threshold.')
    ap.add_argument('--soft-no-friction', action='store_true', help='Disable friction forces in soft contact model.')
    ap.add_argument('--soft-verbose', action='store_true', help='Verbose soft contact debug output.')
    ap.add_argument('--both-models', action='store_true', help='Run both contact models (hard and soft) in a single sweep, ignoring --contact-model selection.')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    dt = args.dt
    steps = args.frames if args.frames>0 else int(args.seconds/dt)
    mus = parse_mu_list(args.mu_list)
    fsigmas = parse_float_list(args.fsigma_list)
    seed_list = [int(s) for s in args.seed_list.split(',') if s.strip()] if args.seed_list else [args.noise_seed]

    # --- Post-only early exit path: load existing summaries and jump to plotting ---
    if args.post_only:
        summary_csv = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_summary.csv")
        if not os.path.exists(summary_csv):
            raise SystemExit(f"post-only requested but summary file not found: {summary_csv}")
        summary_df = pd.read_csv(summary_csv)
        agg_csv = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_summary_agg.csv")
        agg_df = pd.read_csv(agg_csv) if os.path.exists(agg_csv) else None
        # Skip simulation loops entirely by jumping to plotting section below
        print(f"Loaded existing summary rows={len(summary_df)} post-only mode")
        # Plot sections expect summary_df and possibly agg_df; proceed to plotting block
        # Re-use existing plotting logic by setting a flag and going to plotting
        goto_plotting = True
    else:
        goto_plotting = False

    summary_rows = []
    if goto_plotting:
        # Skip filling summary_rows; jump to summary/plotting area
        pass
    # Determine which contact models to sweep
    contact_models = ['hard','soft'] if args.both_models else [args.contact_model]
    for mu in mus:
        if goto_plotting:
            break  # ensure we do not enter simulation when post-only
        mu_tag = f"{mu:.2f}".replace('.','_')
        # Choose list of fsigmas: if provided use it; else fall back to single-value mode
        run_fsigmas = fsigmas if len(fsigmas)>0 else [args.noise_fsigma]
        # Store series for overlay plotting per mu
        per_mu_series = []  # list of (contact_model, fsigma, mean_series, std_series, nseeds) for overlay plots
        for fs in run_fsigmas:
            # Determine noise enablement and tau matching fs
            noise_on = args.noise or (fs != 0.0)
            tau = fs if args.fsigma_list else args.noise_tau if args.noise else fs
            # Iterate selected contact models
            for contact_model in contact_models:
                seed_series = []
                for s in seed_list:
                    extra = ""
                    if noise_on:
                        fs_tag = ("%g" % fs).replace('.','_')
                        tau_tag = ("%g" % tau).replace('.','_')
                        extra = f"_noise_f{fs_tag}_t{tau_tag}"
                    seed_tag = f"_seed{s}" if len(seed_list) > 1 else ""
                    scene_path = os.path.join(args.outdir, f"scene_confined_n{args.n}_{contact_model}_mu{mu_tag}{extra}{seed_tag}.json")
                    csv_path = os.path.join(args.outdir, f"confined_n{args.n}_{contact_model}_mu{mu_tag}{extra}{seed_tag}.csv")
                    soft_pe_path = None
                    if contact_model == 'soft':
                        soft_pe_path = os.path.join(args.outdir, f"soft_pe_n{args.n}_mu{mu_tag}{extra}{seed_tag}.csv")

                    scene = build_scene(scene_path, box_size=args.box, rod_length=args.length, rod_diameter=args.diameter,
                                        lin_speed=args.lin_speed, ang_speed=args.ang_speed, n_rods=args.n, seed=args.seed,
                                        friction=mu, restitution=1.0,
                                        noise_enabled=noise_on, noise_fsigma=fs,
                                        noise_tau=tau, noise_seed=s,
                                        contact_model=contact_model,
                                        soft_delta=args.soft_delta, soft_k=args.soft_k, soft_nu=args.soft_nu,
                                        soft_enable_friction=(not args.soft_no_friction), soft_verbose=args.soft_verbose,
                                        dt=dt)
                    print(f"Scene -> {scene_path} | contact={contact_model} | μ={mu:.2f} | fSigma={fs:g} | seed={s} | rods={len(scene['scene']['bodies'])} | dt={dt:.6e} steps={steps}")
                    run_sim(scene_path, csv_path, steps, contact_model, dt, soft_pe_path)
                    metrics = aggregate_energy(csv_path, steady_frac=args.steady_frac)
                    # Load soft potential energy metrics if soft contact enabled
                    soft_pe_final = float('nan'); soft_pe_steady = float('nan')
                    if soft_pe_path and os.path.exists(os.path.join(os.path.dirname(csv_path), os.path.basename(soft_pe_path))):
                        try:
                            pe_df = pd.read_csv(soft_pe_path)
                            if 'soft_PE' in pe_df.columns:
                                soft_pe_final = float(pe_df['soft_PE'].iloc[-1])
                                # steady-state tail analogous to KE tail
                                n_pe = len(pe_df['soft_PE'])
                                s0_pe = max(0, int((1.0 - max(0.0, min(1.0, args.steady_frac))) * n_pe))
                                soft_pe_steady = float(pe_df['soft_PE'].iloc[s0_pe:].mean())
                        except Exception as e:
                            print(f"Soft PE load failed: {e}")
                    print(f"  KE: initial={metrics['initial']:.6f} final={metrics['final']:.6f} drift={metrics['drift']:+.4f}% span={metrics['span']:.4f}% steady_mean={metrics['steady_mean']:.6f}")
                    seed_series.append(metrics['series'])
                    summary_rows.append({
                        'mu': mu,
                        'seed': s,
                        'csv': os.path.basename(csv_path),
                        'noise_enabled': bool(noise_on),
                        'noise_fsigma': float(fs if noise_on else 0.0),
                        'noise_tau': float(tau if noise_on else 0.0),
                        'contact_model': contact_model,
                        'initial': metrics['initial'],
                        'final': metrics['final'],
                        'drift': metrics['drift'],
                        'span': metrics['span'],
                        'mean': metrics['mean'],
                        'steady_mean': metrics['steady_mean'],
                        'steady_std': metrics['steady_std'],
                        'lin_initial': metrics['lin_initial'],
                        'lin_final': metrics['lin_final'],
                        'lin_mean': metrics['lin_mean'],
                        'lin_steady_mean': metrics['lin_steady_mean'],
                        'lin_steady_std': metrics['lin_steady_std'],
                        'rot_initial': metrics['rot_initial'],
                        'rot_final': metrics['rot_final'],
                        'rot_mean': metrics['rot_mean'],
                        'rot_steady_mean': metrics['rot_steady_mean'],
                        'rot_steady_std': metrics['rot_steady_std'],
                        'soft_pe_final': soft_pe_final,
                        'soft_pe_steady_mean': soft_pe_steady,
                    })
                # Aggregate series across seeds for plotting bands (per contact model separately)
                if seed_series:
                    A = np.stack(seed_series, axis=0)
                    per_mu_series.append((contact_model, fs, A.mean(axis=0), A.std(axis=0), A.shape[0]))

        # After completing runs for this mu, plot overlaid KE vs time curves across fsigma
        try:
            if len(per_mu_series) > 0:
                import matplotlib.pyplot as plt
                times = np.arange(0, steps) * dt
                fig, ax = plt.subplots(figsize=(7,4))
                for model, fs, mean_series, std_series, nseeds in per_mu_series:
                    ax.plot(times[:len(mean_series)], mean_series, label=f"{model}-f={fs:g} (n={nseeds})")
                    if nseeds > 1:
                        ax.fill_between(times[:len(mean_series)],
                                        mean_series - std_series,
                                        mean_series + std_series,
                                        alpha=0.15)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Total KE (J)')
                ax.set_title(f'KE vs time | μ={mu:.2f}')
                ax.grid(True, alpha=0.3)
                ax.legend(title='fSigma=tau')
                fig.tight_layout()
                plot_ts = os.path.join(args.outdir, f"ke_timeseries_n{args.n}_box{args.box:.2f}_mu{mu_tag}.png")
                fig.savefig(plot_ts, dpi=150)
                print("Saved:", plot_ts)

                # Optional log-y version
                fig2, ax2 = plt.subplots(figsize=(7,4))
                for model, fs, mean_series, std_series, nseeds in per_mu_series:
                    ax2.semilogy(times[:len(mean_series)], np.maximum(mean_series, 1e-12), label=f"{model}-f={fs:g} (n={nseeds})")
                    if nseeds > 1:
                        lo = np.maximum(mean_series - std_series, 1e-12)
                        hi = np.maximum(mean_series + std_series, 1e-12)
                        ax2.fill_between(times[:len(mean_series)], lo, hi, alpha=0.15)
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Total KE (J) [log]')
                ax2.set_title(f'KE vs time (log) | μ={mu:.2f}')
                ax2.grid(True, alpha=0.3)
                ax2.legend(title='fSigma=tau')
                fig2.tight_layout()
                plot_ts_log = os.path.join(args.outdir, f"ke_timeseries_n{args.n}_box{args.box:.2f}_mu{mu_tag}_log.png")
                fig2.savefig(plot_ts_log, dpi=150)
                print("Saved:", plot_ts_log)
        except Exception as e:
            print("Per-μ time-series plotting skipped:", e)

    # If post-only mode: load existing summary and skip new simulation loops
    if args.post_only:
        summary_csv = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_summary.csv")
        if not os.path.exists(summary_csv):
            raise SystemExit(f"post-only requested but summary file not found: {summary_csv}")
        summary_df = pd.read_csv(summary_csv)
        agg_csv = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_summary_agg.csv")
        agg_df = pd.read_csv(agg_csv) if os.path.exists(agg_csv) else None
    else:
        summary_df = pd.DataFrame(summary_rows).sort_values(['contact_model','mu','noise_fsigma','seed'] if 'seed' in summary_rows[0] else ['contact_model','mu'])
        summary_csv = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print("\nSaved:", summary_csv)
        # Include contact model and soft PE metrics in the printed subset for clarity
        cols_to_show = ['contact_model','mu','noise_fsigma','seed','initial','final','drift','span','steady_mean','soft_pe_final','soft_pe_steady_mean']
        print(summary_df[[c for c in cols_to_show if c in summary_df.columns]])
        # Aggregated stats across seeds (if multi-seed)
        agg_df = None
        if 'seed' in summary_df.columns and summary_df['seed'].nunique() > 1:
            agg = summary_df.groupby(['contact_model','mu','noise_fsigma']).agg(
                final_mean=('final','mean'), final_std=('final','std'),
                drift_mean=('drift','mean'), drift_std=('drift','std'),
                steady_mean_mean=('steady_mean','mean'), steady_mean_std=('steady_mean','std'),
                lin_final_mean=('lin_final','mean'), lin_final_std=('lin_final','std'),
                rot_final_mean=('rot_final','mean'), rot_final_std=('rot_final','std'),
                count=('final','count')
            ).reset_index()
            agg_df = agg
            agg_csv = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_summary_agg.csv")
            agg.to_csv(agg_csv, index=False)
            print("Saved:", agg_csv)

    # Plot KE vs mu and drift% vs mu; handle multi-fsigma by drawing one line per fSigma
    try:
        import matplotlib.pyplot as plt
        uniq_fs = sorted(set(summary_df['noise_fsigma'].tolist())) if 'noise_fsigma' in summary_df.columns else [0.0]
        if len(uniq_fs) <= 1:
            fig, ax = plt.subplots(1,2, figsize=(10,4))
            sdf = summary_df.sort_values(['contact_model','mu'])
            ax[0].plot(sdf['mu'], sdf['final'], 'o-')
            ax[0].set_xlabel('μ')
            ax[0].set_ylabel('Final KE (J)')
            ax[0].set_title('Final KE vs μ')
            ax[0].grid(True, alpha=0.3)

            ax[1].plot(sdf['mu'], sdf['drift'], 's-')
            ax[1].set_xlabel('μ')
            ax[1].set_ylabel('Drift (%) = (final-initial)/initial*100')
            ax[1].set_title('KE drift vs μ')
            ax[1].grid(True, alpha=0.3)

            fig.tight_layout()
            plot_path = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_plots.png")
            fig.savefig(plot_path, dpi=150)
            print("Saved:", plot_path)
        else:
            # Multi-curve final KE vs μ for each fSigma
            fig, ax = plt.subplots(1,2, figsize=(11,4))
            if agg_df is not None:
                for contact in sorted(agg_df['contact_model'].unique()):
                    for fs in uniq_fs:
                        sdf = agg_df[(agg_df['noise_fsigma'] == fs) & (agg_df['contact_model']==contact)].sort_values('mu')
                        if sdf.empty: continue
                        ax[0].errorbar(sdf['mu'], sdf['final_mean'], yerr=sdf['final_std'], fmt='-o', capsize=3, label=f"{contact}-f{fs:g}")
                        ax[1].errorbar(sdf['mu'], sdf['drift_mean'], yerr=sdf['drift_std'], fmt='-s', capsize=3, label=f"{contact}-f{fs:g}")
            else:
                for contact in sorted(summary_df['contact_model'].unique()):
                    for fs in uniq_fs:
                        sdf = summary_df[(summary_df['noise_fsigma'] == fs) & (summary_df['contact_model']==contact)].sort_values('mu')
                        if sdf.empty: continue
                        ax[0].plot(sdf['mu'], sdf['final'], marker='o', label=f"{contact}-f{fs:g}")
                        ax[1].plot(sdf['mu'], sdf['drift'], marker='s', label=f"{contact}-f{fs:g}")
            ax[0].set_xlabel('μ'); ax[0].set_ylabel('Final KE (J)'); ax[0].set_title('Final KE vs μ'); ax[0].grid(True, alpha=0.3)
            ax[1].set_xlabel('μ'); ax[1].set_ylabel('Drift (%)'); ax[1].set_title('KE drift vs μ'); ax[1].grid(True, alpha=0.3)
            ax[0].legend(title='model & fSigma'); ax[1].legend(title='model & fSigma')
            fig.tight_layout()
            plot_path = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_plots.png")
            fig.savefig(plot_path, dpi=150)
            print("Saved:", plot_path)

        # Linear vs Rotational breakdown (single or multi-fsigma)
        if 'lin_final' in summary_df.columns and 'rot_final' in summary_df.columns:
            if len(uniq_fs) <= 1:
                fig2, ax2 = plt.subplots(1,2, figsize=(10,4))
                sdf = summary_df.sort_values('mu')
                ax2[0].plot(sdf['mu'], sdf['lin_final'], 'o-', color='tab:blue')
                ax2[0].set_xlabel('μ')
                ax2[0].set_ylabel('Final Linear KE (J)')
                ax2[0].set_title('Final Linear KE vs μ')
                ax2[0].grid(True, alpha=0.3)
                ax2[1].plot(sdf['mu'], sdf['rot_final'], 's-', color='tab:orange')
                ax2[1].set_xlabel('μ')
                ax2[1].set_ylabel('Final Rotational KE (J)')
                ax2[1].set_title('Final Rotational KE vs μ')
                ax2[1].grid(True, alpha=0.3)
                fig2.tight_layout()
                plot2_path = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_linrot.png")
                fig2.savefig(plot2_path, dpi=150)
                print("Saved:", plot2_path)
            else:
                fig2, ax2 = plt.subplots(1,2, figsize=(11,4))
                for fs in uniq_fs:
                    sdf = summary_df[summary_df['noise_fsigma'] == fs].sort_values('mu')
                    ax2[0].plot(sdf['mu'], sdf['lin_final'], marker='o', label=f"f={fs:g}")
                    ax2[1].plot(sdf['mu'], sdf['rot_final'], marker='s', label=f"f={fs:g}")
                ax2[0].set_xlabel('μ'); ax2[0].set_ylabel('Final Linear KE (J)'); ax2[0].set_title('Final Linear KE vs μ'); ax2[0].grid(True, alpha=0.3)
                ax2[1].set_xlabel('μ'); ax2[1].set_ylabel('Final Rotational KE (J)'); ax2[1].set_title('Final Rotational KE vs μ'); ax2[1].grid(True, alpha=0.3)
                ax2[0].legend(title='fSigma=tau'); ax2[1].legend(title='fSigma=tau')
                fig2.tight_layout()
                plot2_path = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_linrot.png")
                fig2.savefig(plot2_path, dpi=150)
                print("Saved:", plot2_path)
    except Exception as e:
        print("Plotting skipped:", e)

    # =============== Ratio plot: Final KE vs sqrt(fSigma)/mu ===============
    try:
        import matplotlib.pyplot as plt
        # Choose dataset for ratio plot: aggregated if present else per-seed
        if agg_df is not None:
            df_ratio = agg_df.copy()
            df_ratio['ratio'] = df_ratio.apply(lambda r: (math.sqrt(r['noise_fsigma'])/r['mu']) if r['mu']>0 else float('nan'), axis=1)
            # Linear scale plot with error bars, separated by contact_model
            fig, ax = plt.subplots(figsize=(6.5,4))
            for contact in sorted(df_ratio['contact_model'].unique()):
                sub = df_ratio[df_ratio['contact_model']==contact]
                ax.errorbar(sub['ratio'], sub['final_mean'], yerr=sub['final_std'], fmt='o', alpha=0.8, label=contact)
            ax.set_xlabel(r'$\sqrt{f_\sigma}/\mu$')
            ax.set_ylabel('Final KE (J)')
            ax.set_title('Final KE vs $\sqrt{f_\sigma}/\mu$')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            ratio_path = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_ratio.png")
            fig.savefig(ratio_path, dpi=150)
            print('Saved:', ratio_path)
            # Log y plot
            fig2, ax2 = plt.subplots(figsize=(6.5,4))
            for contact in sorted(df_ratio['contact_model'].unique()):
                sub = df_ratio[df_ratio['contact_model']==contact]
                ax2.errorbar(sub['ratio'], sub['final_mean'], yerr=sub['final_std'], fmt='o', alpha=0.8, label=contact)
            ax2.set_xlabel(r'$\sqrt{f_\sigma}/\mu$')
            ax2.set_ylabel('Final KE (J, log)')
            ax2.set_yscale('log')
            ax2.set_title('Final KE vs $\sqrt{f_\sigma}/\mu$ (log y)')
            ax2.grid(True, which='both', alpha=0.3)
            ax2.legend()
            fig2.tight_layout()
            ratio_log_path = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_ratio_log.png")
            fig2.savefig(ratio_log_path, dpi=150)
            print('Saved:', ratio_log_path)
        else:
            df_ratio = summary_df.copy()
            df_ratio['ratio'] = df_ratio.apply(lambda r: (math.sqrt(r['noise_fsigma'])/r['mu']) if r['mu']>0 else float('nan'), axis=1)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(df_ratio['ratio'], df_ratio['final'], c=df_ratio['mu'], cmap='viridis', alpha=0.8)
            ax.set_xlabel(r'$\sqrt{f_\sigma}/\mu$')
            ax.set_ylabel('Final KE (J)')
            ax.set_title('Final KE vs $\sqrt{f_\sigma}/\mu$ (per-seed)')
            ax.grid(True, alpha=0.3)
            fig.colorbar(ax.collections[0], ax=ax, label='μ')
            fig.tight_layout()
            ratio_path = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_ratio.png")
            fig.savefig(ratio_path, dpi=150)
            print('Saved:', ratio_path)
            # Log y
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.scatter(df_ratio['ratio'], df_ratio['final'], c=df_ratio['mu'], cmap='viridis', alpha=0.8)
            ax2.set_xlabel(r'$\sqrt{f_\sigma}/\mu$')
            ax2.set_ylabel('Final KE (J, log)')
            ax2.set_yscale('log')
            ax2.set_title('Final KE vs $\sqrt{f_\sigma}/\mu$ (log y, per-seed)')
            ax2.grid(True, which='both', alpha=0.3)
            fig2.colorbar(ax2.collections[0], ax=ax2, label='μ')
            fig2.tight_layout()
            ratio_log_path = os.path.join(args.outdir, f"friction_sweep_n{args.n}_box{args.box:.2f}_ratio_log.png")
            fig2.savefig(ratio_log_path, dpi=150)
            print('Saved:', ratio_log_path)
    except Exception as e:
        print('Ratio plotting skipped:', e)

if __name__=='__main__':
    main()
