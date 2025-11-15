#!/usr/bin/env python3
"""
High-energy 10-rod confined periodic box test for kinetic energy conservation.
Steps:
 1. Generate scene with larger rods and higher initial linear/angular velocities.
 2. Run headless simulation with soft contacts (elastic, no damping).
 3. Aggregate total KE per logged frame and report drift over runtime.

Default duration: 5 seconds (60000 steps @ dt=1/12000).
Optionally extend to 10 seconds via CLI.
"""
import json, argparse, numpy as np, pandas as pd, subprocess, pathlib, math, sys
from datetime import datetime

DEF_DT = 1.0/12000.0

def build_scene(path: str, box_size=0.80, rod_length=0.50, rod_diameter=0.02,
                density=1000.0, lin_speed=1.0, ang_speed=5.0, n_rods=10, seed=1234):
    rng = np.random.default_rng(seed)
    half = box_size/2.0
    bodies = []
    for i in range(n_rods):
        # Position with margin to avoid immediate overlap (approx half length)
        margin = rod_length*0.5
        pos = rng.uniform(-half+margin, half-margin, size=3)
        # Random axis
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        # Linear velocity random direction scaled
        v_dir = rng.normal(size=3); v_dir /= np.linalg.norm(v_dir)
        v_lin = (lin_speed * v_dir).tolist()
        # Angular velocity random
        w_dir = rng.normal(size=3); w_dir /= np.linalg.norm(w_dir)
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
            "restitution": 1.0,
            "friction": 0.0
        })
    scene = {
        "physics": {
            "gravity": [0.0,0.0,0.0],
            "dt": DEF_DT,
            "lin_damp": 0.0,
            "ang_damp": 0.0,
            "friction": 0.0,
            "restitution": 1.0,
            "soft_contact": {
                "enabled": True,
                "delta": 0.005,
                "k_scaler": 500,
                "mu": 0.0,
                "enable_friction": False,
                "verbose": False
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
    with open(path,'w') as f: json.dump(scene,f,indent=2)
    return scene

def run_sim(scene_file: str, out_csv: str, steps: int, perrod_max: int):
    cmd = ["../build/rigidbody_viewer_3d","--headless","--scene",scene_file,
           "--steps",str(steps),"--perrod",out_csv,"--perrod-max",str(perrod_max),
           "--soft-contact","--k-scaler","500","--delta","0.005","--mu","0.0","--verbose","false"]
    print("Running:"," ".join(cmd))
    t0 = datetime.now()
    proc = subprocess.run(cmd,capture_output=True,text=True)
    dt = (datetime.now()-t0).total_seconds()
    if proc.returncode!=0:
        print(proc.stderr)
        raise SystemExit(f"Simulation failed RC={proc.returncode}")
    print(f"Simulation finished in {dt:.2f}s wall time.")

def aggregate_energy(csv_path: str):
    df = pd.read_csv(csv_path)
    if not {"frame","rod","KE_total"}.issubset(df.columns):
        print("CSV missing columns; got:",df.columns.tolist()); return None
    ke_frame = df.groupby('frame')['KE_total'].sum()
    ke0 = ke_frame.iloc[0]; kef = ke_frame.iloc[-1]
    drift_pct = (kef-ke0)/ke0*100.0 if ke0!=0 else float('nan')
    max_dev_pct = (ke_frame.max()-ke_frame.min())/ke_frame.mean()*100.0 if ke_frame.mean()!=0 else float('nan')
    print("\n=== Energy Summary ===")
    print(f"Frames logged: {len(ke_frame)}")
    print(f"Initial KE: {ke0:.6f} J  Final KE: {kef:.6f} J  Change: {drift_pct:+.4f}%")
    print(f"Mean KE: {ke_frame.mean():.6f} J  Drift span: {max_dev_pct:.4f}%")
    return ke_frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds',type=float,default=5.0,help='Duration in seconds (5 or 10 recommended)')
    ap.add_argument('--box',type=float,default=0.80)
    ap.add_argument('--length',type=float,default=0.50)
    ap.add_argument('--diameter',type=float,default=0.02)
    ap.add_argument('--lin-speed',type=float,default=1.0)
    ap.add_argument('--ang-speed',type=float,default=5.0)
    ap.add_argument('--seed',type=int,default=1234)
    args = ap.parse_args()

    steps = int(args.seconds/DEF_DT)
    perrod_max = min(steps, 8000)  # cap logging to reasonable size

    scene_path = 'scene_confined_n10_high_energy.json'
    csv_path = 'confined_n10_high_energy.csv'

    scene = build_scene(scene_path, box_size=args.box, rod_length=args.length, rod_diameter=args.diameter,
                        lin_speed=args.lin_speed, ang_speed=args.ang_speed, seed=args.seed)
    print(f"Scene written -> {scene_path} | rods={len(scene['scene']['bodies'])} | steps={steps} dt={DEF_DT:.6f}")
    print(f"Target simulated time: {steps*DEF_DT:.2f} s")

    run_sim(scene_path, csv_path, steps, perrod_max)
    ke_series = aggregate_energy(csv_path)
    if ke_series is None: return

    # Simple drift metric over quarters
    q = len(ke_series)//4
    if q>0:
        print("Quarterly KE (J):", [f"{ke_series.iloc[i*q]:.6f}" for i in range(4)])

if __name__=='__main__':
    main()
