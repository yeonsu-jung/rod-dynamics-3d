#!/usr/bin/env python3
"""
Hard-contact (impulse solver) kinetic energy baseline for 10 rods.
Matches initial conditions style of high-energy soft-contact test but DISABLES soft penalty contacts.
We use a larger dt typical of original hard solver (dt = 1/600 ≈ 0.0016667) and restitution=1.0.
Outputs aggregated KE drift over the run.
"""
import json, argparse, numpy as np, pandas as pd, subprocess, math, sys, pathlib, time

DEF_DT_HARD = 1.0/600.0  # original larger timestep for impulse solver

def build_scene(path: str, box_size=0.80, rod_length=0.50, rod_diameter=0.02,
                density=1000.0, lin_speed=1.0, ang_speed=5.0, n_rods=10, seed=4321):
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
            "restitution": 1.0,
            "friction": 0.0
        })
    scene = {
        "physics": {
            "gravity": [0.0,0.0,0.0],
            "dt": DEF_DT_HARD,
            "lin_damp": 0.0,
            "ang_damp": 0.0,
            "friction": 0.0,
            "restitution": 1.0,
            "soft_contact": {  # explicitly disabled
                "enabled": False
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
    with open(path,'w') as f: json.dump(scene,f,indent=2)
    return scene

def run_sim(scene_file: str, out_csv: str, steps: int, perrod_max: int):
    cmd = ["../build/rigidbody_viewer_3d","--headless","--scene",scene_file,
           "--steps",str(steps),"--perrod",out_csv,"--perrod-max",str(perrod_max),
           "--velIters","100","--dt",f"{DEF_DT_HARD}","--no-split-impulse","--no-split-orient"]
    print("Running hard-contact:"," ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd,capture_output=True,text=True)
    dt_wall = time.time()-t0
    if proc.returncode != 0:
        print(proc.stderr)
        raise SystemExit(f"Simulation failed rc={proc.returncode}")
    print(f"Finished in {dt_wall:.2f}s wall time")

def aggregate_energy(csv_path: str):
    df = pd.read_csv(csv_path)
    if not {"frame","rod","KE_total"}.issubset(df.columns):
        print("CSV missing columns; got:",df.columns.tolist()); return None
    ke_frame = df.groupby('frame')['KE_total'].sum()
    ke0 = ke_frame.iloc[0]; kef = ke_frame.iloc[-1]
    drift_pct = (kef-ke0)/ke0*100.0 if ke0!=0 else float('nan')
    max_dev_pct = (ke_frame.max()-ke_frame.min())/ke_frame.mean()*100.0 if ke_frame.mean()!=0 else float('nan')
    print("\n=== Hard-Contact Energy Summary ===")
    print(f"Frames logged: {len(ke_frame)}")
    print(f"Initial KE: {ke0:.6f} J  Final KE: {kef:.6f} J  Change: {drift_pct:+.4f}%")
    print(f"Mean KE: {ke_frame.mean():.6f} J  Drift span: {max_dev_pct:.4f}%")
    return ke_frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds',type=float,default=5.0)
    ap.add_argument('--box',type=float,default=0.80)
    ap.add_argument('--length',type=float,default=0.50)
    ap.add_argument('--diameter',type=float,default=0.02)
    ap.add_argument('--lin-speed',type=float,default=1.0)
    ap.add_argument('--ang-speed',type=float,default=5.0)
    ap.add_argument('--seed',type=int,default=4321)
    args = ap.parse_args()

    steps = int(args.seconds/DEF_DT_HARD)
    perrod_max = min(steps, steps)  # log all frames (larger dt keeps count reasonable)

    scene_path = 'scene_confined_n10_hard.json'
    csv_path = 'confined_n10_hard.csv'

    scene = build_scene(scene_path, box_size=args.box, rod_length=args.length, rod_diameter=args.diameter,
                        lin_speed=args.lin_speed, ang_speed=args.ang_speed, seed=args.seed)
    print(f"Scene -> {scene_path} | rods={len(scene['scene']['bodies'])} | dt={DEF_DT_HARD:.6f} steps={steps}")
    print(f"Simulated time: {steps*DEF_DT_HARD:.2f} s")

    run_sim(scene_path, csv_path, steps, perrod_max)
    ke_series = aggregate_energy(csv_path)
    if ke_series is None: return
    q = len(ke_series)//4
    if q>0:
        print("Quarterly KE (J):", [f"{ke_series.iloc[i*q]:.6f}" for i in range(4)])

if __name__=='__main__':
    main()
