#!/usr/bin/env python3
"""
Two-rod hard contact kinetic energy test with and without friction.
We create identical initial conditions (head-on approach) and run twice:
  1) friction=0.0
  2) friction=0.6
Soft contact disabled. Restitution=1.0 (elastic normal), Baume=0.
Outputs KE summary for each run.
"""
import json, subprocess, pandas as pd, numpy as np, argparse, time, math

DT = 1.0/600.0  # hard contact timestep
STEPS_DEFAULT = 2000  # ~3.33 seconds

def build_scene(path: str, friction: float, gap: float, speed: float, offset: float, seed: int = 2024):
    """Create a two-rod head-on (or glancing if offset!=0) scene.
    gap: half distance between centers initially (so initial center separation = 2*gap)
    speed: scalar linear speed each rod toward the other
    offset: lateral y-offset to introduce tangential component for frictional effects
    """
    rng = np.random.default_rng(seed)
    length = 0.5
    diameter = 0.05
    half_gap = gap
    # Rod A at -x moving +x, Rod B at +x moving -x (ensure collision)
    bodies = []
    bodies.append({
        "pos": [-half_gap, offset, 0.0],
        "length": length,
        "diameter": diameter,
        "density": 1000.0,
        "rot_axis": [0,1,0],
        "rot_deg": 0.0,
        "v_lin": [speed, 0.0, 0.0],
        "v_ang": [0.0, 0.0, 0.0],
        "restitution": 1.0,
        "friction": friction
    })
    bodies.append({
        "pos": [half_gap, -offset, 0.0],
        "length": length,
        "diameter": diameter,
        "density": 1000.0,
        "rot_axis": [1,0,0],
        "rot_deg": 0.0,
        "v_lin": [-speed, 0.0, 0.0],
        "v_ang": [0.0, 0.0, 0.0],
        "restitution": 1.0,
        "friction": friction
    })
    scene = {
        "physics": {
            "gravity": [0.0,0.0,0.0],
            "dt": DT,
            "lin_damp": 0.0,
            "ang_damp": 0.0,
            "friction": 0.0,
            "restitution": 1.0,
            "soft_contact": {"enabled": False},
            "solver": {
                "velIters": 100,
                "baumgarte": 0.0,
                "allowedPen": 0.0005,
                "splitImpulse": False,
                "splitOrient": False
            }
        },
        "scene": {
            "periodic": {"enabled": False},
            "bodies": bodies
        }
    }
    with open(path,'w') as f: json.dump(scene,f,indent=2)
    return scene

def run(scene_file: str, out_csv: str, steps: int):
    cmd = ["../build/rigidbody_viewer_3d","--headless","--scene",scene_file,
        "--steps",str(steps),"--perrod",out_csv,"--perrod-max",str(steps),
        "--velIters","100","--dt",f"{DT}","--no-split-impulse","--no-split-orient"]
    print("Running:"," ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd,capture_output=True,text=True)
    dt_wall = time.time()-t0
    if proc.returncode!=0:
        print(proc.stderr)
        raise SystemExit("Simulation failed")
    print(f"Finished in {dt_wall:.2f}s")

def visualize(scene_file: str):
    """Launch OpenGL viewer (non-headless) for the provided scene file."""
    cmd = ["../build/rigidbody_viewer_3d","--scene",scene_file]
    print("Launching viewer:"," ".join(cmd))
    # Do not capture output; allow interactive window.
    subprocess.Popen(cmd)

def summarize(csv_path: str, length: float):
    df = pd.read_csv(csv_path)
    # Distance between centers per frame using position columns px,py,pz
    try:
        pivot = df.pivot(index='frame', columns='rod', values=['px','py','pz'])
        frames = pivot.index.values
        x0 = pivot['px'][0].values; y0 = pivot['py'][0].values; z0 = pivot['pz'][0].values
        x1 = pivot['px'][1].values; y1 = pivot['py'][1].values; z1 = pivot['pz'][1].values
        d = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
        contact_threshold = length  # crude: when center distance <= length treat as contact
        contact_frames = np.where(d <= contact_threshold)[0]
        first_contact = int(frames[contact_frames[0]]) if contact_frames.size>0 else None
    except Exception as e:
        print(f"Position pivot failed: {e}")
        first_contact = None
    grouped = df.groupby('frame')
    ke_frame = grouped['KE_total'].sum()
    ke_lin_frame = grouped['KE_lin'].sum()
    ke_rot_frame = grouped['KE_rot'].sum()
    ke0 = ke_frame.iloc[0]; kef = ke_frame.iloc[-1]
    lin0 = ke_lin_frame.iloc[0]; linf = ke_lin_frame.iloc[-1]
    rot0 = ke_rot_frame.iloc[0]; rotf = ke_rot_frame.iloc[-1]
    drift_pct = (kef-ke0)/ke0*100.0 if ke0!=0 else float('nan')
    max_dev = (ke_frame.max()-ke_frame.min())/ke_frame.mean()*100.0 if ke_frame.mean()!=0 else float('nan')
    print("\nKE Summary:")
    print(f" Total KE: initial={ke0:.6f} J final={kef:.6f} J drift={drift_pct:+.4f}%")
    print(f" Linear KE: initial={lin0:.6f} J final={linf:.6f} J")
    print(f" Rotational KE: initial={rot0:.6f} J final={rotf:.6f} J")
    print(f" Mean total={ke_frame.mean():.6f} J span={max_dev:.4f}% frames={len(ke_frame)}")
    if first_contact is not None:
        print(f" First contact frame (threshold d <= {contact_threshold:.3f}): {first_contact}")
    else:
        print(" No contact detected under threshold criterion.")
    return dict(initial=ke0, final=kef, drift=drift_pct, span=max_dev, first_contact=first_contact,
                lin_initial=lin0, lin_final=linf, rot_initial=rot0, rot_final=rotf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=STEPS_DEFAULT)
    ap.add_argument('--frictionA',type=float,default=0.0,help='First friction scenario')
    ap.add_argument('--frictionB',type=float,default=0.6,help='Second friction scenario (ignored if --friction-list used)')
    ap.add_argument('--friction-list',type=str,default='',help='Comma separated list of friction coefficients to sweep (overrides A/B)')
    ap.add_argument('--visualize',action='store_true',help='If set, only build first scene and open OpenGL viewer (no headless runs)')
    ap.add_argument('--gap',type=float,default=0.6,help='Initial half-gap between centers (so sep = 2*gap)')
    ap.add_argument('--speed',type=float,default=0.8,help='Linear speed magnitude toward other rod')
    ap.add_argument('--offset',type=float,default=0.05,help='Lateral y-offset to create tangential component')
    args = ap.parse_args()

    length = 0.5
    closing_speed = 2*args.speed
    initial_sep = 2*args.gap
    end_gap = initial_sep - length  # distance between end caps initially
    t_collision = end_gap/closing_speed if closing_speed>0 else math.inf
    frame_collision_pred = int(math.ceil(t_collision/DT)) if closing_speed>0 else None
    print(f"Predicted collision time ~{t_collision:.4f}s at frame ~{frame_collision_pred} (dt={DT:.6f})")
    print(f"Initial center separation={initial_sep:.3f}, end-to-end gap={end_gap:.3f}")
    print(f"Offset (y)={args.offset:.3f} -> introduces tangential component for frictional effects")

    if args.visualize:
        # Build a single scene (use frictionA) and launch viewer, then exit.
        build_scene('scene_two_rods_view.json', friction=args.frictionA, gap=args.gap, speed=args.speed, offset=args.offset)
        visualize('scene_two_rods_view.json')
        return

    if args.friction_list:
        frictions = [float(f.strip()) for f in args.friction_list.split(',') if f.strip()]
    else:
        frictions = [args.frictionA, args.frictionB]

    results = []
    for fr in frictions:
        tag = f"{fr}".replace('.','_')
        scene_file = f"scene_two_rods_f{tag}.json"
        out_csv = f"two_rods_f{tag}.csv"
        build_scene(scene_file, friction=fr, gap=args.gap, speed=args.speed, offset=args.offset)
        run(scene_file, out_csv, args.steps)
        res = summarize(out_csv, length)
        res['friction'] = fr
        results.append(res)

    print("\nSweep Summary:")
    header = f"{'fric':>6} {'KE0':>10} {'KEf':>10} {'drift%':>8} {'lin0':>10} {'linf':>10} {'rot0':>10} {'rotf':>10} {'contact':>8}"
    print(header)
    for r in results:
        print(f"{r['friction']:6.3f} {r['initial']:10.6f} {r['final']:10.6f} {r['drift']:8.4f} {r['lin_initial']:10.6f} {r['lin_final']:10.6f} {r['rot_initial']:10.6f} {r['rot_final']:10.6f} {str(r['first_contact']):>8}")

if __name__=='__main__':
    main()
