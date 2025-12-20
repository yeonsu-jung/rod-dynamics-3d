
"""
Analyze initial pair distance distribution for colliding vs non-colliding pairs.
Supports both perrod.csv (CSV) and traj_AR*.txt (TXT) formats.
"""

import argparse
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import re
from pathlib import Path
from tqdm import tqdm

def load_scene_params(scene_json):
    with open(scene_json, 'r') as f:
        data = json.load(f)
    
    bodies = data.get('scene', {}).get('bodies', [])
    if not bodies:
        raise ValueError("No bodies found in scene.json")
    
    rod_params = bodies[0]
    length = rod_params.get('length', 1.0)
    if 'diameter' in rod_params:
        radius = rod_params['diameter'] / 2.0
    else:
        radius = rod_params.get('radius', 0.005)
        
    return length, radius

def q_to_rot_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

def spherical_to_director(theta, phi):
    """
    Convert spherical coordinates to director vector.
    Assuming standard physics convention: 
    z = cos(theta)
    x = sin(theta)cos(phi)
    y = sin(theta)sin(phi)
    """
    st = math.sin(theta)
    ct = math.cos(theta)
    sp = math.sin(phi)
    cp = math.cos(phi)
    return np.array([st*cp, st*sp, ct])

def get_rod_endpoints_q(pos, rot_q, length):
    R = q_to_rot_matrix(rot_q)
    half_axis = np.array([0, 0, length / 2.0]) 
    v = R @ half_axis
    return pos - v, pos + v

def get_rod_endpoints_dir(pos, director, length):
    v = director * (length / 2.0)
    return pos - v, pos + v

def load_trajectory_csv(csv_path):
    data = {}
    frames = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_idx = int(row['frame'])
            r_idx = int(row['rod'])
            
            px, py, pz = float(row['px']), float(row['py']), float(row['pz'])
            qw, qx, qy, qz = float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz'])
            
            if f_idx not in data:
                data[f_idx] = {}
                frames.append(f_idx)
            
            data[f_idx][r_idx] = {
                'pos': np.array([px, py, pz]),
                'q': np.array([qw, qx, qy, qz]),
                'type': 'quat'
            }
            
    frames = sorted(list(set(frames)))
    return frames, data

def load_trajectory_txt(txt_path):
    """
    Parses traj_AR*.txt format:
    # rod_radius = 0.05
    FRAME 0 ...
    x y z theta phi length
    ...
    """
    data = {}
    frames = []
    radius = 0.05 # default
    length_val = 1.0 # default
    
    current_frame = None
    rod_idx = 0
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("#"):
                if "rod_radius" in line:
                    parts = line.split('=')
                    if len(parts) > 1:
                        try:
                            radius = float(parts[1].strip())
                        except:
                            pass
                continue
            
            if line.startswith("FRAME"):
                parts = line.split()
                try:
                    current_frame = int(parts[1])
                    if current_frame not in data:
                        data[current_frame] = {}
                        frames.append(current_frame)
                    rod_idx = 0
                except:
                    current_frame = None
                continue
            
            if current_frame is not None:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        px, py, pz = float(parts[0]), float(parts[1]), float(parts[2])
                        theta, phi = float(parts[3]), float(parts[4])
                        l_val = float(parts[5]) if len(parts) > 5 else 1.0
                        length_val = l_val # Assume constant length
                        
                        data[current_frame][rod_idx] = {
                            'pos': np.array([px, py, pz]),
                            'theta': theta,
                            'phi': phi,
                            'length': l_val,
                            'type': 'spherical'
                        }
                        rod_idx += 1
                    except ValueError:
                        continue

    frames = sorted(list(set(frames)))
    return frames, data, radius, length_val

def dist_segment_segment(p1, p2, q1, q2):
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-8 and e < 1e-8:
        return np.linalg.norm(r)
    if a < 1e-8:
        t = np.clip(np.dot(q2-q1, p1-q1) / e, 0.0, 1.0)
        return np.linalg.norm(p1 - (q1 + t*d2))
    if e < 1e-8:
        s = np.clip(np.dot(p2-p1, q1-p1) / a, 0.0, 1.0)
        return np.linalg.norm((p1 + s*d1) - q1)

    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a*e - b*b
    
    if denom < 1e-8:
        s = 0.0
        t = f / e
        t = np.clip(t, 0.0, 1.0)
    else:
        s = (b*f - c*e) / denom
        t = (b*s + f) / e 
        if s < 0.0:
            s = 0.0
            t = f / e
            t = np.clip(t, 0.0, 1.0)
        elif s > 1.0:
            s = 1.0
            t = (b + f) / e
            t = np.clip(t, 0.0, 1.0)
        
        if t < 0.0:
            t = 0.0
            s = -c / a
            s = np.clip(s, 0.0, 1.0)
        elif t > 1.0:
            t = 1.0
            s = (b - c) / a
            s = np.clip(s, 0.0, 1.0)
            
    c1 = p1 + s * d1
    c2 = q1 + t * d2
    return np.linalg.norm(c1 - c2)

def load_endpoints_txt(txt_path):
    """
    Parses x_relaxed.txt format:
    # metadata
    x1 y1 z1 x2 y2 z2
    ...
    """
    data = {}
    rod_idx = 0
    # Single frame (T=0)
    data[0] = {}
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            if len(parts) >= 6:
                try:
                    p1 = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                    p2 = np.array([float(parts[3]), float(parts[4]), float(parts[5])])
                    data[0][rod_idx] = {'p1': p1, 'p2': p2, 'type': 'endpoints'}
                    rod_idx += 1
                except ValueError:
                    continue
    return [0], data, 0.05, 1.0 # Default R, L (usually overridden by scene or header)

def analyze_dataset(frames, data, length, radius, stride, output_name, title_suffix="", initial_data=None):
    if not frames and initial_data is None:
        print("  No data found.")
        return

    # 1. Compute Initial Distances (T=0)
    # If initial_data matches 'x_relaxed', usage that.
    # formatting: initial_data is (frames=[0], data={0: {rod_id: ...}})
    
    if initial_data:
        rods0 = initial_data[1][0]
    else:
        if not frames: return
        frame0 = frames[0]
        rods0 = data[frame0]
        
    rod_indices = sorted(rods0.keys())
    
    endpoints0 = {}
    for r in rod_indices:
        entry = rods0[r]
        if entry['type'] == 'quat':
            p1, p2 = get_rod_endpoints_q(entry['pos'], entry['q'], length)
        elif entry['type'] == 'spherical':
            director = spherical_to_director(entry['theta'], entry['phi'])
            l_use = entry.get('length', length)
            p1, p2 = get_rod_endpoints_dir(entry['pos'], director, l_use)
        elif entry['type'] == 'endpoints':
            p1, p2 = entry['p1'], entry['p2']
            
        endpoints0[r] = (p1, p2)
        
    initial_dists = {} 
    pairs = []
    for i in range(len(rod_indices)):
        for j in range(i+1, len(rod_indices)):
            r1, r2 = rod_indices[i], rod_indices[j]
            p1a, p2a = endpoints0[r1]
            p1b, p2b = endpoints0[r2]
            d = dist_segment_segment(p1a, p2a, p1b, p2b)
            initial_dists[(r1, r2)] = d
            pairs.append((r1, r2))
            
    # 2. Check for collisions
    collided_pairs = set()
    threshold = 2.0 * radius + 1e-4
    
    if frames:
        print(f"  Scanning {len(frames)} frames for collisions (N={len(rod_indices)})...")
        for f in tqdm(frames[::stride]):
            rods_t = data[f]
            endpoints_t = {}
            for r in rod_indices:
                if r in rods_t:
                    entry = rods_t[r]
                    if entry['type'] == 'quat':
                        p1, p2 = get_rod_endpoints_q(entry['pos'], entry['q'], length)
                    elif entry['type'] == 'spherical':
                        director = spherical_to_director(entry['theta'], entry['phi'])
                        l_use = entry.get('length', length)
                        p1, p2 = get_rod_endpoints_dir(entry['pos'], director, l_use)
                    # Note: x_relaxed usually doesn't have trajectory, so 'endpoints' type unlikely here unless static check
                    endpoints_t[r] = (p1, p2)
            
            for (r1, r2) in pairs:
                if (r1, r2) in collided_pairs:
                    continue
                if r1 not in endpoints_t or r2 not in endpoints_t:
                    continue
                    
                p1a, p2a = endpoints_t[r1]
                p1b, p2b = endpoints_t[r2]
                d = dist_segment_segment(p1a, p2a, p1b, p2b)
                if d < threshold:
                    collided_pairs.add((r1, r2))
    
    # 3. Aggregate
    dists_colliding = []
    dists_non_colliding = []
    
    for (r1, r2), d_init in initial_dists.items():
        if (r1, r2) in collided_pairs:
            dists_colliding.append(d_init)
        else:
            dists_non_colliding.append(d_init)
            
    # 4. Plot
    if not dists_colliding and not dists_non_colliding:
        print("  No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    diameter = 2.0 * radius
    
    # Avoid zero division if radius is tiny or 0
    if diameter < 1e-9: diameter = 1.0
        
    v_col = np.array(dists_colliding) / diameter
    v_non = np.array(dists_non_colliding) / diameter
    
    if len(v_non) > 0:
        plt.hist(v_non, bins=50, alpha=0.5, label='Non-Colliding', density=True, color='blue')
    if len(v_col) > 0:
        plt.hist(v_col, bins=50, alpha=0.5, label='Colliding', density=True, color='red')
    
    plt.xlabel("Initial Min Distance / Diameter")
    plt.ylabel("Density")
    plt.title(f"Initial Pair Distance Distribution{title_suffix}\n(Collided: {len(dists_colliding)}, Non-Collided: {len(dists_non_colliding)})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_name, dpi=150)
    plt.close()
    print(f"  Saved {output_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_folder", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    
    if not args.sweep_folder.exists():
        sys.exit(f"Folder not found: {args.sweep_folder}")

    # Heuristic to detect structure
    targets = [] # (file_path, context_type, params)
    
    for path in args.sweep_folder.rglob("*"):
        if path.name == "perrod.csv":
            targets.append(('csv', path.parent, path))
        elif path.name.startswith("traj_") and path.name.endswith(".txt"):
            targets.append(('txt', path.parent, path))

    if not targets:
        print("No perrod.csv or traj_*.txt found.")
        sys.exit(0)
        
    for t_type, parent, path in sorted(targets, key=lambda x: str(x[2])):
        if t_type == 'csv':
            print(f"Processing CSV Run: {parent.name}...")
            scene_json = parent / "scene.json"
            # Check for explicitly preferred x_relaxed.txt
            x_relaxed = parent / "x_relaxed.txt"
            init_data = None
            if x_relaxed.exists():
                print("  Using x_relaxed.txt for initial state.")
                init_data = load_endpoints_txt(x_relaxed)

            if not scene_json.exists():
                print("  Missing scene.json")
                continue
            try:
                L, R = load_scene_params(scene_json)
                frames, data = load_trajectory_csv(path)
                out = parent / "initial_distance_collision_hist.png"
                analyze_dataset(frames, data, L, R, args.stride, out, title_suffix=f"\n{parent.name}", initial_data=init_data if init_data else None)
            except Exception as e:
                print(f"  Error: {e}")
                
        elif t_type == 'txt':
            print(f"Processing TXT Run: {path.name} in {parent.name}...")
            # Detect corresponding x_relaxed_AR*.txt
            # pattern: traj_AR10.txt -> x_relaxed_AR10.txt
            stem = path.stem # traj_AR10
            ar_suffix = stem.replace("traj_", "") # AR10
            x_relaxed_name = f"x_relaxed_{ar_suffix}.txt"
            x_relaxed_path = parent / x_relaxed_name
            
            init_data = None
            if x_relaxed_path.exists():
                print(f"  Using {x_relaxed_name} for initial state.")
                init_data = load_endpoints_txt(x_relaxed_path)
            
            try:
                frames, data, R, L = load_trajectory_txt(path)
                out = parent / f"initial_distance_collision_hist_{stem}.png"
                analyze_dataset(frames, data, L, R, args.stride, out, title_suffix=f"\n{parent.name}/{path.name}", initial_data=init_data if init_data else None)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  Error: {e}")

if __name__ == "__main__":
    main()
