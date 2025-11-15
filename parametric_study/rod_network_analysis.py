#!/usr/bin/env python3
"""
Rod network analysis utility for per-rod trajectory CSVs produced by the simulator.

Reads a per-rod CSV (columns: frame,rod,px,py,pz,vx,vy,vz,wx,wy,wz,qw,qx,qy,qz,KE_lin,KE_rot,KE_total)
and reconstructs per-frame centroid positions, orientation axes, and endpoints for a set of rods.

Provides:
  - Endpoints array (F x N x 2 x 3)
  - Orientation axis array (F x N x 3)
  - Relative velocity norm time series (system-level dispersion metric)
  - Optional NPZ export and simple matplotlib plot.

Usage example:
  python3 rod_network_analysis.py \
      --csv runs/_both_models_demo_patch_test/confined_n20_hard_mu0_10_noise_f1e-05_t1e-05_seed1.csv \
      --length 0.5 --frame-skip 10 --plot

If you also have the scene JSON, you can omit --length and pass --scene to auto-detect the first body's length.
"""

import argparse, json, os, sys, numpy as np
from pathlib import Path
import pandas as pd

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion array q[...,4]."""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return q / norm

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by quaternion(s) q (w,x,y,z). Supports broadcasting.
    q: (...,4), v: (...,3) -> rotated (...,3)
    Formula: v' = v + 2*cross(q_vec, cross(q_vec, v) + q_w * v)
    """
    qw = q[..., 0]
    qvec = q[..., 1:4]
    # cross(q_vec, v)
    c1 = np.cross(qvec, v)
    # cross(q_vec, c1 + qw * v)
    c2 = np.cross(qvec, c1 + (qw[..., None] * v))
    return v + 2.0 * c2

def infer_length_from_scene(scene_path: str) -> float:
    try:
        with open(scene_path, 'r') as f:
            scene = json.load(f)
        bodies = scene.get('scene', {}).get('bodies', [])
        if not bodies:
            raise ValueError('No bodies in scene JSON')
        L = bodies[0].get('length')
        if L is None:
            raise ValueError('First body has no length field')
        return float(L)
    except Exception as e:
        raise RuntimeError(f"Failed to infer length from scene: {e}")

def load_perrod(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {'frame','rod','px','py','pz','qw','qx','qy','qz'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in per-rod CSV: {missing}")
    return df

def build_frame_arrays(df: pd.DataFrame):
    """Return frames, positions (F x N x 3), quaternions (F x N x 4)."""
    # Determine unique frames and rod count
    frames = np.sort(df['frame'].unique())
    rods = np.sort(df['rod'].unique())
    F = len(frames); N = len(rods)
    pos = np.zeros((F, N, 3), dtype=np.float64)
    quat = np.zeros((F, N, 4), dtype=np.float64)
    frame_index_map = {f: i for i, f in enumerate(frames)}
    # Pivot manually for speed
    for row in df.itertuples(index=False):
        fi = frame_index_map[row.frame]
        ri = row.rod
        pos[fi, ri, :] = (row.px, row.py, row.pz)
        quat[fi, ri, :] = (row.qw, row.qx, row.qy, row.qz)
    quat = quat_normalize(quat)
    return frames, pos, quat

def compute_orientation_axes(quat: np.ndarray, local_axis=(0.0, 1.0, 0.0)) -> np.ndarray:
    v = np.asarray(local_axis, dtype=np.float64)
    # Broadcast v to match quat shape
    v_b = np.broadcast_to(v, quat.shape[:-1] + (3,))
    axes = quat_rotate(quat, v_b)
    # Normalize (should already be unit length if quat is unit)
    norms = np.linalg.norm(axes, axis=-1, keepdims=True)
    axes = np.where(norms == 0, axes, axes / norms)
    return axes

def compute_endpoints(pos: np.ndarray, axes: np.ndarray, length: float):
    half = 0.5 * length
    r1 = pos - half * axes
    r2 = pos + half * axes
    # Shape (F,N,2,3)
    return np.stack([r1, r2], axis=2)

def relative_velocity_dispersion(pos: np.ndarray, frame_skip: int) -> np.ndarray:
    """Compute system-level relative velocity norm (dispersion) at sampled frames.
    pos: (F,N,3); returns array length K where K ~ F/frame_skip - 1.
    Uses forward difference between sampled frames.
    """
    sampled_indices = list(range(0, pos.shape[0]-frame_skip, frame_skip))
    vals = []
    for i in sampled_indices:
        p0 = pos[i]
        p1 = pos[i + frame_skip]
        vv = p1 - p0  # displacement over skip interval
        centroid = vv.mean(axis=0)
        rel = vv - centroid
        vals.append(np.linalg.norm(rel))
    return np.asarray(vals), sampled_indices

def analyze(csv_path: str, length: float, frame_skip: int, local_axis=(0,1,0), npz_out: str=None, plot: bool=False):
    df = load_perrod(csv_path)
    frames, pos, quat = build_frame_arrays(df)
    axes = compute_orientation_axes(quat, local_axis=local_axis)
    endpoints = compute_endpoints(pos, axes, length)
    rv_norm, sampled_idx = relative_velocity_dispersion(pos, frame_skip)

    results = {
        'frames': frames,
        'positions': pos,
        'quaternions': quat,
        'axes': axes,
        'endpoints': endpoints,
        'rv_norm': rv_norm,
        'rv_sampled_frame_indices': sampled_idx,
        'length': length,
        'num_frames': pos.shape[0],
        'num_rods': pos.shape[1]
    }

    if npz_out:
        np.savez_compressed(npz_out,
                            frames=frames,
                            positions=pos,
                            quaternions=quat,
                            axes=axes,
                            endpoints=endpoints,
                            rv_norm=rv_norm,
                            rv_sampled_frame_indices=np.asarray(sampled_idx),
                            length=length)
        print(f"Saved NPZ: {npz_out}")

    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,3))
            plt.plot(rv_norm, marker='o', ms=3)
            plt.xlabel(f'Sampled frame (skip={frame_skip})')
            plt.ylabel('Relative velocity norm')
            plt.title('Rod network relative velocity dispersion')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            out_png = (Path(npz_out).with_suffix('.png') if npz_out else Path(csv_path).with_suffix('.rv_norm.png'))
            plt.savefig(out_png, dpi=150)
            print(f"Saved plot: {out_png}")
        except Exception as e:
            print(f"Plotting failed (matplotlib not available?): {e}")

    return results

def main():
    ap = argparse.ArgumentParser(description='Analyze rod network from per-rod CSV produced by simulator.')
    ap.add_argument('--csv', required=True, help='Per-rod CSV path (from --perrod option).')
    ap.add_argument('--scene', default=None, help='Optional scene JSON to auto-infer rod length.')
    ap.add_argument('--length', type=float, default=None, help='Rod length (if not given, inferred from --scene or defaults to 0.5).')
    ap.add_argument('--local-axis', type=str, default='0,1,0', help='Local axis treated as rod axis before rotation (default Y axis). Format: ax,ay,az')
    ap.add_argument('--frame-skip', type=int, default=20, help='Analyze every Nth frame for dispersion metric.')
    ap.add_argument('--npz', type=str, default=None, help='Optional output .npz path for arrays.')
    ap.add_argument('--plot', action='store_true', help='Generate relative velocity norm plot.')
    args = ap.parse_args()

    if args.length is None:
        if args.scene:
            length = infer_length_from_scene(args.scene)
            print(f"Inferred length={length} from scene")
        else:
            length = 0.5
            print("Length not provided; defaulting to 0.5")
    else:
        length = args.length

    try:
        axis_vals = [float(x) for x in args.local_axis.split(',') if x.strip()]
        if len(axis_vals) != 3:
            raise ValueError
        local_axis = tuple(axis_vals)
    except Exception:
        raise SystemExit("--local-axis must be three comma-separated floats, e.g. 0,1,0")

    results = analyze(args.csv, length=length, frame_skip=args.frame_skip,
                      local_axis=local_axis, npz_out=args.npz, plot=args.plot)

    print("\nSummary:")
    print(f"  Frames: {results['num_frames']} | Rods: {results['num_rods']} | Length: {results['length']}")
    print(f"  RV-norm samples: {len(results['rv_norm'])} (skip={args.frame_skip})")
    if results['rv_norm'].size > 0:
        print(f"  RV-norm min/mean/max: {results['rv_norm'].min():.6g} / {results['rv_norm'].mean():.6g} / {results['rv_norm'].max():.6g}")

if __name__ == '__main__':
    main()
