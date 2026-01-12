#!/usr/bin/env python3
"""export_endpoints.py

Standalone script to export endpoints_formatted.csv from MuJoCo simulation runs.
Processes xipos_over_time.txt and xmat_over_time.txt to generate rod endpoints.

Usage:
    python3 parametric_study/export_endpoints.py /path/to/run/directory
    python3 parametric_study/export_endpoints.py /path/to/parent/directory --recursive
"""

import argparse
import re
import numpy as np
from pathlib import Path
from typing import Optional

# Match folder name for AR: ..._AR123_...
AR_RE = re.compile(r"_AR(\d+)")


def extract_ar_from_path(run_dir: Path) -> int:
    """Extract aspect ratio from directory name."""
    m = AR_RE.search(run_dir.name)
    if m:
        return int(m.group(1))
    # Try parent directories
    for parent in run_dir.parents:
        m = AR_RE.search(parent.name)
        if m:
            return int(m.group(1))
    return 0


def export_endpoints(run_dir: Path, rod_length: float = 1.0, force: bool = False) -> bool:
    """
    Export endpoints_formatted.csv from xipos and xmat trajectory files.
    
    Args:
        run_dir: Directory containing xipos_over_time.txt and xmat_over_time.txt
        rod_length: Length of each rod (default: 1.0)
        force: Overwrite existing endpoints_formatted.csv if True
        
    Returns:
        True if successful, False otherwise
    """
    xipos_path = run_dir / "xipos_over_time.txt"
    xmat_path = run_dir / "xmat_over_time.txt"
    x_relaxed_path = run_dir / "x_relaxed.txt"
    out_path = run_dir / "endpoints_formatted.csv"
    
    # Check if output already exists
    if out_path.exists() and not force:
        print(f"Skipping {run_dir.name}: endpoints_formatted.csv already exists")
        return True
    
    # Check for input files
    if not xipos_path.exists() or not xmat_path.exists():
        print(f"Skipping {run_dir.name}: Missing xipos/xmat files")
        return False
    
    if not x_relaxed_path.exists():
        print(f"Skipping {run_dir.name}: Missing x_relaxed.txt (needed for original orientations)")
        return False
    
    try:
        # Load original orientations from x_relaxed.txt (endpoints format)
        x_relaxed = np.loadtxt(x_relaxed_path, delimiter=' ')
        if x_relaxed.ndim == 1:
            x_relaxed = x_relaxed[None, :]
        
        # Convert from endpoints to centroid-orientation format
        # x_relaxed has shape (N, 6): [x1, y1, z1, x2, y2, z2]
        r1_orig = x_relaxed[:, 0:3]
        r2_orig = x_relaxed[:, 3:6]
        original_orientations = (r2_orig - r1_orig) / np.linalg.norm(r2_orig - r1_orig, axis=1, keepdims=True)
        
        # Load trajectory data
        X_flat = np.loadtxt(xipos_path, delimiter=',')
        M_flat = np.loadtxt(xmat_path, delimiter=',')
        
        # Ensure 2D arrays
        if X_flat.ndim == 1:
            X_flat = X_flat[None, :]
        if M_flat.ndim == 1:
            M_flat = M_flat[None, :]
        
        n_steps = X_flat.shape[0]
        
        # Infer number of bodies
        n_cols_x = X_flat.shape[1]
        n_cols_m = M_flat.shape[1]
        n_bodies_x = n_cols_x // 3
        n_bodies_m = n_cols_m // 9
        
        # Consistency check
        if n_bodies_x != n_bodies_m:
            print(f"Error in {run_dir.name}: Body count mismatch X({n_bodies_x}) vs M({n_bodies_m})")
            return False
        
        # Slice off world body (first body) if present
        if n_bodies_x > 1:
            X_sliced = X_flat[:, 3:]
            M_sliced = M_flat[:, 9:]
            n_rods = n_bodies_x - 1
        else:
            X_sliced = X_flat
            M_sliced = M_flat
            n_rods = n_bodies_x
        
        # Check consistency with x_relaxed
        if n_rods != original_orientations.shape[0]:
            print(f"Error in {run_dir.name}: Rod count mismatch. Trajectory has {n_rods} rods, x_relaxed has {original_orientations.shape[0]}")
            return False
        
        # Reshape to (T, N, 3) and (T, N, 3, 3)
        C = X_sliced.reshape(n_steps, n_rods, 3)
        R = M_sliced.reshape(n_steps, n_rods, 3, 3)
        
        # Compute rod endpoints using original orientations
        # This is the KEY difference: apply rotation matrix to original orientations
        # cylinder_axes = R @ original_orientations
        cylinder_axes = np.einsum('tnij,nj->tni', R, original_orientations)
        
        h = rod_length / 2.0
        P1 = C - h * cylinder_axes
        P2 = C + h * cylinder_axes
        
        # Extract AR from directory name
        ar = extract_ar_from_path(run_dir)
        rod_radius = 1.0 / ar if ar > 0 else 0.005
        
        print(f"Exporting {run_dir.name}: {n_steps} steps, {n_rods} rods, AR={ar}")
        
        # Write output
        with open(out_path, 'w') as f:
            # Write metadata comments
            f.write(f"#rod_radius={rod_radius}\n")
            f.write(f"#rod_length={rod_length}\n")
            # Write header
            f.write("frame,id,x1,y1,z1,x2,y2,z2\n")
            
            # Write data
            for t in range(n_steps):
                for i in range(n_rods):
                    p1 = P1[t, i]
                    p2 = P2[t, i]
                    line = f"{t},{i},{p1[0]:.6f},{p1[1]:.6f},{p1[2]:.6f},{p2[0]:.6f},{p2[1]:.6f},{p2[2]:.6f}"
                    f.write(line + "\n")
        
        return True
        
    except Exception as e:
        print(f"Failed to export {run_dir.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Export endpoints_formatted.csv from MuJoCo simulation runs"
    )
    ap.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing simulation run(s)"
    )
    ap.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process all subdirectories containing xipos_over_time.txt"
    )
    ap.add_argument(
        "--rod-length",
        type=float,
        default=1.0,
        help="Rod length (default: 1.0)"
    )
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute and overwrite existing endpoints_formatted.csv files"
    )
    args = ap.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Directory not found: {args.input_dir}")
        return
    
    if args.recursive:
        # Find all directories containing xipos_over_time.txt
        candidates = [p.parent for p in args.input_dir.rglob("xipos_over_time.txt")]
        print(f"Found {len(candidates)} run directories")
        
        success_count = 0
        for run_dir in candidates:
            if export_endpoints(run_dir, rod_length=args.rod_length, force=args.recompute):
                success_count += 1
        
        print(f"\nSuccessfully exported {success_count}/{len(candidates)} runs")
    else:
        # Process single directory
        if export_endpoints(args.input_dir, rod_length=args.rod_length, force=args.recompute):
            print("Export successful")
        else:
            print("Export failed")


if __name__ == "__main__":
    main()
