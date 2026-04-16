import pandas as pd
import numpy as np
import sys
import time

def min_dist_segments(p1_a, p2_a, p1_b, p2_b):
    # Vectorized segment-segment distance
    # Only supports pairs, not cross product.
    # We will pass arrays of shape (K, 3) where K is number of pairs to check.
    
    u = p2_a - p1_a
    v = p2_b - p1_b
    w = p1_a - p1_b
    
    a = np.sum(u**2, axis=1)
    b = np.sum(u*v, axis=1)
    c = np.sum(v**2, axis=1)
    d = np.sum(u*w, axis=1)
    e = np.sum(v*w, axis=1)
    
    D = a*c - b*b
    sD = D
    tD = D
    
    SMALL_NUM = 1e-8
    
    # Compute sc and tc
    sc = np.zeros_like(D)
    tc = np.zeros_like(D)
    
    # Parallel check
    parallel_mask = D < SMALL_NUM
    not_parallel = ~parallel_mask
    
    sN = np.zeros_like(D)
    tN = np.zeros_like(D)
    
    # Non-parallel case
    if np.any(not_parallel):
        sN[not_parallel] = (b[not_parallel]*e[not_parallel] - c[not_parallel]*d[not_parallel])
        tN[not_parallel] = (a[not_parallel]*e[not_parallel] - b[not_parallel]*d[not_parallel])
        
        # Check limits for s
        mask_s_neg = sN < 0.0
        sN[mask_s_neg] = 0.0
        tN[mask_s_neg] = e[mask_s_neg]
        tD[mask_s_neg] = c[mask_s_neg]
        
        mask_s_gt = sN > sD
        sN[mask_s_gt] = sD[mask_s_gt]
        tN[mask_s_gt] = e[mask_s_gt] + b[mask_s_gt]
        tD[mask_s_gt] = c[mask_s_gt]
        
    # Parallel case
    if np.any(parallel_mask):
        sN[parallel_mask] = 0.0
        tN[parallel_mask] = e[parallel_mask]
        tD[parallel_mask] = c[parallel_mask]
        
    # Check limits for t
    mask_t_neg = tN < 0.0
    tN[mask_t_neg] = 0.0
    
    # Recompute s for t < 0
    mask_recomp_s = mask_t_neg # broad simplification, actually only need if we changed t
    if np.any(mask_t_neg):
        # We effectively clamped t to 0. Minimize distance P1(s) to P2(0) => | P1 + s*u - P3 |^2
        # s = -d/a
        # Be careful with indices
        sN[mask_t_neg] = -d[mask_t_neg]
        sD[mask_t_neg] = a[mask_t_neg] # clamp s later
        
    mask_t_gt = tN > tD
    tN[mask_t_gt] = tD[mask_t_gt]
    
    # Recompute s for t > 1
    if np.any(mask_t_gt):
        sN[mask_t_gt] = (-d[mask_t_gt] + b[mask_t_gt])
        sD[mask_t_gt] = a[mask_t_gt]

    # Final clamp for s
    mask_s_neg_final = sN < 0.0
    sN[mask_s_neg_final] = 0.0
    mask_s_gt_final = sN > sD
    sN[mask_s_gt_final] = sD[mask_s_gt_final]
    
    # Division
    sc = np.zeros_like(sN)
    tc = np.zeros_like(tN)
    
    mask_sd_nz = sD > SMALL_NUM
    sc[mask_sd_nz] = sN[mask_sd_nz] / sD[mask_sd_nz]
    
    mask_td_nz = tD > SMALL_NUM
    tc[mask_td_nz] = tN[mask_td_nz] / tD[mask_td_nz]
    
    # dP
    dP = w + (sc[:, np.newaxis] * u) - (tc[:, np.newaxis] * v)
    return np.linalg.norm(dP, axis=1)

def analyze(path):
    print(f"Checking {path}...")
    
    header_box = 1.1 # default based on previous knowledge if missing
    expected_len = None
    expected_dia = None
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("# rod_length="):
                expected_len = float(line.split('=')[1].strip())
            if line.startswith("# rod_diameter="):
                expected_dia = float(line.split('=')[1].strip())
            if line.startswith("# box_size="):
                header_box = float(line.split('=')[1].strip())
            if not line.startswith("#"):
                break
    
    print(f"Header: L={expected_len}, D={expected_dia}, Box={header_box}")

    try:
        df = pd.read_csv(path, comment='#')
    except Exception as e:
        print(f"Error: {e}")
        return

    p0 = df[['x0', 'y0', 'z0']].values
    p1 = df[['x1', 'y1', 'z1']].values
    
    # Generate all pairs
    N = len(p0)
    print(f"Analyzing {N} rods. Generating {N*(N-1)//2} pairs...")
    
    idx_triu = np.triu_indices(N, k=1)
    
    p0_a = p0[idx_triu[0]]
    p1_a = p1[idx_triu[0]]
    p0_b = p0[idx_triu[1]]
    p1_b = p1[idx_triu[1]]
    
    # --- RAW COORDINATES CHECK ---
    print("Computing distances in RAW coordinates...")
    t0 = time.time()
    dists = min_dist_segments(p0_a, p1_a, p0_b, p1_b)
    print(f"Time: {time.time()-t0:.2f}s")
    
    min_d = np.min(dists)
    print(f"Minimum Distance (Raw): {min_d:.6f}")
    if expected_dia and min_d < expected_dia:
        print(f"WARNING: Overlap in raw coordinates! Min Dist {min_d} < Diameter {expected_dia}")
        print(f"Number of overlapping pairs: {np.sum(dists < expected_dia)}")
    else:
        print("Raw coordinates look valid (no overlaps).")

    # --- WRAPPED COORDINATES CHECK (Box 1.1) ---
    box = header_box
    print(f"\nChecking effective wrapping into box size {box}...")
    
    # Center wraps to [-L/2, L/2]
    # x_wrapped = x - box * round(x / box)
    
    # We apply wrap to centers? No, to spatial segments.
    # Strictly, dist_PBC = min(|dx + k*L|)
    # For seg-seg PBC distance, it's complex. 
    # But as a lower bound, let's just wrap the CENTERS and check min distance of centers.
    # Or wrap endpoints and recheck segment distance (ignoring boundary crossing for simplicity, just checking fold-over).
    
    def wrap(x):
        return x - box * np.round(x / box)
    
    p0_w = wrap(p0)
    p1_w = wrap(p1) # Important: this breaks continuity if rod crosses boundary!
                    # But for checking 'fold over' overlaps, it's a good proxy for chaos.
    
    # Re-calculate segment lengths to see if we broke them
    l_w = np.linalg.norm(p1_w - p0_w, axis=1)
    broken = np.sum(np.abs(l_w - 1.0) > 0.1)
    print(f"Wrapping endpoints into {box} box broke {broken} rods (split across boundary).")
    
    # Just check point density or simple overlap proxy
    # Let's check center-center distance
    centers = 0.5*(p0 + p1)
    centers_w = wrap(centers)
    
    # Pairwise center dists
    c_a = centers_w[idx_triu[0]]
    c_b = centers_w[idx_triu[1]]
    
    diff = c_a - c_b
    # Nearest image convention for distance check?
    diff = diff - box * np.round(diff / box)
    d_centers = np.linalg.norm(diff, axis=1)
    
    min_c = np.min(d_centers)
    print(f"Min Center-Center Dist (Wrapped): {min_c:.6f}")
    if min_c < expected_dia:
        print(f"CRITICAL: Center-center overlap detected in wrapped box! ({min_c} < {expected_dia})")
    elif min_c < 0.1: # Heuristic
        print(f"Wrapped system is extremely dense (min center spacing {min_c:.4f}).")

if __name__ == "__main__":
    analyze(sys.argv[1])
