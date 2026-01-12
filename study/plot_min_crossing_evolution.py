import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm

# ==============================================================================
# 1. METRIC FUNCTIONS (Adapted from study/analyze_general_dataset.py)
# ==============================================================================

def calculate_min_crossing_number(rod_index, dta):
    """
    Calculates the minimum crossing number (intersection depth) for a specific rod.
    dta: (N, 6) array where each row is [x1, y1, z1, x2, y2, z2]
    """
    p_start = dta[rod_index, :3]
    p_end = dta[rod_index, 3:]
    picked_axis = p_end - p_start
    norm = np.linalg.norm(picked_axis)
    if norm < 1e-6: return 0
    picked_axis = picked_axis / norm

    # Construct basis orthogonal to the rod axis
    if np.abs(picked_axis[0]) < 0.9: ref_vec = np.array([1, 0, 0])
    else: ref_vec = np.array([0, 1, 0])
    u_vec = np.cross(picked_axis, ref_vec); u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(picked_axis, u_vec); v_vec /= np.linalg.norm(v_vec)

    # Relative positions of other rods
    p_s = dta[:, :3] - p_start
    p_e = dta[:, 3:] - p_start
    
    # Exclude the rod itself
    mask = np.ones(dta.shape[0], dtype=bool)
    mask[rod_index] = False
    p_s = p_s[mask]; p_e = p_e[mask]

    # Project onto the orthogonal plane (u, v)
    u_s = np.dot(p_s, u_vec); v_s = np.dot(p_s, v_vec)
    u_e = np.dot(p_e, u_vec); v_e = np.dot(p_e, v_vec)

    # Calculate angles in the orthogonal plane
    ang1 = np.arctan2(v_s, u_s)
    ang2 = np.arctan2(v_e, u_e)
    
    intervals = []
    for i in range(len(ang1)):
        a1, a2 = ang1[i], ang2[i]
        
        # We need the shorter arc between a1 and a2
        # BUT, wait, the crossing number logic relies on the "shadow" of the other rods.
        # The shadow is defined by the angular interval covered by the rod.
        # However, naive arctan2 diff might take the wrong way around.
        # Let's ensure we take the interval corresponding to the rod segment.
        # The rod segment is a straight line in 3D. In projection, it's a line segment.
        # The angle logic assumes the projection doesn't cross the origin (which means collision).
        # Assuming no exact Intersections (collisions), the rod covers an angular interval < pi.
        
        diff = a2 - a1
        # Normalize diff to [-pi, pi]
        while diff > np.pi: diff -= 2*np.pi
        while diff < -np.pi: diff += 2*np.pi
        
        # If we assume the rod doesn't go "the long way" around the viewer (which implies it's "behind" us if we were at the origin, but we are looking from the axis outward)...
        # Actually, since we project onto the plane orthogonal to the axis, a rod segment maps to a line segment in the 2D plane.
        # The angular interval is simply the angle range subtended by this line segment from the origin.
        # Since the line segment does not pass through the origin (no collision), the angle difference must be in (-pi, pi).
        
        if diff < 0:
            a1, a2 = a2, a1 # Swap so we go from a1 to a2 in positive direction? 
            # No, let's stick to standard intervals [start, end]
        
        # Let's use the logic from analyze_general_dataset.py directly to be safe
        # Logic from reference:
        # if a1 > a2: a1, a2 = a2, a1
        # diff = a2 - a1
        # if diff < np.pi: intervals.append((a1, a2))
        # else: intervals.append((a2, np.pi)); intervals.append((-np.pi, a1))
        
        # Re-implementing EXACT reference logic:
        a_start, a_end = ang1[i], ang2[i]
        if a_start > a_end: a_start, a_end = a_end, a_start
        d = a_end - a_start
        if d < np.pi:
            intervals.append((a_start, a_end))
        else:
            intervals.append((a_end, np.pi))
            intervals.append((-np.pi, a_start))
            
    if not intervals: return 0

    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda x: x[0])
    
    min_depth = float('inf')
    current_depth = 0
    last_angle = -np.pi
    
    for angle, change in events:
        if angle > last_angle + 1e-9:
            min_depth = min(min_depth, current_depth)
        current_depth += change
        last_angle = angle
    if np.pi > last_angle + 1e-9:
        min_depth = min(min_depth, current_depth)
        
    if min_depth == float('inf'): return 0
    return int(min_depth)

# ==============================================================================
# 2. MAIN PROCESSING
# ==============================================================================

def main():
    csv_path = "/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/relax3rd_second_complete_run_analysis/relax3rd_N100_sweep/20260101-135327_180_271_742_AR1000_Friction0.1_Kick0.1/endpoints.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Input format: frame, rod, x1, y1, z1, x2, y2, z2
    frames = sorted(df['frame'].unique())
    print(f"Found {len(frames)} frames.")
    
    results = []
    
    for f in tqdm(frames, desc="Processing frames"):
        frame_df = df[df['frame'] == f]
        
        # Construct (N, 6) array
        # Sort by rod index to be safe/consistent
        frame_df = frame_df.sort_values('rod')
        dta = frame_df[['x1', 'y1', 'z1', 'x2', 'y2', 'z2']].values
        
        N = dta.shape[0]
        
        # Calculate min crossing number for this packing
        # This is defined as min(crossing_number(rod_i)) over all rods i
        
        min_vals = []
        # Optimization: We can parallelize this if needed, but for 200 frames * 1000 rods it might take a bit.
        # Let's assess: 200 * 1000 = 200,000 checks. 
        # Python loop might be slow.
        # But let's try basic loop first.
        
        packing_min_crossing = float('inf')
        
        for i in range(N):
            mc = calculate_min_crossing_number(i, dta)
            if mc < packing_min_crossing:
                packing_min_crossing = mc
                
            # Optional optimization: if packing_min_crossing hits 0, we can stop evaluating this frame?
            # Ideally yes, because min cannot be < 0.
            if packing_min_crossing == 0:
                break
                
        results.append({
            'frame': f,
            'min_crossing_number': packing_min_crossing
        })

    # Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv("min_crossing_evolution.csv", index=False)
    print("Saved min_crossing_evolution.csv")
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(res_df['frame'], res_df['min_crossing_number'], marker='o', linestyle='-')
    plt.title("Evolution of Minimum Crossing Number")
    plt.xlabel("Frame")
    plt.ylabel("Min Crossing Number")
    plt.grid(True)
    plt.savefig("min_crossing_evolution.png")
    print("Saved min_crossing_evolution.png")

if __name__ == "__main__":
    main()
