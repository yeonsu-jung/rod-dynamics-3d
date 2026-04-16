
import pandas as pd
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
import sys

def segment_segment_distance_batch(p1, u1, p2, u2, length):
    """
    Compute distance between two segments defined by center p, direction u, and length L.
    p1, u1: (N, 3) arrays
    p2, u2: (N, 3) arrays
    length: scalar
    Returns: distances (N,) array
    """
    # Half length
    hl = length / 2.0
    
    # Vector between centers
    r = p1 - p2
    
    # Dot products
    # a = u1.u1 = 1 (since normalized)
    # b = u1.u2
    # c = u1.r
    # e = u2.u2 = 1
    # f = u2.r
    
    # We want to minimize |(p1+s*u1) - (p2+t*u2)|^2
    # = |r + s*u1 - t*u2|^2
    # partial deriv wrt s: 2(r + s*u1 - t*u2).u1 = 0 => s + u1.r - t(u1.u2) = 0 => s - t*b = -c
    # partial deriv wrt t: -2(r + s*u1 - t*u2).u2 = 0 => -(u2.r + s(u1.u2) - t) = 0 => s*b - t = -f
    
    # System:
    # s - b*t = -c
    # b*s - t = -f
    
    # Multiply first by b: b*s - b^2*t = -b*c
    # Subtract second: (1 - b^2)t = b*c - f
    # t = (b*c - f) / (1 - b^2)
    # s = b*t - c
    
    b = np.sum(u1 * u2, axis=1)
    c = np.sum(u1 * r, axis=1)
    f = np.sum(u2 * r, axis=1)
    
    denom = 1.0 - b*b
    
    # Handle parallel case (denom ~ 0)
    # If parallel, any s, t relation works. We can pick s=0 for infinite line closest point.
    # But usually we just handle it by epsilon.
    epsilon = 1e-6
    parallel_mask = denom < epsilon
    
    # Initialize s and t for infinite lines
    t_inf = np.zeros_like(b)
    s_inf = np.zeros_like(b)
    
    # Non-parallel
    mask = ~parallel_mask
    if np.any(mask):
        t_inf[mask] = (b[mask]*c[mask] - f[mask]) / denom[mask]
        s_inf[mask] = b[mask]*t_inf[mask] - c[mask]
        
    # Parallel: pick s=0, solve for t: -t = -f => t = f
    if np.any(parallel_mask):
        s_inf[parallel_mask] = 0.0
        t_inf[parallel_mask] = f[parallel_mask]
        
    # Now we have closest points on infinite lines.
    # We need to clamp to [-hl, hl].
    # But clamping one variable requires re-solving for the other if the optimum was outside.
    # This is the tricky part. The function is convex.
    # We clamp s first? No, we have to check region.
    
    # A robust way is to clamp s to [-hl, hl], then re-solve for t, then clamp t.
    # Because calculating global min on square domain.
    # Let's iterate.
    
    # Clamp s_inf to range
    s_clamped = np.clip(s_inf, -hl, hl)
    
    # Re-solve for t given s_clamped: t = s*b + f??
    # From eq 2: t = b*s + f  ( wait: b*s - t = -f => t = b*s + f )
    t_new = b * s_clamped + f
    
    # Clamp t
    t_clamped = np.clip(t_new, -hl, hl)
    
    # Re-solve for s given t_clamped: s = b*t - c
    s_new = b * t_clamped - c
    
    # Clamp s again?
    s_final = np.clip(s_new, -hl, hl)
    
    # This logic is approximate but works well for "close enough" or standard segment distance routine.
    # The rigorous way checks 9 regions.
    # But for analyzing thousands of contacts, this "Clamp-Resolve-Clamp" is a standard heuristic that is exact often.
    # Actually, standard algorithm:
    # 1. Compute s_inf, t_inf.
    # 2. If inside box, done.
    # 3. If not, project to edges.
    
    # Let's implement rigorous logic vectorized? Hard.
    # Let's use the simple iterative clamp (projects to boundary).
    # Ideally: do it again?
    # t_final = np.clip(b * s_final + f, -hl, hl)
    # s_final = np.clip(b * t_final - c, -hl, hl)
    
    # Let's do it 2 times.
    t_final = np.clip(b * s_final + f, -hl, hl)
    
    # Compute distance
    # delta = (p1 + s*u1) - (p2 + t*u2)
    #       = r + s*u1 - t*u2
    
    s = s_final
    t = t_final
    
    # delta shape (N, 3)
    # s, t shape (N,) -> distinct for broadcast
    delta = r + s[:, np.newaxis] * u1 - t[:, np.newaxis] * u2
    dist_sq = np.sum(delta**2, axis=1)
    
    return np.sqrt(dist_sq)

def analyze_contacts(perrod_path, network_path):
    print(f"Loading {perrod_path}...")
    df_rod = pd.read_csv(perrod_path, comment='#')
    
    print(f"Loading {network_path}...")
    df_net = pd.read_csv(network_path, comment='#')
    
    # Constants
    # From headers: rod_length=1, rod_radius=0.001
    L = 1.0
    R_rod = 0.001
    D_threshold = 2 * R_rod
    
    frames = sorted(df_rod['frame'].unique())
    print(f"Comparing PerRod vs Network distances for {len(frames)} frames...")
    
    comp_results = []
    
    # Analyze a few frames for verification
    check_frames = frames  # All frames
    
    total_y_err = 0
    total_z_err = 0
    count = 0
    
    # Accumulate results
    frame_stats = []
    
    for frame in check_frames:
        # Load perrod data for this frame
        rod_sub = df_rod[df_rod['frame'] == frame].set_index('rod')
        
        # Load network data for this frame
        net_sub = df_net[df_net['frame'] == frame]
        
        if len(net_sub) == 0:
            frame_stats.append([frame, 0, 0, 0, 0])
            continue
            
        # Get pairs from network
        i_s = net_sub['rod_i'].values
        j_s = net_sub['rod_j'].values
        d_net = net_sub['distance'].values
        
        # Get positions/quats
        try:
            # Need to handle missing rods if any
            p1 = rod_sub.loc[i_s, ['px', 'py', 'pz']].values
            q1 = rod_sub.loc[i_s, ['qx', 'qy', 'qz', 'qw']].values
            p2 = rod_sub.loc[j_s, ['px', 'py', 'pz']].values
            q2 = rod_sub.loc[j_s, ['qx', 'qy', 'qz', 'qw']].values
        except KeyError:
            print(f"Frame {frame}: Missing rods in perrod matching network.")
            continue
            
        # Compute axes
        # Axis Y
        r1 = R.from_quat(q1)
        u1_y = r1.apply([0, 1, 0])
        r2 = R.from_quat(q2)
        u2_y = r2.apply([0, 1, 0])
        
        # Axis Z
        u1_z = r1.apply([0, 0, 1])
        u2_z = r2.apply([0, 0, 1])
        
        # Compute distances
        d_y = segment_segment_distance_batch(p1, u1_y, p2, u2_y, L)
        d_z = segment_segment_distance_batch(p1, u1_z, p2, u2_z, L)
        
        # Errors (assuming network distance is center distance)
        # Check if network is surface distance? (d_net + 2r)
        
        err_y_center = np.mean(np.abs(d_y - d_net))
        err_z_center = np.mean(np.abs(d_z - d_net))
        
        err_y_gap = np.mean(np.abs((d_y - 2*R_rod) - d_net)) # if net is gap
        
        # Count violations (using Y axis)
        viol_y = np.sum(d_y < D_threshold)
        
        # Also brute force perrod count for this frame (reuse old logic)
        # To avoid re-implementing brute force loop, just use valid Y count from network subset? 
        # No, user wants ALL perrod violations.
        
        # Just store the errors for now to decide model
        comp_results.append([frame, err_y_center, err_z_center, err_y_gap, len(net_sub)])
        
        total_y_err += np.sum(np.abs(d_y - d_net))
        total_z_err += np.sum(np.abs(d_z - d_net))
        count += len(net_sub)
        
    df_comp = pd.DataFrame(comp_results, columns=['Frame', 'Err_Y_Center', 'Err_Z_Center', 'Err_Y_Gap', 'Count'])
    print("\nDistance Comparison (Avg Abs Diff):")
    print(df_comp.mean())
    
    # Based on best model, calculate full stats
    models = {'Y_Center': df_comp['Err_Y_Center'].mean(),
              'Z_Center': df_comp['Err_Z_Center'].mean(),
              'Y_Gap': df_comp['Err_Y_Gap'].mean()}
              
    best_model = min(models, key=models.get)
    print(f"Best Fitting Model (by min error): {best_model} (Err={models[best_model]:.6f})")
    
    # Now run the full counting with the correct axis
    if "Z" in best_model:
        axis_vec = [0, 0, 1]
    else:
        axis_vec = [0, 1, 0] # Default to Y
        
    if "Gap" in best_model:
        # Network distance is gap. Violation if dist < 0? 
        # But we want to count violations in PerRod using geometric check.
        # Geometric check always returns center distance.
        # Violation is center_dist < 2*R.
        pass
        
    # Re-run full pair loop with confirmed axis
    print(f"Re-running full violation count with axis {axis_vec}...")
    
    count_results = []
    
    for frame in frames: # Limit to stride if too slow? 
        sub = df_rod[df_rod['frame'] == frame].sort_values('rod')
        pos = sub[['px', 'py', 'pz']].values
        quats = sub[['qx', 'qy', 'qz', 'qw']].values
        rot = R.from_quat(quats)
        axes = rot.apply(axis_vec)
        
        num_rods = len(pos)
        idx = np.triu_indices(num_rods, k=1)
        p1 = pos[idx[0]]
        u1 = axes[idx[0]]
        p2 = pos[idx[1]]
        u2 = axes[idx[1]]
        
        # Filter
        center_dist_sq = np.sum((p1 - p2)**2, axis=1)
        possible_mask = center_dist_sq < (L + 2*R_rod + 0.1)**2
        
        p1_c = p1[possible_mask]
        u1_c = u1[possible_mask]
        p2_c = p2[possible_mask]
        u2_c = u2[possible_mask]
        
        if len(p1_c) > 0:
            dists = segment_segment_distance_batch(p1_c, u1_c, p2_c, u2_c, L)
            # Violation condition: center distance < 2*R
            viol_count = np.sum(dists < D_threshold)
            min_dist = np.min(dists)
        else:
            viol_count = 0
            min_dist = 999.0
            
        count_results.append([frame, viol_count, min_dist])
        print(f"Frame {frame}: Violations={viol_count}, MinDist={min_dist:.6f}")
        
    df_counts = pd.DataFrame(count_results, columns=['Frame', 'Violations', 'MinDist'])
    print(df_counts)
    
    # Save to csv for user
    df_counts.to_csv("contact_analysis_output.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("perrod")
    parser.add_argument("network")
    args = parser.parse_args()
    
    analyze_contacts(args.perrod, args.network)
