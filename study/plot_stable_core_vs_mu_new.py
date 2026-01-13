import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os
from pathlib import Path
import time
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from compute_topology import load_rods_from_csv, compute_linking_matrix
from find_stable_core import compute_rod_instability
# We re-implement parts of find_stable_core to be faster for N=1000

def compute_changed_triples_fast(X1, X2):
    """
    Identify changed triples (i,j,k) where v_ijk(T) != v_ijk(0).
    Uses vectorized operations to avoid O(N^3) Python loops.
    """
    N = X1.shape[0]
    changed_triples = []
    
    # We want to find i<j<k such that:
    # X1[i,j]*X1[j,k]*X1[k,i] != X2[i,j]*X2[j,k]*X2[k,i]
    
    # This is still O(N^3) but done in C via numpy/blas if we can formulate it right.
    # However, for N=1000, full tensor is 10^9 elements.
    # We can iterate over i to reduce memory usage to O(N^2).
    
    print(f"  Computing vorticity changes (N={N})...")
    start_time = time.time()
    
    iters = 0
    cnt = 0
    
    # Loop over i
    for i in range(N):
        # We need j > i and k > j
        # Construct submatrices for j, k range
        
        # Vectorized check for fixed i
        # v_ijk = X[i,j] * X[j,k] * X[k,i]
        # X is antisymmetric, so X[k,i] = -X[i,k]
        # v_ijk = - X[i,j] * X[j,k] * X[i,k]
        
        # Let's consider only j > i, k > j
        # We can form a matrix for fixed i: M_jk = v_ijk
        
        # X[i, j:] is a vector of size N-i (call it vec_i)
        # X[i, k:] is also from that vector
        # X[j:, k:] is the block
        
        # Ideally we process batches of i to vectorise better, but one i is fine.
        
        # Get row i: X[i, :]
        row_i = X1[i, :]
        row_i_2 = X2[i, :]
        
        # We need loop over j from i+1 to N-1
        for j in range(i + 1, N):
            # Val1 for all k > j
            # v1 = X1[i,j] * X1[j, k:] * X1[k, i]
            #    = X1[i,j] * X1[j, k:] * (-X1[i, k])
            
            # X1[i, k] for k > j is just row_i[j+1:]
            
            x_ij_1 = X1[i,j]
            x_ij_2 = X2[i,j]
            
            if x_ij_1 == 0 and x_ij_2 == 0:
                continue
                
            # k range: j+1 to N
            x_jk_1 = X1[j, j+1:]
            x_ik_1 = row_i[j+1:] # X1[i, k]
             
            x_jk_2 = X2[j, j+1:]
            x_ik_2 = row_i_2[j+1:]
            
            # Vorticity vectors for k > j
            # X[k, i] = -X[i, k]
            vals1 = x_ij_1 * x_jk_1 * (-x_ik_1)
            vals2 = x_ij_2 * x_jk_2 * (-x_ik_2)
            
            # Find indices where unequal
            diff_indices = np.where(vals1 != vals2)[0]
            
            if len(diff_indices) > 0:
                # k = (j+1) + index
                ks = diff_indices + (j + 1)
                for k in ks:
                    changed_triples.append((i, j, k))
                    
        if i % 100 == 0:
            print(f"    Processed i={i}/{N}")
            
    print(f"  Finished in {time.time() - start_time:.2f}s")
    return changed_triples

def find_stable_core_fast(N, changed_triples):
    """
    Optimized greedy heuristic for stable core.
    """
    print("  Computing stable core...")
    
    # 1. Compute initial instability counts
    rod_instability = np.zeros(N, dtype=int)
    
    # Map for fast updates: rod -> list of triple_indices
    # But storing list of triples for each rod is expensive (3 * num_triples)
    # If 10M triples, 30M ints, ok.
    
    triples_arr = np.array(changed_triples, dtype=np.int32)
    num_triples = len(triples_arr)
    
    if num_triples == 0:
        return list(range(N))
        
    for i in range(num_triples):
        rod_instability[triples_arr[i, 0]] += 1
        rod_instability[triples_arr[i, 1]] += 1
        rod_instability[triples_arr[i, 2]] += 1
        
    # Active rods mask
    active = np.ones(N, dtype=bool)
    
    # Active triples mask
    active_triples = np.ones(num_triples, dtype=bool)
    
    # Rod to triples adjacency (expensive to build? O(num_triples))
    # Let's build it.
    rod_adj = [[] for _ in range(N)]
    for idx, (r1, r2, r3) in enumerate(changed_triples):
        rod_adj[r1].append(idx)
        rod_adj[r2].append(idx)
        rod_adj[r3].append(idx)
        
    removed_count = 0
    
    while True:
        # Find active rod with max instability
        # We only check active rods
        # Using argmax on masked array is slow if we strictly construct it
        # Just iterate or keep a set? iterating 1000 is fast.
        
        max_score = -1
        victim = -1
        
        # fast check
        candidates = np.where(active)[0]
        if len(candidates) == 0:
            break
            
        scores = rod_instability[candidates]
        local_max_idx = np.argmax(scores)
        max_score = scores[local_max_idx]
        victim = candidates[local_max_idx]
        
        if max_score == 0:
            break
            
        # Remove victim
        active[victim] = False
        removed_count += 1
        rod_instability[victim] = 0 # Clear it so it's not picked again
        
        # Deactivate associated triples
        for t_idx in rod_adj[victim]:
            if active_triples[t_idx]:
                active_triples[t_idx] = False
                
                # Decrease score of neighbors
                r1, r2, r3 = changed_triples[t_idx]
                
                # One of them is victim. Update others.
                if r1 != victim and active[r1]:
                     rod_instability[r1] -= 1
                if r2 != victim and active[r2]:
                     rod_instability[r2] -= 1
                if r3 != victim and active[r3]:
                     rod_instability[r3] -= 1
                     
        if removed_count % 100 == 0:
            # print(f"    Removed {removed_count} rods, max_score was {max_score}")
            pass
            
    return sorted(np.where(active)[0].tolist())

def compute_segment_distance(rod_a, rod_b):
    # Same as before, optimized or inlined if needed
    p1 = rod_a[:3]
    d1 = rod_a[3:] - p1
    p2 = rod_b[:3]
    d2 = rod_b[3:] - p2
    
    SMALL_NUM = 0.00000001
    u = d1; v = d2; w = p1 - p2
    a = np.dot(u,u); b = np.dot(u,v); c = np.dot(v,v)
    d = np.dot(u,w); e = np.dot(v,w)
    D = a*c - b*b
    
    sc, sN, sD = 0.0, 0.0, D
    tc, tN, tD = 0.0, 0.0, D
    
    if D < SMALL_NUM:
        sN = 0.0; sD = 1.0; tN = e; tD = c
    else:
        sN = (b*e - c*d); tN = (a*e - b*d)
        if sN < 0.0: sN = 0.0; tN = e; tD = c
        elif sN > sD: sN = sD; tN = e + b; tD = c
            
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0: sN = 0.0
        elif -d > a: sN = sD
        else: sN = -d; sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0: sN = 0.0
        elif (-d + b) > a: sN = sD
        else: sN = (-d + b); sD = a
            
    sc = 0.0 if abs(sN) < SMALL_NUM else sN / sD
    tc = 0.0 if abs(tN) < SMALL_NUM else tN / tD
    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)

def compute_avg_stable_core_distance(rods, core_indices):
    if len(core_indices) < 2:
        return 0.0
    
    # For N=1000, calculating all pairs is 500k, which is fast enough (seconds)
    # But let's sample if too huge?
    # No, simple loop.
    
    distances = []
    core_list = list(core_indices)
    
    # Use random sample if > 500 rods to keep it instant
    if len(core_list) > 300:
        # Sample 10000 pairs
        for _ in range(10000):
            idx1 = np.random.choice(core_list)
            idx2 = np.random.choice(core_list)
            if idx1 != idx2:
                dist = compute_segment_distance(rods[idx1], rods[idx2])
                distances.append(dist)
    else:
        for i in range(len(core_list)):
            idx1 = core_list[i]
            for j in range(i + 1, len(core_list)):
                idx2 = core_list[j]
                dist = compute_segment_distance(rods[idx1], rods[idx2])
                distances.append(dist)
            
    return np.mean(distances) if distances else 0.0

def analyze_directory(dirname):
    # Determine mu from dirname
    # Format: ..._Friction0.0_...
    try:
        parts = dirname.split('_')
        mu_part = [p for p in parts if p.startswith('Friction')][0]
        mu = float(mu_part.replace('Friction', ''))
    except:
        print(f"Skipping {dirname}: cannot parse friction")
        return None
        
    csv_path = os.path.join(dirname, 'endpoints.csv')
    if not os.path.exists(csv_path):
        return None
        
    print(f"Analyzing mu={mu} from {dirname}...")
    
    # Load frame 0
    try:
        rods0, _ = load_rods_from_csv(csv_path, 0)
    except:
        print(f"  Failed to load frame 0")
        return None
        
    # Load last frame (200000)
    # Since reading the whole CSV is slow, we might want to read just the end
    # But load_rods_from_csv using pandas reads all.
    # Speedup: Read only last N lines using unix tail and parse?
    # Or just wait. The file for 200k frames * 1000 rods might be huge.
    # 200M lines. Pandas read_csv will die or take forever.
    # We MUST use a smarter loader.
    
    # Smarter loader for last frame
    print("  Loading last frame (smartly)...")
    # Identify max frame first?"
    # We assume 200000.
    
    target_frame = 200000
    
    # Check if we can just read the tail
    # 1000 rods * 1 frame = 1000 lines.
    # Let's read the last 2000 lines to be safe and parse
    
    import subprocess
    cmd = f"tail -n 2000 {csv_path}"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    tail_output = proc.communicate()[0].decode('utf-8')
    
    from io import StringIO
    df_tail = pd.read_csv(StringIO(tail_output), header=None, names=['frame','rod','x1','y1','z1','x2','y2','z2'])
    
    # Filter for max frame found in tail
    max_frame = df_tail['frame'].max()
    print(f"  Found max frame: {max_frame}")
    
    df_last = df_tail[df_tail['frame'] == max_frame]
    if len(df_last) != 1000:
        print(f"  Warning: Last frame has {len(df_last)} rods instead of 1000")
        
    rods_final = df_last[['x1', 'y1', 'z1', 'x2', 'y2', 'z2']].values
    
    # Only sort by rod id if needed. The tail might be mixed? Usually it's sorted by rod ID.
    # Ensure sorted by rod ID
    df_last = df_last.sort_values('rod')
    rods_final = df_last[['x1', 'y1', 'z1', 'x2', 'y2', 'z2']].values
    
    # Load initial frame efficiency? load_rods_from_csv reads all?
    # We should write a fast reader for frame 0 as well (head)
    
    cmd0 = f"head -n 2000 {csv_path}"
    proc0 = subprocess.Popen(cmd0, shell=True, stdout=subprocess.PIPE)
    head_output = proc0.communicate()[0].decode('utf-8')
    
    # First line is header, skip it
    df_head = pd.read_csv(StringIO(head_output))
    df_0 = df_head[df_head['frame'] == 0]
    df_0 = df_0.sort_values('rod')
    rods0 = df_0[['x1', 'y1', 'z1', 'x2', 'y2', 'z2']].values
    
    N = len(rods0)
    
    # Analysis
    X0 = compute_linking_matrix(rods0)
    X_final = compute_linking_matrix(rods_final)
    
    changed = compute_changed_triples_fast(X0, X_final)
    n_changes = len(changed)
    print(f"  Total changed triples: {n_changes}")
    
    core = find_stable_core_fast(N, changed)
    core_size = len(core)
    
    avg_dist = compute_avg_stable_core_distance(rods_final, core)
    print(f"  Stable core size: {core_size}, Avg Dist: {avg_dist:.4f}")
    
    return {
        'mu': mu,
        'core_size': core_size,
        'core_fraction': core_size / N,
        'n_changes': n_changes,
        'avg_dist': avg_dist
    }

def main():
    base_dir = "scripts/relax3rd_N1000_sweep"
    # Find all subdirectories
    dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    
    results = []
    for d in dirs:
        res = analyze_directory(d)
        if res:
            results.append(res)
            
    # Sort by mu
    results.sort(key=lambda x: x['mu'])
    
    # Extract data
    mus = [r['mu'] for r in results]
    core_sizes = [r['core_size'] for r in results]
    avg_dists = [r['avg_dist'] for r in results]
    n_changes = [r['n_changes'] for r in results]
    
    print("\nResults Summary (N=1000):")
    print("mu   | Core Size | Avg Dist")
    print("-----|-----------|---------")
    for r in results:
        print(f"{r['mu']:<4} | {r['core_size']:<9} | {r['avg_dist']:.4f}")
        
    # Plot 1: Stable Core Size
    plt.figure(figsize=(10, 6))
    plt.plot(mus, core_sizes, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel(r'Friction Coefficient $\mu$', fontsize=14)
    plt.ylabel('Stable Core Size (Number of Rods)', fontsize=14)
    plt.title('Stable Core Size vs Friction (N=1000)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('study/stable_core_vs_mu_new.png', dpi=300)
    
    # Plot 2: Avg Distance
    plt.figure(figsize=(10, 6))
    plt.plot(mus, avg_dists, '^-', linewidth=2, markersize=8, color='green')
    plt.xlabel(r'Friction Coefficient $\mu$', fontsize=14)
    plt.ylabel('Avg Pairwise Distance in Stable Core', fontsize=14)
    plt.title('Packing Density of Stable Core vs Friction (N=1000)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('study/stable_core_avg_dist_vs_mu_new.png', dpi=300)
    
    print("Plots saved.")

if __name__ == '__main__':
    main()
