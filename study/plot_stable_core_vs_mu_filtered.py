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

from compute_topology import compute_linking_matrix
# Re-using optimized functions structure

def compute_parallel_mask(rods, angle_threshold_deg=5.0):
    """
    Compute NxN boolean mask where True indicates the pair (i,j) is 'parallel'.
    Parallel defined as angle < threshold OR angle > 180 - threshold.
    i.e. |dot(u_i, u_j)| > cos(threshold)
    """
    N = len(rods)
    # Vectors: rods[:, 3:6] - rods[:, 0:3]
    vecs = rods[:, 3:6] - rods[:, 0:3]
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    dirs = vecs / norms
    
    # Compute dot products all pairs
    # dots[i, j] = u_i . u_j
    dots = np.dot(dirs, dirs.T)
    
    # Threshold
    threshold_rad = np.radians(angle_threshold_deg)
    cos_thresh = np.cos(threshold_rad)
    
    # Mask: True if |dot| > cos_thresh
    mask = np.abs(dots) > cos_thresh
    
    # Diagonal is True (parallel to self), but we don't use diagonal in triples
    return mask

def compute_changed_triples_filtered(X1, X2, parallel_mask):
    """
    Identify changed triples (i,j,k) where v_ijk(T) != v_ijk(0).
    Filter out triples where any pair (i,j), (j,k), (i,k) is parallel.
    """
    N = X1.shape[0]
    changed_triples = []
    
    print(f"  Computing vorticity changes (N={N}) with filtering...")
    start_time = time.time()
    
    # Loop over i
    for i in range(N):
        row_i = X1[i, :]
        row_i_2 = X2[i, :]
        
        # Parallel checks involving i
        # parallel_mask[i, :] is boolean vector
        
        for j in range(i + 1, N):
            # Check pair (i,j)
            if parallel_mask[i, j]:
                continue
                
            x_ij_1 = X1[i,j]
            x_ij_2 = X2[i,j]
            
            # If both are zero link, maybe irrelevant, but strictly change is 0->0 (no change) or 0->1
            if x_ij_1 == 0 and x_ij_2 == 0:
                continue
                
            x_jk_1 = X1[j, j+1:]
            x_ik_1 = row_i[j+1:] 
             
            x_jk_2 = X2[j, j+1:]
            x_ik_2 = row_i_2[j+1:]
            
            vals1 = x_ij_1 * x_jk_1 * (-x_ik_1)
            vals2 = x_ij_2 * x_jk_2 * (-x_ik_2)
            
            diff_mask = (vals1 != vals2)
            diff_indices = np.where(diff_mask)[0]
            
            if len(diff_indices) > 0:
                ks = diff_indices + (j + 1)
                
                # Filter ks
                # We need to check pairs (j,k) and (i,k)
                # k is in ks
                
                # Vectorized check?
                # parallel_mask[j, ks] -> True if (j,k) parallel
                # parallel_mask[i, ks] -> True if (i,k) parallel
                
                p_jk = parallel_mask[j, ks]
                p_ik = parallel_mask[i, ks]
                
                # Keep if NEITHER is parallel
                valid = ~(p_jk | p_ik)
                
                valid_ks = ks[valid]
                
                for k in valid_ks:
                    changed_triples.append((i, j, k))
                    
        if i % 100 == 0:
            print(f"    Processed i={i}/{N}")
            
    print(f"  Finished in {time.time() - start_time:.2f}s")
    return changed_triples

def find_stable_core_fast(N, changed_triples):
    # Same standard greedy logic
    print("  Computing stable core...")
    rod_instability = np.zeros(N, dtype=int)
    triples_arr = np.array(changed_triples, dtype=np.int32)
    num_triples = len(triples_arr)
    
    if num_triples == 0:
        return list(range(N))
        
    for i in range(num_triples):
        rod_instability[triples_arr[i, 0]] += 1
        rod_instability[triples_arr[i, 1]] += 1
        rod_instability[triples_arr[i, 2]] += 1
        
    active = np.ones(N, dtype=bool)
    active_triples = np.ones(num_triples, dtype=bool)
    
    rod_adj = [[] for _ in range(N)]
    for idx, (r1, r2, r3) in enumerate(changed_triples):
        rod_adj[r1].append(idx)
        rod_adj[r2].append(idx)
        rod_adj[r3].append(idx)
        
    removed_count = 0
    while True:
        candidates = np.where(active)[0]
        if len(candidates) == 0:
            break
        scores = rod_instability[candidates]
        local_max_idx = np.argmax(scores)
        max_score = scores[local_max_idx]
        victim = candidates[local_max_idx]
        
        if max_score == 0:
            break
            
        active[victim] = False
        removed_count += 1
        rod_instability[victim] = 0
        
        for t_idx in rod_adj[victim]:
            if active_triples[t_idx]:
                active_triples[t_idx] = False
                r1, r2, r3 = changed_triples[t_idx]
                if r1 != victim and active[r1]: rod_instability[r1] -= 1
                if r2 != victim and active[r2]: rod_instability[r2] -= 1
                if r3 != victim and active[r3]: rod_instability[r3] -= 1
            
    return sorted(np.where(active)[0].tolist())

def analyze_directory(dirname):
    try:
        parts = dirname.split('_')
        mu_part = [p for p in parts if p.startswith('Friction')][0]
        mu = float(mu_part.replace('Friction', ''))
    except:
        return None
        
    csv_path = os.path.join(dirname, 'endpoints.csv')
    if not os.path.exists(csv_path):
        return None
        
    print(f"Analyzing mu={mu} from {dirname}...")
    
    # Load Smartly (borrowing from previous script ideas)
    import subprocess
    from io import StringIO
    
    # Frame 0
    cmd0 = f"head -n 2000 {csv_path}"
    proc0 = subprocess.Popen(cmd0, shell=True, stdout=subprocess.PIPE)
    head_output = proc0.communicate()[0].decode('utf-8')
    df_head = pd.read_csv(StringIO(head_output))
    df_0 = df_head[df_head['frame'] == 0].sort_values('rod')
    rods0 = df_0[['x1', 'y1', 'z1', 'x2', 'y2', 'z2']].values
    
    # Frame 200000
    cmd = f"tail -n 2000 {csv_path}"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    tail_output = proc.communicate()[0].decode('utf-8')
    df_tail = pd.read_csv(StringIO(tail_output), header=None, names=['frame','rod','x1','y1','z1','x2','y2','z2'])
    max_frame = df_tail['frame'].max()
    df_last = df_tail[df_tail['frame'] == max_frame].sort_values('rod')
    rods_final = df_last[['x1', 'y1', 'z1', 'x2', 'y2', 'z2']].values
    
    N = len(rods0)
    
    # Compute Parallel Mask (using Final frame? or Initial? or Both?)
    # "If a triple has a pair having angle < 5 degrees..."
    # Usually we should check if they are parallel in EITHER frame?
    # Or just the configuration where the ambiguity exists?
    # Let's be conservative: check ONLY Final frame configuration (current state)
    # The instability is v(T) vs v(0). If they are parallel at T, v(T) might be noise.
    # User said "rule out that triple".
    # Let's compute mask for BOTH frames and union them? No, that might kill too many.
    # Let's try: exclude if parallel in Initial OR Final.
    
    mask0 = compute_parallel_mask(rods0)
    maskF = compute_parallel_mask(rods_final)
    mask = mask0 | maskF # Conservative: if parallel at any point
    
    parallel_pairs_count = np.sum(mask) / 2 # symmetric
    print(f"  Parallel pairs (<5 deg): {parallel_pairs_count} (in either frame)")

    # Analysis
    X0 = compute_linking_matrix(rods0)
    X_final = compute_linking_matrix(rods_final)
    
    changed = compute_changed_triples_filtered(X0, X_final, mask)
    n_changes = len(changed)
    print(f"  Total changed triples (filtered): {n_changes}")
    
    core = find_stable_core_fast(N, changed)
    core_size = len(core)
    
    print(f"  Stable core size (filtered): {core_size}")
    
    return {
        'mu': mu,
        'core_size': core_size,
        'core_fraction': core_size / N,
        'n_changes': n_changes
    }

def main():
    base_dir = "scripts/relax3rd_N1000_sweep"
    dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    
    results = []
    for d in dirs:
        res = analyze_directory(d)
        if res:
            results.append(res)
            
    results.sort(key=lambda x: x['mu'])
    
    mus = [r['mu'] for r in results]
    core_sizes = [r['core_size'] for r in results]
    
    print("\nResults Summary (N=1000, Filtered < 5 deg):")
    print("mu   | Core Size (Flt) | Changed Triples")
    print("-----|-----------------|----------------")
    for r in results:
        print(f"{r['mu']:<4} | {r['core_size']:<15} | {r['n_changes']}")
        
    # Standard Results for Comparison (Hardcoded from previous run)
    # 0.0:  36
    # 0.05: 95
    # 0.1:  406
    # 0.15: 582
    # 0.2:  636
    # 0.4:  679
    # 1.0:  693
    standard_map = {0.0: 36, 0.05: 95, 0.1: 406, 0.15: 582, 0.2: 636, 0.4: 679, 1.0: 693}
    standard_sizes = [standard_map.get(m, 0) for m in mus]
    
    plt.figure(figsize=(10, 6))
    plt.plot(mus, standard_sizes, 'o--', linewidth=2, markersize=8, color='gray', label='Standard')
    plt.plot(mus, core_sizes, '*-', linewidth=2, markersize=10, color='red', label='Filtered (< 5°)')
    
    plt.xlabel(r'Friction Coefficient $\mu$', fontsize=14)
    plt.ylabel('Stable Core Size (Number of Rods)', fontsize=14)
    plt.title('Stable Core Size: Standard vs Angle Filtered', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig('study/stable_core_vs_mu_filtered.png', dpi=300)
    print("Plot saved to study/stable_core_vs_mu_filtered.png")

if __name__ == '__main__':
    main()
