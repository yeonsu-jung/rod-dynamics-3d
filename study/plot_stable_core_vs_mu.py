import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from compute_topology import load_rods_from_csv, compute_linking_matrix
from find_stable_core import compute_vorticity_tensor, find_diffs, find_stable_core

def compute_segment_distance(rod_a, rod_b):
    """
    Compute minimum distance between two line segments.
    rod_a, rod_b: [x1, y1, z1, x2, y2, z2]
    """
    p1 = rod_a[:3]
    d1 = rod_a[3:] - p1
    p2 = rod_b[:3]
    d2 = rod_b[3:] - p2
    
    # We want to minimize |(p1 + t1*d1) - (p2 + t2*d2)| for t1,t2 in [0,1]
    # This involves solving a small linear system or using a robust algorithm
    # For speed and simplicity, we can use a discretized approximation or an exact formula
    # Using exact formula implementation for segments
    
    SMALL_NUM = 0.00000001
    
    u = d1
    v = d2
    w = p1 - p2
    
    a = np.dot(u,u)
    b = np.dot(u,v)
    c = np.dot(v,v)
    d = np.dot(u,w)
    e = np.dot(v,w)
    D = a*c - b*b
    
    sc, sN, sD = 0.0, 0.0, D
    tc, tN, tD = 0.0, 0.0, D
    
    if D < SMALL_NUM:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
            
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a
            
    sc = 0.0 if abs(sN) < SMALL_NUM else sN / sD
    tc = 0.0 if abs(tN) < SMALL_NUM else tN / tD
    
    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)

def compute_avg_stable_core_distance(rods, core_indices):
    """
    Compute average pairwise distance between rods in the stable core.
    """
    if len(core_indices) < 2:
        return 0.0
        
    distances = []
    core_list = list(core_indices)
    
    # Calculate for all pairs in the core
    count = 0
    total_dist = 0.0
    
    # If core is large, random sampling might be faster, but N=200 is small enough
    for i in range(len(core_list)):
        idx1 = core_list[i]
        for j in range(i + 1, len(core_list)):
            idx2 = core_list[j]
            dist = compute_segment_distance(rods[idx1], rods[idx2])
            distances.append(dist)
            
    return np.mean(distances)

def analyze_dataset(filepath, mu):
    print(f"Analyzing mu={mu} from {filepath}...")
    
    try:
        rods0, _ = load_rods_from_csv(filepath, 0)
        rods_final, _ = load_rods_from_csv(filepath, 106)
    except Exception as e:
        print(f"Error loading data for mu={mu}: {e}")
        return None

    N = len(rods0)
    
    # Compute Linking Matrices & Vorticities
    X0 = compute_linking_matrix(rods0)
    X_final = compute_linking_matrix(rods_final)
    
    v0 = compute_vorticity_tensor(X0)
    v_final = compute_vorticity_tensor(X_final)
    
    # Find Differences
    changed = find_diffs(v0, v_final)
    n_changes = len(changed)
    
    # Find Stable Core
    core = find_stable_core(N, changed)
    core_size = len(core)
    
    # Compute avg distance within stable core in FINAL frame
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
    datasets = [
        (0.05, "study/topology_analysis_data/endpoints_formatted_n200_ar300_mu0.05.csv"),
        (0.1,  "study/topology_analysis_data/endpoints_formatted_n200_ar300_mu0.1.csv"),
        (0.15, "study/topology_analysis_data/endpoints_formatted_n200_ar300_mu0.15.csv"),
        (0.2,  "study/topology_analysis_data/endpoints_formatted_n200_ar300_mu0.2.csv"),
        (0.4,  "study/topology_analysis_data/endpoints_formatted_n200_ar300_mu0.4.csv"),
        (1.0,  "study/topology_analysis_data/endpoints_formatted_n200_ar300_mu1.0.csv")
    ]
    
    results = []
    
    for mu, filepath in datasets:
        res = analyze_dataset(filepath, mu)
        if res:
            results.append(res)
            
    # Extract data for plotting
    mus = [r['mu'] for r in results]
    core_sizes = [r['core_size'] for r in results]
    avg_dists = [r['avg_dist'] for r in results]
    
    print("\nResults Summary:")
    print("mu   | Core Size | Avg Dist")
    print("-----|-----------|---------")
    for r in results:
        print(f"{r['mu']:<4} | {r['core_size']:<9} | {r['avg_dist']:.4f}")

    # --- Plot: Avg Distance vs Mu ---
    plt.figure(figsize=(10, 6))
    plt.plot(mus, avg_dists, '^-', linewidth=2, markersize=8, color='green')
    
    plt.xlabel(r'Friction Coefficient $\mu$', fontsize=14)
    plt.ylabel('Avg Pairwise Distance in Stable Core', fontsize=14)
    plt.title('Packing Density of Stable Core vs Friction', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save
    plt.savefig('study/stable_core_avg_dist_vs_mu.png', dpi=300)
    print("Saved study/stable_core_avg_dist_vs_mu.png")
    
    # --- Re-save previous plots too ---
    plt.figure(figsize=(10, 6))
    plt.plot(mus, core_sizes, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel(r'Friction Coefficient $\mu$', fontsize=14)
    plt.ylabel('Stable Core Size (Number of Rods)', fontsize=14)
    plt.title('Stable Core Size vs Friction', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 205)
    plt.savefig('study/stable_core_vs_mu.png', dpi=300)

if __name__ == '__main__':
    main()
