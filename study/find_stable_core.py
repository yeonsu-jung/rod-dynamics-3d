#!/usr/bin/env python3
"""
Identify unstable rods and find the maximum stable core of rods 
that preserve their relative topology over time.
"""

import numpy as np
import pandas as pd
import sys
import argparse
from typing import List, Set, Tuple, Dict
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from compute_topology import (
    load_rods_from_csv,
    compute_linking_matrix
)


def compute_vorticity_tensor(X: np.ndarray) -> Dict[Tuple[int, int, int], int]:
    """
    Compute vorticity v_ijk for all i < j < k.
    Returns a dictionary mapping (i, j, k) -> v_ijk.
    """
    N = len(X)
    vorticities = {}
    
    # Iterate over all strictly increasing triples
    # For N=200, this is ~1.3 million entries
    for i in range(N):
        for j in range(i + 1, N):
            # Precompute partial product to save time
            x_ij = X[i, j]
            if x_ij == 0:
                continue
                
            for k in range(j + 1, N):
                # v_ijk = x_ij * x_jk * x_ki
                # Note: x_ki = -x_ik
                
                v = x_ij * X[j, k] * X[k, i]
                if v != 0:
                    vorticities[(i, j, k)] = v
                    
    return vorticities


def find_diffs(v1: Dict[Tuple[int, int, int], int], v2: Dict[Tuple[int, int, int], int]) -> Set[Tuple[int, int, int]]:
    """
    Identify triples where vorticity changed.
    """
    changed_triples = set()
    
    # Check all keys in v1
    for triple, val1 in v1.items():
        val2 = v2.get(triple, 0)
        if val1 != val2:
            changed_triples.add(triple)
            
    # Check keys in v2 not in v1
    for triple, val2 in v2.items():
        if triple not in v1:
            # val1 is 0
            if val2 != 0:
                changed_triples.add(triple)
                
    return changed_triples


def compute_rod_instability(N: int, changed_triples: Set[Tuple[int, int, int]]) -> np.ndarray:
    """
    Compute instability score for each rod.
    Score = number of changed triples the rod participates in.
    """
    scores = np.zeros(N, dtype=int)
    for i, j, k in changed_triples:
        scores[i] += 1
        scores[j] += 1
        scores[k] += 1
    return scores


def find_stable_core(N: int, changed_triples: Set[Tuple[int, int, int]]) -> List[int]:
    """
    Find maximum stable cluster using reverse greedy heuristic.
    """
    current_rods = set(range(N))
    current_triples = changed_triples.copy()
    
    # Map rod -> set of triples it is involved in
    rod_to_triples = {r: set() for r in range(N)}
    for t in changed_triples:
        i, j, k = t
        rod_to_triples[i].add(t)
        rod_to_triples[j].add(t)
        rod_to_triples[k].add(t)
        
    while current_triples:
        # Find rod with max instability in the CURRENT set
        # (i.e., involved in most REMAINING changed triples)
        max_instability = -1
        worst_rod = -1
        
        for r in current_rods:
            instability = len(rod_to_triples[r])
            if instability > max_instability:
                max_instability = instability
                worst_rod = r
        
        if max_instability == 0:
            break
            
        # Remove worst rod
        current_rods.remove(worst_rod)
        
        # Remove associated triples from tracking
        triples_to_remove = rod_to_triples[worst_rod].copy()
        
        for t in triples_to_remove:
            i, j, k = t
            current_triples.remove(t)
            
            # Update other rods' counts
            # We don't need to update the set of triples for the removed rod
            # But we must remove this triple from the other two rods' sets
            if i != worst_rod and i in rod_to_triples:
                rod_to_triples[i].discard(t)
            if j != worst_rod and j in rod_to_triples:
                rod_to_triples[j].discard(t)
            if k != worst_rod and k in rod_to_triples:
                rod_to_triples[k].discard(t)
                
        # print(f"Removed rod {worst_rod}, remaining triples: {len(current_triples)}")

    return sorted(list(current_rods))


def main():
    parser = argparse.ArgumentParser(description='Analyze topological stability of rods')
    parser.add_argument('input', type=str, help='Input CSV file')
    parser.add_argument('--frame1', type=int, default=0, help='Start frame')
    parser.add_argument('--frame2', type=int, default=106, help='End frame')
    
    args = parser.parse_args()
    
    print(f"Analyzing {args.input}")
    print(f"Comparing Frame {args.frame1} vs Frame {args.frame2}")
    
    # Load data
    rods1, _ = load_rods_from_csv(args.input, args.frame1)
    rods2, _ = load_rods_from_csv(args.input, args.frame2)
    
    N = len(rods1)
    
    # Compute Linking Matrices
    print("Computing linking matrices...")
    X1 = compute_linking_matrix(rods1)
    X2 = compute_linking_matrix(rods2)
    
    # Compute Vorticities
    print("Computing vorticity tensors...")
    v1 = compute_vorticity_tensor(X1)
    v2 = compute_vorticity_tensor(X2)
    
    # Find Differences
    print("Identifying changed triples...")
    changed = find_diffs(v1, v2)
    print(f"Total changed triples: {len(changed)}")
    
    if len(changed) == 0:
        print("Topology is perfectly preserved!")
        return

    # metrics
    scores = compute_rod_instability(N, changed)
    
    print("\nTop 10 Most Unstable Rods:")
    print("Rod ID | Involved Flips")
    print("-------|---------------")
    indices = np.argsort(scores)[::-1]
    for i in range(10):
        idx = indices[i]
        print(f"{idx:6d} | {scores[idx]}")
        
    # Stable Core
    print("\nComputing Stable Core...")
    core = find_stable_core(N, changed)
    print(f"Stable Core Size: {len(core)} rods ({len(core)/N*100:.1f}%)")
    print(f"Stable Rod IDs: {core}")
    
    # Save unstable rods for visualization later?
    # For now just print

if __name__ == '__main__':
    main()
