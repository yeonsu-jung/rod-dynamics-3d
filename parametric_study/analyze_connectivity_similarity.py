import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_edges(csv_path):
    """
    Load edges from CSV and group by frame.
    Returns: dict {frame_id: set of (min, max) edges}
    """
    df = pd.read_csv(csv_path)
    
    # Ensure columns exist
    if not {'frame', 'source', 'target'}.issubset(df.columns):
        raise ValueError(f"Invalid columns in {csv_path}: {df.columns}")
        
    frames = {}
    
    # Iterate efficiently
    # Group by frame
    grouped = df.groupby('frame')
    
    for frame_id, group in grouped:
        # Canonicalize edges to undirected (u, v) where u < v
        # Vectorized approach
        s = group['source'].values
        t = group['target'].values
        
        # Stack and sort along axis 1
        edges = np.sort(np.vstack((s, t)).T, axis=1)
        
        # Convert to set of tuples for fast set operations
        # Using map is faster than iterating rows loops
        edge_set = set(map(tuple, edges))
        frames[frame_id] = edge_set
        
    return frames

def compute_jaccard_evolution(frames_dict):
    """
    Compute Jaccard similarity between t and t+1.
    Returns: (frames, similarities)
    """
    sorted_frames = sorted(frames_dict.keys())
    if not sorted_frames:
        return [], []
        
    similarities = []
    # We maintain 1-to-1 mapping with the "step" or "interval". 
    # Let's say similarity at t is between t and t+1.
    plot_x = [] 
    
    for i in range(len(sorted_frames) - 1):
        f1 = sorted_frames[i]
        f2 = sorted_frames[i+1]
        
        e1 = frames_dict[f1]
        e2 = frames_dict[f2]
        
        intersection = len(e1.intersection(e2))
        union = len(e1.union(e2))
        
        jaccard = intersection / union if union > 0 else 0.0 # Define 1.0 if both empty? usually 0 or 1. If no rods, empty graph -> empty graph = 1.0?
        if union == 0: jaccard = 1.0
        
        similarities.append(jaccard)
        plot_x.append(f1) # Plot against the starting frame
        
    return plot_x, similarities

def main():
    parser = argparse.ArgumentParser(description="Analyze Connectivity Jaccard Similarity")
    parser.add_argument("input_files", nargs='+', type=Path, help="Paths to connectivity CSV files")
    parser.add_argument("--output", "-o", type=Path, default=Path("connectivity_similarity.png"), help="Output plot path")
    
    args = parser.parse_args()
    
    plt.figure(figsize=(10, 6))
    
    for p in args.input_files:
        print(f"Processing {p}...")
        try:
            frames = load_edges(p)
            x, y = compute_jaccard_evolution(frames)
            
            # Label extracts useful info? e.g. parent folder name
            label = p.parent.name
            plt.plot(x, y, label=label, marker='.', markersize=2, alpha=0.7)
            
            # Also save raw data?
            out_csv = p.parent / f"jaccard_{p.name}"
            pd.DataFrame({'frame': x, 'jaccard': y}).to_csv(out_csv, index=False)
            print(f"Saved stats to {out_csv}")
            
        except Exception as e:
            print(f"Error processing {p}: {e}")
            
    plt.xlabel("Frame")
    plt.ylabel("Jaccard Similarity (t vs t+1)")
    plt.title("Connectivity Stability Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"Saving plot to {args.output}")
    plt.savefig(args.output, dpi=150)

if __name__ == "__main__":
    main()
