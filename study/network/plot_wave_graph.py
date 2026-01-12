
"""
Plot node-edge diagram for the first accumulated contact network window.
Usage: python3 study/network/plot_wave_graph.py <sweep_folder> --width 100
"""

import argparse
import csv
import matplotlib.pyplot as plt
import networkx as nx
import sys
from pathlib import Path

def build_graph_first_window(network_csv, width):
    G = nx.Graph()
    # Ensure all expected nodes exist (0 to 199 for N=200, though we don't know N for sure here)
    # We will just add nodes as we see them, or maybe scan for max ID?
    # Better to just add edges.
    
    with open(network_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(row['frame'])
                if frame >= width:
                    # We only care about the first window [0, width)
                    # Optimization: if frames are sorted, we can break.
                    # But often they are sorted. Let's risking breaking early if frame > width + 100
                    if frame > width + 1000: 
                        break
                    continue
                
                i = int(row['rod_i'])
                j = int(row['rod_j'])
            except (ValueError, KeyError):
                continue

            if i < 0 or j < 0:
                continue
            
            G.add_edge(i, j)
            
    return G

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_folder", type=Path)
    parser.add_argument("--width", type=int, default=100, help="Window width in frames")
    args = parser.parse_args()

    if not args.sweep_folder.exists():
        sys.exit(f"Folder not found: {args.sweep_folder}")

    count = 0
    for sub in args.sweep_folder.iterdir():
        if not sub.is_dir():
            continue
        
        net_csv = sub / "network.csv"
        if not net_csv.exists():
            continue
        
        print(f"Processing {sub.name}...")
        G = build_graph_first_window(net_csv, args.width)
        
        if G.number_of_nodes() == 0:
            print("  Empty graph (no contacts).")
            continue

        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, seed=42, k=0.15) # k controls spacing
        
        # Color nodes by degree? Or just simple blue.
        # Let's do simple for now.
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color='skyblue', alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
        # nx.draw_networkx_labels(G, pos, font_size=6) # Labels might clutter
        
        plt.title(f"Contact Network (Frames 0-{args.width})\n{sub.name}")
        plt.axis('off')
        
        out_path = sub / "graph_window0.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved {out_path}")
        count += 1

    print(f"Generated {count} graph plots.")

if __name__ == "__main__":
    main()
