
"""
Accumulate network contact data over square-wave windows (e.g., 100 frames every 1000) 
and compute aggregate statistics.

Usage: python3 study/network/analyze_wave.py <sweep_folder> --period 1000 --width 100
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Try to reuse contact_analysis logic if possible, or reimplement lightweight version 
# since we need accumulation.
# We'll use a graph implementation for degree calculation.

class ContactGraph:
    def __init__(self):
        # edge -> weight
        self.edges = defaultdict(int)
        self.nodes = set()

    def add_contact(self, u, v):
        if u > v:
            u, v = v, u
        self.edges[(u, v)] += 1
        self.nodes.add(u)
        self.nodes.add(v)

    def degrees(self):
        deg = defaultdict(int)
        for (u, v), w in self.edges.items():
            if w > 0:
                deg[u] += 1
                deg[v] += 1
        return deg
    
    def edge_count(self):
        return len(self.edges)
    
    def node_count(self):
        return len(self.nodes)

def parse_ar_from_name(name):
    import re
    m = re.search(r"AR(\d+)", name)
    if m:
        return int(m.group(1))
    return None

def analyze_file(network_csv, period, width, rod_count):
    # We want to group frames into windows: [0, width), [period, period+width), etc.
    # For each window, accumulate contacts.
    
    windows = {} # start_frame -> ContactGraph

    with open(network_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame = int(row['frame'])
                i = int(row['rod_i'])
                j = int(row['rod_j'])
            except (ValueError, KeyError):
                continue

            if i < 0 or j < 0:
                continue
            
            # Determine window start
            # Window index = floor(frame / period)
            # Window start = index * period
            # Check if within width
            
            w_idx = frame // period
            w_start = w_idx * period
            w_offset = frame % period
            
            if w_offset >= width:
                # Outside the "ON" phase
                continue
            
            if w_start not in windows:
                windows[w_start] = ContactGraph()
            
            windows[w_start].add_contact(i, j)

    # Compute stats for each window
    results = []
    for start in sorted(windows.keys()):
        G = windows[start]
        degs = G.degrees()
        avg_deg = sum(degs.values()) / rod_count if rod_count > 0 else 0
        results.append({
            'window_start': start,
            'edges': G.edge_count(),
            'active_nodes': G.node_count(),
            'avg_degree': avg_deg
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_folder", type=Path)
    parser.add_argument("--period", type=int, default=1000)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--rods", type=int, default=200, help="Total rods for normalization")
    args = parser.parse_args()

    if not args.sweep_folder.exists():
        sys.exit(f"Folder not found: {args.sweep_folder}")

    # Prepare output CSV
    out_csv = args.sweep_folder / "wave_network_stats.csv"
    
    all_rows = []

    for sub in args.sweep_folder.iterdir():
        if not sub.is_dir():
            continue
        
        net_csv = sub / "network.csv"
        if not net_csv.exists():
            continue
        
        ar = parse_ar_from_name(sub.name)
        label = sub.name
        
        print(f"Analyzing {label}...")
        stats = analyze_file(net_csv, args.period, args.width, args.rods)
        
        for s in stats:
            s['ar'] = ar if ar is not None else -1
            s['label'] = label
            all_rows.append(s)

    # Write aggregate
    if all_rows:
        keys = ['label', 'ar', 'window_start', 'edges', 'active_nodes', 'avg_degree']
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Written stats to {out_csv}")
    else:
        print("No data found.")

if __name__ == "__main__":
    main()
