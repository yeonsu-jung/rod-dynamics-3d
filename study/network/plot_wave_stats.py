
"""
Plot evolution of mean degrees from square-wave accumulated network analysis.
Usage: python3 study/network/plot_wave_stats.py <sweep_folder>
"""

import argparse
import csv
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
from pathlib import Path

def read_stats_csv(csv_path):
    data = defaultdict(list)
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ar = int(row['ar'])
                time = int(row['window_start'])
                deg = float(row['avg_degree'])
                data[ar].append((time, deg))
            except (ValueError, KeyError):
                continue
    
    # Sort data for each AR by time
    for ar in data:
        data[ar].sort(key=lambda x: x[0])
    
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_folder", type=Path)
    args = parser.parse_args()

    csv_path = args.sweep_folder / "wave_network_stats.csv"
    if not csv_path.exists():
        sys.exit(f"Stats file not found: {csv_path}")

    data = read_stats_csv(csv_path)
    if not data:
        sys.exit("No data found in CSV.")

    sorted_ars = sorted(data.keys())
    
    # Use a divergent colormap (Spectral)
    # Use index-based spacing to ensure distinct colors for clustered ARs
    cmap = plt.get_cmap("Spectral")
    import numpy as np
    
    colors = {ar: cmap(i / (len(sorted_ars) - 1)) if len(sorted_ars) > 1 else cmap(0.5) 
              for i, ar in enumerate(sorted_ars)}

    scales = [
        ("linear", "linear", "wave_degree_linear.png"),
        ("linear", "log", "wave_degree_semilogy.png"),
        ("log", "log", "wave_degree_loglog.png")
    ]

    for xscale, yscale, filename in scales:
        plt.figure(figsize=(10, 6))
        
        for ar in sorted_ars:
            points = data[ar]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            color = colors[ar]
            label = f"AR={ar}"
            
            plt.plot(xs, ys, marker='o', linestyle='-', label=label, color=color, alpha=0.8, markeredgecolor='w', markeredgewidth=0.5)

        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.xlabel("Time (Frame Window Start)")
        plt.ylabel("Mean Degree (Accumulated)")
        plt.title(f"Evolution of Mean Degree\n{args.sweep_folder.name}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, which="both")
        plt.tight_layout()

        out_path = args.sweep_folder / filename
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
        plt.close()

if __name__ == "__main__":
    main()
