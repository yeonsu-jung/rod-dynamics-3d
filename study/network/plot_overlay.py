
"""
Overlay all-pairs RMS displacement analysis from a network sweep folder.
Usage: python3 study/network/plot_overlay.py <sweep_folder>
"""

import argparse
import csv
import matplotlib.pyplot as plt
import re
import sys
from pathlib import Path

def parse_ar_from_name(name):
    m = re.search(r"AR(\d+)", name)
    if m:
        return int(m.group(1))
    return None

def read_rms_csv(csv_path):
    times = []
    rms_vals = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Expecting columns like: frame, count, mean_dmin, rms_dmin, std_dmin
                # Assuming 'mean_dmin' or 'rms_dmin' is the metric of interest.
                # The user requested 'total overlaid plot', usually RMS displacement (or RMS dmin).
                # Wait, 'allpairs_rms_dmin.csv' suggests the metric is RMS of minimum distances.
                # Let's check columns in a moment. But typically:
                # frame,count,mean_dmin,rms_dmin,std_dmin
                # We can deduce time if dt is known, but usually frame is x-axis.
                # Let's plot Frame vs RMS_DMIN.
                frame = int(row['frame'])
                val = float(row['rms_dmin'])
                times.append(frame)
                rms_vals.append(val)
            except (ValueError, KeyError):
                continue
    return times, rms_vals

def main():
    parser = argparse.ArgumentParser(description="Overlay RMS plots from sweep.")
    parser.add_argument("sweep_folder", type=Path, help="Path to the sweep run folder")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    if not args.sweep_folder.exists():
        print(f"Error: Folder not found: {args.sweep_folder}")
        sys.exit(1)

    # Find all subfolders with the CSV
    data = []
    for sub in args.sweep_folder.iterdir():
        if not sub.is_dir():
            continue
        csv_path = sub / "allpairs_rms_dmin.csv"
        if not csv_path.exists():
            continue
        
        ar = parse_ar_from_name(sub.name)
        if ar is None:
            label = sub.name
            ar = 0 # Sort key
        else:
            label = f"AR={ar}"
        
        frames, vals = read_rms_csv(csv_path)
        if not frames:
            continue
            
        data.append({
            'ar': ar,
            'label': label,
            'frames': frames,
            'vals': vals
        })

    if not data:
        print("No data found to plot.")
        sys.exit(0)

    # Sort by AR
    data.sort(key=lambda x: x['ar'])

    # Use a divergent colormap (Spectral)
    # Use index-based spacing to ensure distinct colors for clustered ARs
    cmap = plt.get_cmap("Spectral")
    import numpy as np
    
    sorted_ars = [d['ar'] for d in data]
    # Create a mapping from AR to color
    # Note: data is already sorted by AR
    colors = {d['ar']: cmap(i / (len(data) - 1)) if len(data) > 1 else cmap(0.5) 
              for i, d in enumerate(data)}

    scales = [
        ("linear", "linear", "overlay_rms_linear.png"),
        ("linear", "log", "overlay_rms_semilogy.png"),
        ("log", "log", "overlay_rms_loglog.png")
    ]

    for xscale, yscale, filename in scales:
        plt.figure(figsize=(10, 6))
        
        for d in data:
            color = colors[d['ar']]
            plt.plot(d['frames'], d['vals'], label=d['label'], color=color, alpha=0.8, marker='o', markersize=2, markeredgecolor='none')

        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.xlabel("Frame")
        plt.ylabel("All-Pairs RMS Minimum Distance")
        plt.title(f"RMS Distance Overlay ({xscale}-{yscale})\n{args.sweep_folder.name}")
        
        # Adjust legend position
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.grid(True, alpha=0.3, which="both")
        plt.tight_layout()

        # Handle user-provided out path if singular? 
        # If user provides --out, we might just append suffix or ignore.
        # For now, let's stick to auto-naming inside the folder unless --out is dir.
        # But to be safe with existing arg logic:
        if args.out:
            # If args.out is specific file, use it as base
            base = args.out.parent / args.out.stem
            out_path = Path(f"{base}_{filename.split('_')[-1]}")
        else:
            out_path = args.sweep_folder / filename
            
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
        plt.close() # Close figure to free memory

if __name__ == "__main__":
    main()
