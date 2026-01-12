import csv
import pathlib
import sys
import numpy as np

root = pathlib.Path(sys.argv[1])
target_n = 200
target_fric = 0.4

print(f"Scanning {root} for N={target_n}, Friction={target_fric}...")
summary_files = list(root.rglob("summary.csv"))
print(f"Found {len(summary_files)} summary files.")

data = [] # (AR, EntNorm)

for csv_path in summary_files:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fric = float(row.get("friction", -1))
                n = int(float(row.get("N", 0)))
                ar = int(float(row.get("AR", 0)))
                val = float(row.get("ent_norm_end", "nan"))
                
                if n == target_n and abs(fric - target_fric) < 1e-6:
                    data.append((ar, val))
            except:
                pass

print(f"Found {len(data)} points.")
data.sort(key=lambda x: x[0])

# Group by AR
from itertools import groupby
for ar, group in groupby(data, key=lambda x: x[0]):
    vals = [g[1] for g in group]
    mean_val = np.mean(vals)
    print(f"AR={ar}: Mean={mean_val:.4f}, Raw={vals}")
