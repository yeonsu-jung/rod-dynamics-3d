import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
friction = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

print(f"Scanning {root} for friction={friction}...")
summary_files = list(root.rglob("summary.csv"))
print(f"Found {len(summary_files)} summary files.")

found_n = set()
points = 0

for csv_path in summary_files:
    print(f"Reading {csv_path}...")
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fric = float(row.get("friction", -1))
                if abs(fric - friction) > 1e-6:
                    continue
                n = int(float(row.get("N", 0)))
                ar = int(float(row.get("AR", 0)))
                found_n.add(n)
                points += 1
                # print(f"  Found N={n}, AR={ar}")
            except:
                pass

print(f"Total points: {points}")
print(f"Found N values: {sorted(list(found_n))}")
