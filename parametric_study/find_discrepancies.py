#!/usr/bin/env python3
"""find_discrepancies.py

Finds pairs of data points with similar Alpha/N but different Normalized Entanglement.
Specifically for Friction = 1.0 (or closest available).
"""

import csv
import argparse
from pathlib import Path
import math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file", type=Path, help="Path to exported collapsed data CSV")
    ap.add_argument("--tolerance", type=float, default=0.05, help="Relative tolerance for Alpha/N similarity (default 5%)")
    ap.add_argument("--diff-threshold", type=float, default=0.1, help="Absolute difference threshold for entanglement (default 0.1)")
    ap.add_argument("--friction", type=float, default=1.0, help="Friction value to analyze")
    ap.add_argument("--x-col", type=str, default="alpha_over_N", help="Column name for X metric")
    args = ap.parse_args()

    data = []
    with args.csv_file.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fric = float(row["friction"])
                if abs(fric - args.friction) > 0.05: # Loose match for friction
                    continue
                
                # Check for requested x-col. If not present, try to calculate? 
                # The new CSVs have N_over_alpha.
                if args.x_col in row:
                    x_val = float(row[args.x_col])
                else:
                    # Fallback/Calc
                    a = float(row["alpha"])
                    n = int(float(row["N"]))
                    if args.x_col == "N_over_alpha":
                        x_val = n / a
                    else:
                        x_val = a / n
                
                item = {
                    "alpha": float(row["alpha"]),
                    "N": int(float(row["N"])),
                    "x": x_val,
                    "y": float(row["final_norm_entanglement"])
                }
                data.append(item)
            except ValueError:
                continue

    print(f"Loaded {len(data)} points for friction ~ {args.friction}")
    
    # Sort by x
    data.sort(key=lambda d: d["x"])
    
    found_any = False
    
    print(f"\nSearching for pairs with X similarity < {args.tolerance*100}% and Y diff > {args.diff_threshold}...\n")
    print(f"{'Alpha1':<8} {'N1':<6} {'X1':<10} {'Y1':<10} | {'Alpha2':<8} {'N2':<6} {'X2':<10} {'Y2':<10} | {'X_Diff%':<8} {'Y_Diff':<8}")
    print("-" * 100)
    
    # Compare every pair? O(N^2) but N is small (hundreds). 
    # Or just sliding window.
    # Let's do all pairs within tolerance since "similar" is local.
    
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            d1 = data[i]
            d2 = data[j]
            
            # Check X similarity
            # use relative difference based on d1
            x_diff = abs(d1["x"] - d2["x"])
            avg_x = (d1["x"] + d2["x"]) / 2.0
            if avg_x == 0: continue
            
            x_rel_diff = x_diff / avg_x
            
            if x_rel_diff > args.tolerance:
                # Since sorted, if we exceed tolerance, we can break inner loop 
                # (optimization, assumed X is positive)
                break
                
            # Check Y difference
            y_diff = abs(d1["y"] - d2["y"])
            
            if y_diff > args.diff_threshold:
                found_any = True
                print(f"{d1['alpha']:<8.1f} {d1['N']:<6} {d1['x']:<10.4f} {d1['y']:<10.4f} | "
                      f"{d2['alpha']:<8.1f} {d2['N']:<6} {d2['x']:<10.4f} {d2['y']:<10.4f} | "
                      f"{x_rel_diff*100:<7.2f}% {y_diff:<8.4f}")

    if not found_any:
        print("No discrepancies found matching criteria.")

if __name__ == "__main__":
    main()
