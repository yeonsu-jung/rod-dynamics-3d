
import pandas as pd
import glob
import os
import sys

def combine_results(n_value, input_dir):
    print(f"Combining results for N={n_value} in {input_dir}")
    pattern = os.path.join(input_dir, f"stable_core_N{n_value}_AR*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No files found matching {pattern}")
        return

    dfs = [pd.read_csv(f) for f in csv_files]
    if not dfs:
        return
        
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sort
    if 'mu' in combined.columns and 'AR' in combined.columns:
        combined = combined.sort_values(['mu', 'AR'])
    elif 'AR' in combined.columns:
        combined = combined.sort_values('AR')
        
    output_file = os.path.join(input_dir, f"stable_core_vs_ar_N{n_value}_all.csv")
    combined.to_csv(output_file, index=False)
    print(f"Saved {output_file} with {len(combined)} rows")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python combine_results.py <N> <INPUT_DIR>")
        sys.exit(1)
        
    combine_results(sys.argv[1], sys.argv[2])
