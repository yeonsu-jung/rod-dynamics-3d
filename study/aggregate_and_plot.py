
import pandas as pd
import glob
import os
import subprocess
import sys

def aggregate_data(base_dir, output_file):
    print(f"Searching for data in {base_dir}...")
    
    # improved pattern matching to capture all N folder results
    # Match both aggregate files and individual run files
    patterns = [
        os.path.join(base_dir, "N*", "stable_core_vs_ar_N*_all.csv"),
        os.path.join(base_dir, "N*", "stable_core_N*_mu*.csv")
    ]
    
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    
    if not files:
        print("No files found matching pattern!")
        return False
        
    print(f"Found {len(files)} files.")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Ensure N column exists if it's missing (can extract from filename or content)
            if 'N' not in df.columns:
                # Try to extract N from filename "stable_core_vs_ar_N{N}_all.csv"
                basename = os.path.basename(f)
                import re
                match = re.search(r'N(\d+)', basename)
                if match:
                    df['N'] = int(match.group(1))
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        print("No valid data loaded.")
        return False
        
    combined = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates if any (in case both summary and individual files are picked up)
    # We define duplicates as having same N, AR, mu (and possibly other metrics)
    # But usually just N, AR, mu is enough to identify a unique simulation metric point
    if 'N' in combined.columns and 'AR' in combined.columns and 'mu' in combined.columns:
        before_len = len(combined)
        combined = combined.drop_duplicates(subset=['N', 'AR', 'mu'])
        after_len = len(combined)
        if before_len > after_len:
            print(f"Removed {before_len - after_len} duplicate rows.")
    
    # Sort
    if 'N' in combined.columns:
        combined = combined.sort_values(['N', 'AR', 'mu'] if 'AR' in combined.columns and 'mu' in combined.columns else ['N'])
        
    combined.to_csv(output_file, index=False)
    print(f"Saved aggregated data to {output_file}")
    print(f"Total rows: {len(combined)}")
    return True

def generate_plots(data_file):
    # Plot with jet (mu overlay)
    cmd_jet = [
        sys.executable, 
        'study/plot_stable_core_vs_n.py', 
        data_file, 
        '--output', 'study/stable_core_vs_n_jet',
        '--colormap', 'jet',
        '--hue', 'mu'
    ]
    print(f"Running: {' '.join(cmd_jet)}")
    subprocess.run(cmd_jet, check=True)
    
    # Plot with viridis (mu overlay)
    cmd_viridis = [
        sys.executable, 
        'study/plot_stable_core_vs_n.py', 
        data_file, 
        '--output', 'study/stable_core_vs_n_viridis',
        '--colormap', 'viridis',
        '--hue', 'mu'
    ]
    print(f"Running: {' '.join(cmd_viridis)}")
    subprocess.run(cmd_viridis, check=True)
    
    # Plot with AR overlay (jet)
    cmd_ar = [
        sys.executable, 
        'study/plot_stable_core_vs_n.py', 
        data_file, 
        '--output', 'study/stable_core_vs_n_by_AR',
        '--colormap', 'jet',
        '--hue', 'AR'
    ]
    print(f"Running: {' '.join(cmd_ar)}")
    subprocess.run(cmd_ar, check=True)

    # Also update the default one (jet requested as preference)
    cmd_default = [
        sys.executable, 
        'study/plot_stable_core_vs_n.py', 
        data_file, 
        '--output', 'study/stable_core_analysis/stable_core_vs_n',
        '--colormap', 'jet',
        '--hue', 'mu'
    ]
    print(f"Running: {' '.join(cmd_default)}")
    subprocess.run(cmd_default, check=True)

    # Generate AR-specific plots
    print("Generating AR-specific plots...")
    df = pd.read_csv(data_file)
    if 'AR' in df.columns:
        unique_ars = sorted(df['AR'].unique())
        for ar in unique_ars:
            # Format AR for filename (handle integers cleanly)
            ar_str = str(int(ar)) if ar.is_integer() else str(ar).replace('.', '_')
            output_name = f'study/stable_core_analysis/stable_core_vs_n_AR{ar_str}'
            
            cmd_ar_filtered = [
                sys.executable, 
                'study/plot_stable_core_vs_n.py', 
                data_file, 
                '--output', output_name, 
                '--colormap', 'jet', 
                '--hue', 'mu',
                '--filter-ar', str(ar)
            ]
            print(f"Running for AR={ar}: {' '.join(cmd_ar_filtered)}")
            try:
                subprocess.run(cmd_ar_filtered, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate plot for AR={ar}: {e}")

if __name__ == "__main__":
    base_dir = "/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/stable_core_ar_analysis"
    output_csv = "stable_core_vs_n_aggregated.csv"
    
    if aggregate_data(base_dir, output_csv):
        generate_plots(output_csv)
