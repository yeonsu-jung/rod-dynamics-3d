
import os
import glob
import subprocess
import re

# Base directory
base_dir = "study/stable_core_analysis"
script_path = "study/plot_stable_core_vs_ar.py"
python_exec = "/n/home01/yjung/.conda/envs/mujoco-env/bin/python"

# Find all N directories
n_dirs = glob.glob(os.path.join(base_dir, "N*"))

for n_dir in n_dirs:
    dir_name = os.path.basename(n_dir)
    # Extract N value
    match = re.match(r"N(\d+)", dir_name)
    if not match:
        continue
    n_val = match.group(1)
    
    # Find the CSV file
    csv_file = os.path.join(n_dir, f"stable_core_N{n_val}.csv")
    if not os.path.exists(csv_file):
        print(f"Skipping {n_val}: {csv_file} not found")
        continue

    print(f"Processing N={n_val}...")
    
    # Define output prefix
    output_prefix = os.path.join(n_dir, f"stable_core_vs_ar_N{n_val}")
    
    # Construct command
    cmd = [
        python_exec,
        script_path,
        csv_file,
        "--output", output_prefix,
        "--n-value", n_val,
        "--individual-plots"
    ]
    
    # Run command
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully generated plots for N={n_val}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating plots for N={n_val}: {e}")

print("Batch plot generation complete.")
