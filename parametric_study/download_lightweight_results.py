import os
import subprocess

# Configuration
remote_user = "yjung"
remote_host = "odyssey.rc.fas.harvard.edu"
# Base path where the N-sweep folders are located
remote_path = "/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs"
# Update this to your local Dropbox or data directory
local_path = "/Users/yeonsu/Harvard University Dropbox/Yeonsu Jung/Data/lightweight_runs_results"

# Ensure local directory exists
os.makedirs(local_path, exist_ok=True)

# Construct rsync command
# We target only the lightweight sweep folders and specific summary/output files.
cmd = [
    "rsync",
    "-avm",               # Archive, verbose, prune empty dirs
    "--include=relax3rd_lightweight_*_sweep/",
    "--include=relax3rd_lightweight_*_sweep/summary.csv",
    "--include=relax3rd_lightweight_*_sweep/*/",
    "--include=relax3rd_lightweight_*_sweep/*/output.csv",
    "--include=relax3rd_lightweight_*_sweep/*/scene.json",
    "--exclude=*",        # Exclude everything else (per-rod, binaries, etc)
    f"{remote_user}@{remote_host}:{remote_path}/",
    local_path
]

print(f"Running command: {' '.join(cmd)}")

try:
    subprocess.run(cmd, check=True)
    print("\nDownload completed successfully.")
    print(f"Data saved to: {local_path}")
except subprocess.CalledProcessError as e:
    print(f"Error occurred during rsync: {e}")
