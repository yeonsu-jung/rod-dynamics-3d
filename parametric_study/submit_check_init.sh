#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 0-00:10
#SBATCH -p seas_compute
#SBATCH --mem=4000
#SBATCH -o check_init_%j.out
#SBATCH -e check_init_%j.err
#SBATCH --mail-type=END
#SBATCH --job-name=check_init

set -euo pipefail

# Load modules if needed (adjust based on your cluster environment)
# module load python 

echo "======================================"
echo "Check Init Job"
echo "PWD: $(pwd)"
echo "======================================"

cd build

./rigidbody_viewer_3d \
  --scene ../assets/scenes/default_with_csv.json \
  --init-csv ../initial-configs/rods932_bruteforce_checked_v3.csv \
  --output output_rods932_bruteforce_checked_v3_prestep.csv \
  --headless \
  --steps 1 \
  --debug-min-gap \
  --check-init-nonpenetration

echo ""
echo "Job complete."
