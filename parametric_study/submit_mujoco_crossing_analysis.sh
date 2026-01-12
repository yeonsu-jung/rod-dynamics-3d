#!/bin/bash
#SBATCH -p seas_compute
#SBATCH -J mujoco_crossings
#SBATCH -t 1-00:00:00
#SBATCH --mem=1G
#SBATCH -c 4
#SBATCH -o crossing_analysis_%j.out
#SBATCH -e crossing_analysis_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yjung@g.harvard.edu

# Usage: sbatch parametric_study/submit_mujoco_crossing_analysis.sh <batch_directory_path>

module load python
mamba activate mujoco-env

BATCH_DIR=$1
BINARY_PATH="build/compute_min_crossing"

if [ -z "$BATCH_DIR" ]; then
    echo "Error: No batch directory provided."
    echo "Usage: sbatch parametric_study/submit_mujoco_crossing_analysis.sh <batch_dir>"
    exit 1
fi

echo "=========================================="
echo "Starting MuJoCo Crossing Analysis Job"
echo "Batch Directory: $BATCH_DIR"
echo "Date: $(date)"
echo "=========================================="

cd /n/home01/yjung/Github/rod-dynamics-3d || exit

if [ ! -f "$BINARY_PATH" ]; then
    echo "Compiling compute_min_crossing..."
    g++ -O3 -fopenmp -I./src/include src/tools/compute_min_crossing.cpp -o build/compute_min_crossing
fi

python3 parametric_study/analyze_crossing_mujoco.py "$BATCH_DIR" --binary "$BINARY_PATH" --jobs $SLURM_CPUS_PER_TASK

echo "Analysis complete."
