#!/bin/bash
#SBATCH -p seas_compute
#SBATCH -J mujoco_analysis
#SBATCH -t 1-00:00:00
#SBATCH --mem=12G
#SBATCH -c 8
#SBATCH -o analysis_%j.out
#SBATCH -e analysis_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yjung@g.harvard.edu

# Usage: sbatch parametric_study/submit_mujoco_analysis.sh <batch_directory_path>
module load python
mamba activate mujoco-env

BATCH_DIR=$1

if [ -z "$BATCH_DIR" ]; then
    echo "Error: No batch directory provided."
    echo "Usage: sbatch parametric_study/submit_mujoco_analysis.sh <batch_dir>"
    exit 1
fi

echo "=========================================="
echo "Starting MuJoCo Analysis Job"
echo "Batch Directory: $BATCH_DIR"
echo "Date: $(date)"
echo "=========================================="

# Repository root assumption
cd /n/home01/yjung/Github/rod-dynamics-3d || exit

python3 parametric_study/analyze_mujoco_batch.py "$BATCH_DIR"

echo "Analysis complete."
