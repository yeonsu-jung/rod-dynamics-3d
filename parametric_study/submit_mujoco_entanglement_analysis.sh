#!/bin/bash
#SBATCH -J mujoco_entanglement
#SBATCH -o analysis_entanglement_%j.out
#SBATCH -e analysis_entanglement_%j.err
#SBATCH -p seas_compute
#SBATCH --mem=8G
#SBATCH -t 0-04:00
#SBATCH -c 1

source ~/.bashrc
module load python
mamba activate mujoco-env

# Argument 1: Batch directory
BATCH_DIR="$1"

if [ -z "$BATCH_DIR" ]; then
    echo "Usage: sbatch submit_mujoco_entanglement_analysis.sh <BATCH_DIR>"
    exit 1
fi

echo "Starting MuJoCo Entanglement Analysis Job"
echo "Batch Directory: $BATCH_DIR"
echo "Date: $(date)"

python3 parametric_study/analyze_entanglement_mujoco.py "$BATCH_DIR" --jobs 1

echo "Analysis complete."
