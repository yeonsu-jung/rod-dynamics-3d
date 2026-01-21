#!/bin/bash
# Iterator script to submit connectivity analysis for all N directories
# Usage: ./parametric_study/iter_submit_mujoco_connectivity.sh

BASE_DIR="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs_mujoco/relaxation_3rd_multithreading_3rd_iterated_runs"
SUBMIT_SCRIPT="parametric_study/submit_mujoco_connectivity_analysis.sh"

echo "Submitting connectivity analysis jobs..."
echo "Base Dir: $BASE_DIR"

# Find all N* directories
DIRS=$(find "$BASE_DIR" -maxdepth 1 -type d -name "N*" | sort -V)

if [ -z "$DIRS" ]; then
    echo "No N* directories found."
    exit 1
fi

COUNT=0
for d in $DIRS; do
    N_DIR=$(basename "$d")
    echo "Submitting: $N_DIR"
    sbatch "$SUBMIT_SCRIPT" "$d"
    COUNT=$((COUNT+1))
done

echo "Submitted $COUNT jobs."
