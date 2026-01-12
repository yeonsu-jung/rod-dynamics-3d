#!/bin/bash
# Usage: ./parametric_study/iter_submit_connectivity.sh

BASE_DIR="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/relax3rd_second_complete_run_analysis"
SUBMIT_SCRIPT="parametric_study/submit_connectivity_analysis.sh"

echo "Comparing connectivity analysis jobs..."
echo "Base Dir: $BASE_DIR"

# Find directories matching relax3rd_N*_sweep
# Using sort -V for version sort (N10, N15, N100...)
DIRS=$(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d -name "relax3rd_N*_sweep" | sort -V)

if [ -z "$DIRS" ]; then
    echo "No directories found."
    exit 1
fi

COUNT=0
for d in $DIRS; do
    echo "Submitting: $(basename "$d")"
    sbatch "$SUBMIT_SCRIPT" "$d"
    COUNT=$((COUNT+1))
done

echo "Submitted $COUNT jobs."
