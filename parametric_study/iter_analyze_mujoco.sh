#!/bin/bash
# iter_analyze_mujoco.sh
# 
# Iterates over mujoco_*_sweep folders in runs_mujoco/ and submits analysis jobs.

RUNS_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs_mujoco"
SCRIPT="parametric_study/submit_mujoco_analysis.sh"

echo "Scanning $RUNS_BASE for mujoco_*_sweep..."

for dir in "$RUNS_BASE"/mujoco_*_sweep; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir") # e.g., mujoco_N500_sweep
        
        # Extract N
        # remove prefix "mujoco_N"
        tmp=${dirname#mujoco_N}
        # remove suffix "_sweep"
        N=${tmp%%_sweep}
        
        if ! [[ "$N" =~ ^[0-9]+$ ]]; then
            echo "Skipping $dirname (cannot parse N)"
            continue
        fi

        echo "---------------------------------------------------"
        echo "Submitting analysis for: $dirname"
        
        sbatch "$SCRIPT" "$dir"
    fi
done

echo "Done submitting analysis jobs."
