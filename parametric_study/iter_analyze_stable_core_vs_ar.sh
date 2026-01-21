#!/bin/bash
# iter_analyze_stable_core_vs_ar.sh
# 
# Iterates over relax3rd_N*_sweep folders and submits stable core vs AR analysis jobs.

RUNS_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/relax3rd_second_complete_run_analysis"
SCRIPT="parametric_study/submit_stable_core_ar_analysis.sh"
OUTPUT_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/stable_core_ar_analysis"

mkdir -p "$OUTPUT_BASE"

echo "Scanning $RUNS_BASE for relax3rd_*_sweep..."

for dir in "$RUNS_BASE"/relax3rd_*_sweep; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir") # e.g., relax3rd_N500_sweep
        
        # Extract N:
        tmp=${dirname#*_N}
        N=${tmp%%_*}
        
        # Check if N is a number
        if ! [[ "$N" =~ ^[0-9]+$ ]]; then
            echo "Skipping $dirname (cannot parse N)"
            continue
        fi
        
        echo "---------------------------------------------------"
        echo "Submitting AR analysis for: $(basename "$dir")"
        echo "N=$N"
        
        sbatch "$SCRIPT" "$dir" "$N" "$OUTPUT_BASE"
    fi
done

echo "Done submitting AR analysis jobs."
