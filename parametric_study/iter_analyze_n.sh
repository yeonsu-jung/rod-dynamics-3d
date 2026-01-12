#!/bin/bash
# iter_analyze_n.sh
# 
# Iterates over relax2nd_N*_sweep folders in runs/ and submits analysis jobs.

# RUNS_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs"
# RUNS_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/relax2nd_first_complete_run_analysis"
RUNS_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/relax3rd_second_complete_run_analysis"
SCRIPT="parametric_study/submit_analysis_pandas.sh"

echo "Scanning $RUNS_BASE for relax3rd_*_sweep..."

for dir in "$RUNS_BASE"/relax3rd_*_sweep; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir") # e.g., relax3rd_N500_sweep
        
        # Extract N:
        # 1. Remove everything up to "N" (e.g. "relax3rd_N")
        tmp=${dirname#*_N}
        # 2. Remove suffix (e.g. "_sweep")
        N=${tmp%%_*}
        
        # Check if N is a number
        if ! [[ "$N" =~ ^[0-9]+$ ]]; then
            echo "Skipping $dirname (cannot parse N)"
            continue
        fi

        # Filter for N=500 and N=1000 per user request
        # if [ "$N" -ne 500 ] && [ "$N" -ne 1000 ]; then
        # if [ "$N" -ne 500 ]; then
        #     echo "Skipping N=$N"
        #     continue
        # fi
        
        echo "---------------------------------------------------"
        echo "Submitting analysis for: $(basename "$dir")"
        
        sbatch "$SCRIPT" "$dir"
    fi
done

echo "Done submitting analysis jobs."
