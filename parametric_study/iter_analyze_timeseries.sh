#!/bin/bash
# iter_analyze_timeseries.sh
# 
# Iterates over runs and submits time series analysis jobs.

RUNS_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs_mujoco/relaxation_3rd_multithreading_3rd_iterated_runs"
OUTPUT_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/timeseries_analysis"
SCRIPT="parametric_study/submit_timeseries_analysis.sh"

mkdir -p "$OUTPUT_BASE"

echo "Scanning $RUNS_BASE for endpoints_formatted.csv files..."

# Find all N* directories
for n_dir in "$RUNS_BASE"/N*; do
    if [ ! -d "$n_dir" ]; then
        continue
    fi
    
    N=$(basename "$n_dir" | sed 's/N//')
    echo "Processing N=$N..."
    
    OUTPUT_N_DIR="${OUTPUT_BASE}/N${N}"
    mkdir -p "$OUTPUT_N_DIR"
    
    # Find all endpoints files in this N directory
    COUNT=0
    for endpoints_file in $(find "$n_dir" -name "endpoints_formatted.csv" | sort); do
        echo "  Submitting: $(basename $(dirname "$endpoints_file"))"
        sbatch "$SCRIPT" "$endpoints_file" "$OUTPUT_N_DIR"
        COUNT=$((COUNT+1))
    done
    
    echo "  Submitted $COUNT jobs for N=$N"
done

echo "Done submitting time series analysis jobs."
