#!/bin/bash
#SBATCH -p seas_compute
#SBATCH -J topology_N200
#SBATCH -t 0-02:00:00
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -o topology_N200_%j.out
#SBATCH -e topology_N200_%j.err
#SBATCH --mail-type=END

# Batch topology analysis for N200 MuJoCo runs
# Usage: sbatch parametric_study/submit_topology_analysis.sh

module load python
mamba activate mujoco-env

TARGET_DIR="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs_mujoco/relaxation_3rd_multithreading_3rd_iterated_runs/N200"
OUTPUT_DIR="${TARGET_DIR}/topology_analysis"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Starting Topology Analysis for N200"
echo "Target: $TARGET_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Find all endpoints_formatted.csv files
RUNS=$(find "$TARGET_DIR" -name "endpoints_formatted.csv" | sort)
NUM_RUNS=$(echo "$RUNS" | wc -l)

echo "Found $NUM_RUNS runs with endpoints_formatted.csv"
echo ""

# Process each run
COUNT=0
for ENDPOINTS_FILE in $RUNS; do
    RUN_DIR=$(dirname "$ENDPOINTS_FILE")
    RUN_NAME=$(basename "$RUN_DIR")
    
    echo "[$((COUNT+1))/$NUM_RUNS] Processing: $RUN_NAME"
    
    # Run topology analysis
    python3 study/analyze_topology_evolution.py "$ENDPOINTS_FILE" > "${OUTPUT_DIR}/${RUN_NAME}_topology.txt" 2>&1
    
    # Copy the generated CSV to output directory
    if [ -f "topology_evolution.csv" ]; then
        mv topology_evolution.csv "${OUTPUT_DIR}/${RUN_NAME}_topology.csv"
    fi
    
    COUNT=$((COUNT+1))
done

echo ""
echo "=========================================="
echo "Topology analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
