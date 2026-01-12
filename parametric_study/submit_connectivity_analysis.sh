#!/bin/bash
#SBATCH -p seas_compute
#SBATCH -J connectivity_analysis
#SBATCH -t 0-02:00:00
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH -o connectivity_analysis_%j.out
#SBATCH -e connectivity_analysis_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yjung@g.harvard.edu

# Usage: sbatch parametric_study/submit_connectivity_analysis.sh <target_directory>

if [ -z "$1" ]; then
    echo "Error: No target directory provided."
    exit 1
fi

TARGET_DIR="$1"
DIR_NAME=$(basename "$TARGET_DIR")

#SBATCH --job-name=conn_analysis_${DIR_NAME}
#SBATCH -o connectivity_${DIR_NAME}_%j.out
#SBATCH -e connectivity_${DIR_NAME}_%j.err

module load python
mamba activate mujoco-env

BINARY_PATH="build/extract_connectivity_perrod"
PY_SCRIPT="parametric_study/analyze_connectivity_batch.py"
OUTPUT_DIR="${TARGET_DIR}/connectivity_analysis"

# Compile binary if needed
if [ ! -f "$BINARY_PATH" ]; then
    echo "Compiling extraction tool..."
    g++ -std=c++17 -O3 -fopenmp src/tools/extract_connectivity_perrod.cpp -o "$BINARY_PATH"
fi

echo "=========================================="
echo "Starting Connectivity Analysis for $TARGET_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python3 "$PY_SCRIPT" "$TARGET_DIR" --binary "$BINARY_PATH" --output-dir "$OUTPUT_DIR" --dt 0.0005 --jobs $SLURM_CPUS_PER_TASK

echo "Analysis complete."
