#!/bin/bash
#SBATCH -p seas_compute
#SBATCH -J mujoco_connectivity
#SBATCH -t 0-04:00:00
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -o mujoco_connectivity_%j.out
#SBATCH -e mujoco_connectivity_%j.err
#SBATCH --mail-type=END

# Usage: sbatch parametric_study/submit_mujoco_connectivity_analysis.sh <target_directory>

if [ -z "$1" ]; then
    echo "Error: No target directory provided."
    echo "Usage: sbatch $0 <target_directory>"
    exit 1
fi

TARGET_DIR="$1"
DIR_NAME=$(basename "$TARGET_DIR")

module load python
mamba activate mujoco-env

BINARY_PATH="build/extract_connectivity"
PY_SCRIPT="parametric_study/analyze_connectivity_mujoco.py"
OUTPUT_DIR="${TARGET_DIR}/connectivity_analysis"

# Compile binary if needed
if [ ! -f "$BINARY_PATH" ]; then
    echo "Compiling connectivity extraction tool..."
    g++ -std=c++17 -O3 -fopenmp src/tools/extract_connectivity_endpoints.cpp -o "$BINARY_PATH"
fi

echo "=========================================="
echo "Starting MuJoCo Connectivity Analysis"
echo "Target: $TARGET_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python3 "$PY_SCRIPT" "$TARGET_DIR" --binary "$BINARY_PATH" --output-dir "$OUTPUT_DIR" --jobs $SLURM_CPUS_PER_TASK

echo "Analysis complete."
