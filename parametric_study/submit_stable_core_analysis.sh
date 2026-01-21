#!/bin/bash
#SBATCH -p sapphire
#SBATCH -J stable_core_analysis
#SBATCH -t 0-04:00:00
#SBATCH --mem=32G
#SBATCH -c 1
#SBATCH -o stable_core_analysis_%j.out
#SBATCH -e stable_core_analysis_%j.err
#SBATCH --mail-type=END

# Batch stable core analysis for a given N value
# Usage: sbatch parametric_study/submit_stable_core_analysis.sh <input_dir> <N> <output_base>

module load python
mamba activate mujoco-env

INPUT_DIR="$1"
N_VALUE="$2"
OUTPUT_BASE="$3"

if [ -z "$INPUT_DIR" ] || [ -z "$N_VALUE" ] || [ -z "$OUTPUT_BASE" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch submit_stable_core_analysis.sh <input_dir> <N> <output_base>"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_BASE}/N${N_VALUE}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Starting Stable Core Analysis for N=$N_VALUE"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Find all endpoints.csv files
RUNS=$(find "$INPUT_DIR" -name "endpoints.csv" | sort)
NUM_RUNS=$(echo "$RUNS" | wc -l)

echo "Found $NUM_RUNS runs with endpoints.csv"
echo ""

# Collect all run directories for batch analysis
RUN_DIRS=""
for ENDPOINTS_FILE in $RUNS; do
    RUN_DIR=$(dirname "$ENDPOINTS_FILE")
    RUN_DIRS="$RUN_DIRS $RUN_DIR"
done

# Run the analysis script on all directories
python3 study/analyze_stable_core_vs_n.py \
    $RUN_DIRS \
    --n-value "$N_VALUE" \
    --output "${OUTPUT_DIR}/stable_core_N${N_VALUE}.csv"

echo ""
echo "=========================================="
echo "Stable core analysis complete!"
echo "Results saved to: ${OUTPUT_DIR}/stable_core_N${N_VALUE}.csv"
echo "=========================================="
