#!/bin/bash
#SBATCH -p seas_compute
#SBATCH -J timeseries_analysis
#SBATCH -t 0-02:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o timeseries_analysis_%j.out
#SBATCH -e timeseries_analysis_%j.err
#SBATCH --mail-type=END

# Time series analysis for a single run
# Usage: sbatch parametric_study/submit_timeseries_analysis.sh <endpoints_csv> [output_dir]

module load python
mamba activate mujoco-env

ENDPOINTS_FILE="$1"
OUTPUT_DIR="${2:-.}"

if [ -z "$ENDPOINTS_FILE" ]; then
    echo "Error: Missing endpoints file argument"
    echo "Usage: sbatch submit_timeseries_analysis.sh <endpoints_csv> [output_dir]"
    exit 1
fi

if [ ! -f "$ENDPOINTS_FILE" ]; then
    echo "Error: File not found: $ENDPOINTS_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Extract run name from file path
RUN_NAME=$(basename $(dirname "$ENDPOINTS_FILE"))
OUTPUT_FILE="${OUTPUT_DIR}/${RUN_NAME}_timeseries.csv"

echo "=========================================="
echo "Starting Time Series Analysis"
echo "Input: $ENDPOINTS_FILE"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

# Run the analysis with stride=10 (analyze every 10th frame)
python3 study/analyze_timeseries.py \
    "$ENDPOINTS_FILE" \
    --output "$OUTPUT_FILE" \
    --stride 10 \
    --plot

echo ""
echo "=========================================="
echo "Time series analysis complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "=========================================="
