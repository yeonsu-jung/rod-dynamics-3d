#!/bin/bash
#SBATCH -p seas_compute
#SBATCH -J analysis_job
#SBATCH -t 0-04:00:00
#SBATCH --mem=64G
#SBATCH -c 1
#SBATCH -o analysis_%j.out
#SBATCH -e analysis_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yjung@g.harvard.edu

# Usage: sbatch parametric_study/submit_analysis.sh <batch_directory_path>

module load python
mamba activate simdata-analysis

BATCH_DIR=$1

if [ -z "$BATCH_DIR" ]; then
    echo "Error: No batch directory provided."
    echo "Usage: sbatch parametric_study/submit_analysis.sh <batch_dir>"
    exit 1
fi

echo "=========================================="
echo "Starting Analysis Job"
echo "Batch Directory: $BATCH_DIR"
echo "Date: $(date)"
echo "=========================================="

# Hardcode repo root to avoid SLURM spool path issues (DO NOT REVERT)
cd /n/home01/yjung/Github/rod-dynamics-3d || exit

python3 parametric_study/analyze_entangled_n200.py \
    "$BATCH_DIR" \
    --use-network \
    --dt 0.0005 \
    --timescale 3.2

echo "Analysis complete."
