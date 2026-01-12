#!/bin/bash
#SBATCH -p seas_compute
#SBATCH -J crossing_analysis
#SBATCH -t 1-00:00:00
#SBATCH --mem=32G
#SBATCH -c 48
#SBATCH -o analysis_crossing_%j.out
#SBATCH -e analysis_crossing_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yjung@g.harvard.edu

# Usage: sbatch parametric_study/submit_crossing_analysis.sh <batch_directory_path>

module load python
mamba activate simdata-analysis

BATCH_DIR=$1

if [ -z "$BATCH_DIR" ]; then
    echo "Error: No batch directory provided."
    echo "Usage: sbatch parametric_study/submit_crossing_analysis.sh <batch_dir>"
    exit 1
fi

echo "=========================================="
echo "Starting Min Crossing Analysis Job"
echo "Batch Directory: $BATCH_DIR"
echo "Date: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Hardcode repo root
cd /n/home01/yjung/Github/rod-dynamics-3d || exit

# Run python script with parallel jobs
python3 parametric_study/analyze_crossing_evolution.py \
    "$BATCH_DIR" \
    --binary build/compute_min_crossing \
    --jobs "$SLURM_CPUS_PER_TASK" \
    --dt 0.0005

echo "Analysis complete."
