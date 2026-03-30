#!/bin/bash
#SBATCH -J free_rod_allpack_allmu_200k_dt5e4
#SBATCH -p seas_compute
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -N 1
#SBATCH -t 0-02:00:00
#SBATCH --mem=64G
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err
#SBATCH --mail-type=END

set -euo pipefail

ROOT=/n/home01/yjung/Github/rod-dynamics-3d
source "$ROOT/.venv/bin/activate"

python "$ROOT/parametric_study/run_descending_mu_stop_test.py" \
    --workers 32 \
    --sim-threads 1 \
    --steps 200000 \
    --dt 5e-4 \
    --mu-values 1.0 0.4 0.2 0.1 \
    --stop-slide-vel-threshold 1e-5 \
    --stop-slide-vel-min-steps 1000 \
    --keep-all-cases \
    --job-name free_rod_allpack_allmu_200k_dt5e4_slurm
