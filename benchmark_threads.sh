#!/bin/bash
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -p seas_compute
#SBATCH --mem=8G
#SBATCH -o benchmark_%j.out
#SBATCH -e benchmark_%j.err

# Ensure we are in the build directory
cd /n/home01/yjung/Github/rod-dynamics-3d/build

echo "========================================"
echo "Running with 2 threads..."
export OMP_NUM_THREADS=2
time ./rigidbody_viewer_3d --headless --steps 1000 --init-csv ../initial-configs/rods_bf_1132.csv --scene ../assets/scenes/default_with_csv.json

echo "========================================"
echo "Running with 12 threads..."
export OMP_NUM_THREADS=12
time ./rigidbody_viewer_3d --headless --steps 1000 --init-csv ../initial-configs/rods_bf_1132.csv --scene ../assets/scenes/default_with_csv.json
