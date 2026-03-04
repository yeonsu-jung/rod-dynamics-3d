#!/bin/bash
#SBATCH -J broadphase_bench
#SBATCH -n 1
#SBATCH -c 4               # 4 CPU threads for the spatial-hash comparison
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o benchmark_broadphase_%j.out
#SBATCH -e benchmark_broadphase_%j.err

set -e  # Stop on any error

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BUILD="${ROOT}/build_cuda"

echo "============================================================"
echo "Broadphase benchmark: CPU spatial-hash vs CUDA O(N^2)"
echo "Node: $(hostname)    GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Date: $(date)"
echo "============================================================"

# --- Load modules -----------------------------------------------------------
module load gcc/13.2.0-fasrc01
module load cuda/12.9.1-fasrc01

echo "CUDA version: $(nvcc --version | grep release | awk '{print $5,$6}')"
echo "GCC  version: $(gcc --version | head -1)"

# --- Configure and build ----------------------------------------------------
mkdir -p "${BUILD}"
cd "${BUILD}"

cmake "${ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_HEADLESS=ON \
    -DENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    > cmake_config.log 2>&1

echo "CMake configure done."
make -j4 benchmark_broadphase 2>&1 | tail -5
echo "Build done."

# --- Run benchmark ----------------------------------------------------------
echo ""
echo "--- OMP threads: ${OMP_NUM_THREADS:-auto} ---"
export OMP_NUM_THREADS=4

"${BUILD}/benchmark_broadphase"

echo ""
echo "Done.  Results above."
