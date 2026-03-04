#!/bin/bash
#SBATCH -J broadphase_bench
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH --mem=16G
#SBATCH -o benchmark_broadphase_%j.out
#SBATCH -e benchmark_broadphase_%j.err

module load cmake
module load gcc/13.2.0-fasrc01
module load cuda/12.9.1-fasrc01

REPO=/n/home01/yjung/Github/rod-dynamics-3d
BUILD=$REPO/build_cuda

mkdir -p "$BUILD"
cd "$BUILD"

cmake "$REPO" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_HEADLESS=ON \
    -DENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc \
    > cmake.log 2>&1

make -j4 benchmark_broadphase 2>&1 | tail -5

echo "========================================"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Host: $(hostname)"
echo "========================================"

./benchmark_broadphase
