#!/bin/bash
# Setup conda environment for MuJoCo MJX (GPU-accelerated simulation)
# Usage: bash scripts/setup_mjx_env.sh

set -e

ENV_NAME="mjx-gpu"

echo "=== Creating conda environment: ${ENV_NAME} ==="
conda create -n ${ENV_NAME} python=3.11 -y

echo "=== Activating environment ==="
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "=== Installing JAX with CUDA 12 support ==="
pip install --upgrade "jax[cuda12]"

echo "=== Installing MuJoCo and MJX ==="
pip install mujoco mujoco-mjx

echo "=== Installing additional dependencies ==="
pip install numpy scipy pyyaml matplotlib

echo ""
echo "=== Verifying installation ==="
python -c "
import mujoco
print(f'MuJoCo version: {mujoco.__version__}')

from mujoco import mjx
print('MJX imported successfully')

import jax
print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')
print()
print('Setup complete! To use:')
print('  conda activate ${ENV_NAME}')
"

echo ""
echo "=== Done ==="
echo "Activate with: conda activate ${ENV_NAME}"
