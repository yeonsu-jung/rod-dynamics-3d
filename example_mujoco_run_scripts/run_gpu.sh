#!/bin/bash
# ==== GPU-enabled run script for MJX simulations ====
# Usage: sh run_gpu.sh NOTE PERIODIC_FLAG JOB_NAME N_VAL SEED_VAL

NOTE=$1
PERIODIC_FLAG=$2
JOB_NAME=$3
N_VAL=$4
SEED_VAL=$5

# Find root directory and script directory
ROOT_DIR=$(git rev-parse --show-toplevel)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Root directory: ${ROOT_DIR}"
echo "Script directory: ${SCRIPT_DIR}"

RUNS_FOLDER="${ROOT_DIR}/runs"
STAMP=$(date +"%Y%m%d-%H%M")

if [ -n "${JOB_NAME}" ] && [ -n "${N_VAL}" ] && [ -n "${SEED_VAL}" ]; then
    RESULT_FOLDER="${RUNS_FOLDER}/${JOB_NAME}/N${N_VAL}/${SEED_VAL}/${STAMP}_RUN_${NOTE}"
else
    RESULT_FOLDER="${RUNS_FOLDER}/${STAMP}_RUN_${NOTE}"
fi

# Exit if folder exists
if [ -d "${RESULT_FOLDER}" ]; then
    echo "Folder ${RESULT_FOLDER} already exists. Exiting."
    exit 1
fi
mkdir -p ${RESULT_FOLDER}

# Copy simulation files (MJX version) — utils are relative to this script
cp "${SCRIPT_DIR}/utils/perturb_rod_packings_mjx.py" "${RESULT_FOLDER}/perturb_rod_packings_mjx.py"
cp "${SCRIPT_DIR}/utils/first_analysis.py" "${RESULT_FOLDER}/first_analysis.py"
cp "${SCRIPT_DIR}/utils/util.py" "${RESULT_FOLDER}/util.py"
cp "${SCRIPT_DIR}/utils/packing_initialization.py" "${RESULT_FOLDER}/packing_initialization.py"
cp "${SCRIPT_DIR}/run_gpu.sh" "${RESULT_FOLDER}/run_gpu.sh"
cp "${SCRIPT_DIR}/run.py" "${RESULT_FOLDER}/run.py"
cp "${SCRIPT_DIR}/options.yml" "${RESULT_FOLDER}/options.yml"

if [ -n "${PERIODIC_FLAG}" ]; then
    echo "mode=${PERIODIC_FLAG}" > "${RESULT_FOLDER}/mode.info"
fi
echo "backend=mjx_gpu" >> "${RESULT_FOLDER}/mode.info"

mkdir -p "${ROOT_DIR}/logs/run_logs" 2>/dev/null
cp "${SCRIPT_DIR}/run.py" "${ROOT_DIR}/logs/run_logs/run_${STAMP}_RUN_${NOTE}.py" 2>/dev/null || true

# Generate GPU SLURM submission script
cat > "${RESULT_FOLDER}/Sbatch.sh" << 'SBATCH_EOF'
#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -c 4                # CPU threads for data I/O
#SBATCH -N 1                # Ensure all cores on one node
#SBATCH -t 0-12:00          # Runtime limit
#SBATCH -p seas_gpu         # GPU partition
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --mem=64000         # Memory (MB) — JIT compilation needs lots of CPU RAM
#SBATCH -o output_%j.out    # STDOUT
#SBATCH -e errors_%j.err    # STDERR
#SBATCH --mail-type=END
#SBATCH --mail-user=jung@seas.harvard.edu

module load python
conda activate mjx-gpu

# JAX's pip-installed NVIDIA packages put .so files under site-packages/nvidia/*/lib/
# We need to add them to LD_LIBRARY_PATH so JAX can find cuSPARSE, cuBLAS, etc.
NVIDIA_LIBS=$(python -c "import nvidia, os; pkgdir=os.path.dirname(nvidia.__file__); dirs=[os.path.join(pkgdir,d,'lib') for d in os.listdir(pkgdir) if os.path.isdir(os.path.join(pkgdir,d,'lib'))]; print(':'.join(dirs))")
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${LD_LIBRARY_PATH}

echo "=== GPU Info ==="
nvidia-smi
echo "================"

time python perturb_rod_packings_mjx.py
time python first_analysis.py
SBATCH_EOF

cd "${RESULT_FOLDER}"
sbatch Sbatch.sh
