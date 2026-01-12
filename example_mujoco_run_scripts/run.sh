#!/bin/bash
# ==== ENTER INFO HERE =====

NOTE=$1
PERIODIC_FLAG=$2
JOB_NAME=$3
N_VAL=$4
SEED_VAL=$5

# DATAPATH="/n/home01/yjung/Github/dismech-rods-main/data/curved_samples/${NOTE}"

# find root directory of the project
ROOT_DIR=$(git rev-parse --show-toplevel)
echo "Root directory: ${ROOT_DIR}"

RUNS_FOLDER="${ROOT_DIR}/runs"

STAMP=$(date +"%Y%m%d-%H%M")

if [ -n "${JOB_NAME}" ] && [ -n "${N_VAL}" ] && [ -n "${SEED_VAL}" ]; then
    # New hierarchical structure
    # runs/{job_name}/N{n}/{seed}/{stamp}_{note}
    RESULT_FOLDER="${RUNS_FOLDER}/${JOB_NAME}/N${N_VAL}/${SEED_VAL}/${STAMP}_RUN_${NOTE}"
else
    # Fallback to authentic behavior
    RESULT_FOLDER="${RUNS_FOLDER}/${STAMP}_RUN_${NOTE}"
fi


# exit if the folder exists
if [ -d "${RESULT_FOLDER}" ]; then
    echo "Folder ${RESULT_FOLDER} already exists. Exiting."
    exit 1
fi
mkdir -p ${RESULT_FOLDER}

cp ${ROOT_DIR}/utils/perturb_rod_packings.py "${RESULT_FOLDER}/perturb_rod_packings.py" 
cp ${ROOT_DIR}/utils/first_analysis.py "${RESULT_FOLDER}/first_analysis.py"
cp ${ROOT_DIR}/utils/util.py "${RESULT_FOLDER}/util.py"


cp ${ROOT_DIR}/utils/packing_initialization.py "${RESULT_FOLDER}/packing_initialization.py"
cp run.sh "${RESULT_FOLDER}/run.sh"
cp run.py "${RESULT_FOLDER}/run.py"

cp options.yml "${RESULT_FOLDER}/options.yml"

# If a second argument is provided ("periodic" or "nonperiodic"),
# record it in a small marker file for bookkeeping.
if [ -n "${PERIODIC_FLAG}" ]; then
    echo "mode=${PERIODIC_FLAG}" > "${RESULT_FOLDER}/mode.info"
fi

cp run.py "${ROOT_DIR}/logs/run_logs/run_${STAMP}_RUN_${NOTE}.py"



# study-specific files
# cp -r examples/rod_packing_case/*.txt ${RESULT_FOLDER}



echo "#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -c 1                # Number of threads per core (-c)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_compute     # Partition to submit to
#SBATCH --mem=3000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e errors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=jung@seas.harvard.edu


module load python
mamba activate mujoco-env
# mamba install -y mujoco numpy mujoco-python-viewer scipy matplotlib yaml


time /n/home01/yjung/.conda/envs/mujoco-env/bin/python  perturb_rod_packings.py
time /n/home01/yjung/.conda/envs/mujoco-env/bin/python  first_analysis.py
" > "${RESULT_FOLDER}/Sbatch.sh"

cd "${RESULT_FOLDER}"
# execute sbatch file
sbatch Sbatch.sh

