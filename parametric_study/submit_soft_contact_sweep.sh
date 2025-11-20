#!/bin/bash
#SBATCH --job-name=soft_contact_sweep
#SBATCH --output=soft_contact_sweep_%j.out
#SBATCH --error=soft_contact_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Soft contact model parametric sweep submission script
# Sweeps over friction coefficients and noise amplitudes

echo "Starting soft contact parametric sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Started at: $(date)"
echo "----------------------------------------"

# Navigate to repository
cd /n/home01/yjung/Github/rod-dynamics-3d

# Check if executable exists
if [ ! -f "build/rigidbody_viewer_3d" ]; then
    echo "ERROR: Executable not found. Building..."
    cd build
    cmake -DBUILD_HEADLESS=ON .. && make -j8
    cd ..
fi

# Run the parametric sweep
python3 parametric_study/sweep_soft_contact.py

EXIT_CODE=$?

echo "----------------------------------------"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
