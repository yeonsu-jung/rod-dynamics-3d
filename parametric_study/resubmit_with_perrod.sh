#!/bin/bash
# resubmit_with_perrod.sh
#
# Re-submits specific dynamics runs with per-rod trajectory output (perrod.csv)
# for video generation.
#
# Parameters:
#   N  = [20, 100, 200]
#   AR = [25, 100, 300]
#   μ  = [0.0, 0.1, 1.0]
#
# This re-uses the existing run directories (scene.json, x_relaxed.txt, binary)
# and just modifies the Sbatch.sh to include --perrod-stride and --perrod flags.

RUNS_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs"
DRY_RUN=${DRY_RUN:-false}

# Parameters to sweep
declare -a N_VALUES=("N20" "N100" "N200")
declare -a AR_VALUES=("AR25" "AR100" "AR300")
declare -a MU_VALUES=("Friction0.0_" "Friction0.1_" "Friction1.0_")

PERROD_STRIDE=1000
STEPS=200000
DT=0.0005
CPUS=8

submitted=0
skipped=0
failed=0

for N in "${N_VALUES[@]}"; do
    SWEEP_DIR="${RUNS_BASE}/dynamics_cpu_${N}_sweep"
    
    if [ ! -d "$SWEEP_DIR" ]; then
        echo "WARNING: Sweep directory not found: $SWEEP_DIR"
        continue
    fi
    
    for AR in "${AR_VALUES[@]}"; do
        for MU in "${MU_VALUES[@]}"; do
            # Find matching run directory
            RUN_DIR=$(ls -d "${SWEEP_DIR}"/*_${AR}_${MU}* 2>/dev/null | head -1)
            
            if [ -z "$RUN_DIR" ]; then
                echo "NOT FOUND: ${N} ${AR} ${MU}"
                ((failed++))
                continue
            fi
            
            # Skip if perrod.csv already exists
            if [ -f "${RUN_DIR}/perrod.csv" ]; then
                echo "SKIP (perrod.csv exists): $(basename $RUN_DIR)"
                ((skipped++))
                continue
            fi
            
            # Check required files exist
            if [ ! -f "${RUN_DIR}/rigidbody_viewer_3d" ] || [ ! -f "${RUN_DIR}/scene.json" ] || [ ! -f "${RUN_DIR}/x_relaxed.txt" ]; then
                echo "ERROR: Missing required files in ${RUN_DIR}"
                ((failed++))
                continue
            fi
            
            # Extract N number from sweep dir name
            N_NUM=${N#N}
            
            # Build the simulation command with perrod enabled
            SIM_CMD="./rigidbody_viewer_3d --headless --scene scene.json --init-csv x_relaxed.txt --output output.csv --steps ${STEPS} --dt ${DT} --threads ${CPUS} --no-csv --output-stride ${PERROD_STRIDE} --perrod-stride ${PERROD_STRIDE} --perrod perrod.csv --entanglement --entanglement-period ${STEPS} --entanglement-threads 0 --entanglement-cutoff 1000.0"
            
            # Write new Sbatch script
            cat > "${RUN_DIR}/Sbatch_perrod.sh" << EOF
#!/bin/bash
#SBATCH -n 1
#SBATCH -c ${CPUS}
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=1G
#SBATCH -o output_perrod_%j.out
#SBATCH -e errors_perrod_%j.err
#SBATCH --mail-type=END
#SBATCH --job-name=perrod_${N}_${AR}

set -euo pipefail
module load python

echo "======================================"
echo "Entangled ${N} dynamics (with perrod)"
echo "AR: ${AR#AR}"
echo "Friction: ${MU}"
echo "PWD: \$(pwd)"
echo "======================================"

# Remove old output.csv so simulation re-runs
rm -f output.csv

echo "Running simulation..."
echo "${SIM_CMD}"
${SIM_CMD}

echo ""
echo "Job complete."
EOF
            
            if [ "$DRY_RUN" = true ]; then
                echo "DRY RUN: ${N} ${AR} ${MU} -> $(basename $RUN_DIR)"
            else
                echo "SUBMIT: ${N} ${AR} ${MU} -> $(basename $RUN_DIR)"
                result=$(sbatch --chdir="${RUN_DIR}" "${RUN_DIR}/Sbatch_perrod.sh" 2>&1)
                if [ $? -ne 0 ]; then
                    echo "  FAILED: $result"
                    ((failed++))
                else
                    echo "  $result"
                    ((submitted++))
                fi
            fi
        done
    done
done

echo ""
echo "=================================="
echo "Submitted: $submitted"
echo "Skipped:   $skipped"
echo "Failed:    $failed"
echo "=================================="
