#!/bin/bash
# iter_submit_n_nsc_thermal.sh
#
# Like iter_submit_n_nsc.sh but uses thermal (Maxwell-Boltzmann) randomInit.
# Instead of separate vSigma/wSpeed, a single --sigma-v sets the translational
# velocity scale; angular velocity follows from equipartition:
#   sigma_w = sqrt(12) * sigma_v / L  (~0.35 rad/s for sigma_v=0.1, L=1)
#
# The engine computes kBT = m * sigma_v^2 per-AR so all rods get the same
# velocity scale regardless of diameter.

# Configuration
INPUT_BASE="/n/home01/yjung/Github/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading"
OUTPUT_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/test_nsc_thermal"
STEPS=200000
DT=0.0005
FRICTIONS="0.0,0.05,0.1,0.15,0.2,0.4,1.0"
SIGMA_V=0.1       # translational velocity scale ~ 0.1 * rod_length
LIMIT=5
STRIDE=1000
DRY_RUN=${DRY_RUN:-false}

# NSC solver parameters
NSC_ITERS=40
NSC_BETA=0.2
NSC_POS_ITERS=5
NSC_POS_PSOR=50

echo "Scanning $INPUT_BASE... (NSC / thermal init, sigma_v=$SIGMA_V)"

for dir in "$INPUT_BASE"/N*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir") # e.g., N200

        # Extract N (remove 'N' prefix)
        N=${dirname#N}

        # Uncomment to filter specific N values:
        # if ! [[ "$N" =~ ^(100|200|300)$ ]]; then
        #     echo "Skipping $dirname"
        #     continue
        # fi

        # Check if N is a number
        if ! [[ "$N" =~ ^[0-9]+$ ]]; then
            echo "Skipping $dirname (cannot parse N)"
            continue
        fi

        JOB_NAME="dynamics_nsc_thermal_v2_${dirname}_sweep"

        echo "---------------------------------------------------"
        echo "Submitting NSC thermal batch for N=$N (Folder: $dirname) ALL ARs"
        echo "Job Name: $JOB_NAME"
        echo "Steps: $STEPS, dt: $DT, sigma_v: $SIGMA_V"

        # Copy this file to the batch directory
        mkdir -p "$OUTPUT_BASE/$dirname"
        cp "$0" "$OUTPUT_BASE/$dirname/submit_thermal.sh"

        DRY_RUN_ARG=""
        if [ "$DRY_RUN" = true ]; then
            DRY_RUN_ARG="--dry-run"
        fi

        python3 parametric_study/submit_entangled.py \
            --n-rods "$N" \
            --input-root "$dir" \
            --job-name "$JOB_NAME" \
            --frictions "$FRICTIONS" \
            --sigma-v "$SIGMA_V" \
            --steps "$STEPS" \
            --dt "$DT" \
            --seed-limit "$LIMIT" \
            --output-stride "$STRIDE" \
            --perrod-stride 0 \
            --no-network \
            --no-csv \
            --ent-period "$STEPS" \
            --nsc \
            --nsc-iters "$NSC_ITERS" \
            --nsc-beta "$NSC_BETA" \
            --nsc-pos-iters "$NSC_POS_ITERS" \
            --nsc-pos-psor "$NSC_POS_PSOR" \
            $DRY_RUN_ARG

    fi
done

echo "Done iterating (NSC thermal)."
