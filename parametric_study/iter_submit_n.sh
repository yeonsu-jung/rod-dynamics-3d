#!/bin/bash
# iter_submit_n.sh
#
# Iterates over N folders in relaxation_3rd_multithreading and submits CPU dynamics batches.
# For large-N GPU runs, use iter_submit_n_gpu.sh.

# Configuration
INPUT_BASE="/n/home01/yjung/Github/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading"
# INPUT_BASE="/n/holylabs/mahadevan_lab/Users/yjung/relaxation/relaxation_large_N_gpuEntangle_cpuRelax"



OUTPUT_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/"
STEPS=200000
DT=0.0005
FRICTIONS="0.0,0.05,0.1,0.15,0.2,0.4,1.0"
KICK=0.1
WSPEED=0.2
LIMIT=5
STRIDE=1000
DRY_RUN=${DRY_RUN:-false}

echo "Scanning $INPUT_BASE..."

for dir in "$INPUT_BASE"/N*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir") # e.g., N200
        
        # Extract N (remove 'N' prefix)
        N=${dirname#N}
        
        # Check if N is a number
        if ! [[ "$N" =~ ^[0-9]+$ ]]; then
            echo "Skipping $dirname (cannot parse N)"
            continue
        fi

        # JOB_NAME="relax2nd_${dirname}_sweep"
        JOB_NAME="dynamics_cpu_${dirname}_sweep"
        
        echo "---------------------------------------------------"
        echo "Submitting batch for N=$N (Folder: $dirname) ALL ARs"
        echo "Job Name: $JOB_NAME"
        echo "Steps: $STEPS, dt: $DT"

        # Copy this file to the batch directory at OUTPUT_BASE/dir
        mkdir -p "$OUTPUT_BASE/$dirname"
        cp "$0" "$OUTPUT_BASE/$dirname/submit_entangled.sh"
        
        
        DRY_RUN_ARG=""
        if [ "$DRY_RUN" = true ]; then
            DRY_RUN_ARG="--dry-run"
        fi

        python3 parametric_study/submit_entangled.py \
            --n-rods "$N" \
            --input-root "$dir" \
            --job-name "$JOB_NAME" \
            --frictions "$FRICTIONS" \
            --init-velocity-sigma "$KICK" \
            --w-speed "$WSPEED" \
            --steps "$STEPS" \
            --dt "$DT" \
            --seed-limit "$LIMIT" \
            --output-stride "$STRIDE" \
            --perrod-stride 0 \
            --no-network \
            --no-csv \
            --ent-period "$STEPS" \
            $DRY_RUN_ARG

    fi
done

echo "Done iterating."
