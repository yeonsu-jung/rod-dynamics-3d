#!/bin/bash
# iter_submit_mujoco.sh
# 
# Iterates over N folders in initial-configs/relaxation_3rd_multithreading and submits MuJoCo batches.

# Configuration
INPUT_BASE="/n/home01/yjung/Github/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading"
OUTPUT_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs_mujoco"
ITERATION_TITLE="relaxation_3rd_multithreading_3rd_iterated_runs"

STEPS=200000
DT=0.0005
FRICTIONS="0.0,0.05,0.1,0.15,0.2,0.4,1.0"
KICK=0.1
LIMIT=1 # 0 = unlimited
STRIDE=100
DRY_RUN=false

echo "Scanning $INPUT_BASE..."

for dir in "$INPUT_BASE"/N*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir") # e.g., N200
        N=${dirname#N}
        
        if ! [[ "$N" =~ ^[0-9]+$ ]]; then
            echo "Skipping $dirname (cannot parse N)"
            continue
        fi

        # if N != 1000, skip
        # if [ "$N" -eq 100 ]; then
        # if N == 500 or 200, skip
        # if [ "$N" -eq 500 ] || [ "$N" -eq 200 ]; then
        #     echo "Skipping $dirname (N=$N)"
        #     continue
        # fi
        

        JOB_NAME="mujoco_${dirname}_sweep"
        
        echo "---------------------------------------------------"
        echo "Submitting MuJoCo batch for N=$N (Folder: $dirname)"
        echo "Job Name: $JOB_NAME"
        echo "Steps: $STEPS, dt: $DT"

        mkdir -p "$OUTPUT_BASE/$ITERATION_TITLE/$dirname"
        cp "$0" "$OUTPUT_BASE/$ITERATION_TITLE/$dirname/iter_submit.sh"
        
        DRY_RUN_ARG=""
        if [ "$DRY_RUN" = true ]; then
            DRY_RUN_ARG="--dry-run"
        fi

        python3 parametric_study/submit_mujoco.py \
            --n-rods "$N" \
            --input-root "$dir" \
            --runs-root "$OUTPUT_BASE" \
            --iteration-title "$ITERATION_TITLE" \
            --job-name "$JOB_NAME" \
            --frictions "$FRICTIONS" \
            --init-velocity-sigma "$KICK" \
            --steps "$STEPS" \
            --dt "$DT" \
            --limit "$LIMIT" \
            --stride "$STRIDE" \
            $DRY_RUN_ARG
    fi
    # break
done

echo "Done iterating."
