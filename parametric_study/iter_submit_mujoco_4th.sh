#!/bin/bash
# iter_submit_mujoco_4th.sh
# 
# 4th iteration with fixed random seeds and adaptive timesteps
# - N <= 200: dt=0.0005, steps=200000 (total time = 100s)
# - N = 500: dt=0.00025, steps=400000 (total time = 100s)
# - N = 1000: dt=0.0001, steps=1000000 (total time = 100s)

# Configuration
INPUT_BASE="/n/home01/yjung/Github/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading"
OUTPUT_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs_mujoco"
ITERATION_TITLE="relaxation_3rd_multithreading_4th_iterated_runs"

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

        # Adaptive timestep and steps based on N
        if [ "$N" -le 200 ]; then
            DT=0.0005
            STEPS=200000
        elif [ "$N" -eq 500 ]; then
            DT=0.00025
            STEPS=400000
        elif [ "$N" -eq 1000 ]; then
            DT=0.0001
            STEPS=1000000
        else
            DT=0.0005
            STEPS=200000
        fi

        JOB_NAME="mujoco_${dirname}_sweep_4th"
        
        echo "---------------------------------------------------"
        echo "Submitting MuJoCo batch for N=$N (Folder: $dirname)"
        echo "Job Name: $JOB_NAME"
        echo "Steps: $STEPS, dt: $DT (total time: $(echo "$STEPS * $DT" | bc)s)"

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
done

echo "Done iterating."
