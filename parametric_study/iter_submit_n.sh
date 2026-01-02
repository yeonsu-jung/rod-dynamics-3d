#!/bin/bash
# iter_submit_n.sh
# 
# Iterates over N folders in initial-configs/relaxation_2nd/ and submits batches.

# Configuration
INPUT_BASE="/n/home01/yjung/Github/rod-dynamics-3d/initial-configs/relaxation_2nd"
STEPS=200000
DT=0.0005
FRICTIONS="1"
# FRICTIONS="1"
KICK=0.1
WAVE_WIDTH=200
WAVE_PERIOD=1000
LIMIT=1
STRIDE=1000

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

        # Filter for N=500 and N=1000 per user request
        if [ "$N" -ne 500 ] && [ "$N" -ne 1000 ]; then
             echo "Skipping N=$N"
             continue
        fi

        JOB_NAME="relax2nd_${dirname}_sweep"
        
        echo "---------------------------------------------------"
        echo "Submitting batch for N=$N (Folder: $dirname)"
        echo "Job Name: $JOB_NAME"
        echo "Steps: $STEPS, dt: $DT"
        
        python3 parametric_study/submit_entangled.py \
            --n-rods "$N" \
            --input-root "$dir" \
            --job-name "$JOB_NAME" \
            --frictions "$FRICTIONS" \
            --init-velocity-sigma "$KICK" \
            --steps "$STEPS" \
            --dt "$DT" \
            --seed-limit "$LIMIT" \
            --output-stride "$STRIDE" \
            --network-wave-width "$WAVE_WIDTH" \
            --network-wave-period "$WAVE_PERIOD"

    fi
done

echo "Done iterating."
