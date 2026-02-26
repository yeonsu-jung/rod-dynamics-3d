#!/bin/bash
# submit_control_datasets.sh
# 
# Submits jobs for control datasets in initial-configs/control_datasets/

# Configuration
INPUT_BASE="/n/home01/yjung/Github/rod-dynamics-3d/initial-configs/control_datasets"
# OUTPUT_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/"
STEPS=200000
DT=0.0005
FRICTIONS="1.0" # As requested or standard? Original script used "1.0" or "0.0,0.05..."
KICK=0.1
# WAVE_WIDTH=200
# WAVE_PERIOD=1000
LIMIT=0 # 0 for all
STRIDE=1000
DRY_RUN=false
JOB_NAME="control_run"

echo "Submitting control datasets from $INPUT_BASE..."

DRY_RUN_ARG=""
if [ "$DRY_RUN" = true ]; then
    DRY_RUN_ARG="--dry-run"
fi

python3 parametric_study/submit_control.py \
    --input-root "$INPUT_BASE" \
    --job-name "$JOB_NAME" \
    --frictions "$FRICTIONS" \
    --init-velocity-sigma "$KICK" \
    --steps "$STEPS" \
    --dt "$DT" \
    --limit "$LIMIT" \
    --output-stride "$STRIDE" \
    $DRY_RUN_ARG

echo "Done."
