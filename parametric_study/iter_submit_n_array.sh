#!/bin/bash
# iter_submit_n_array.sh
# 
# Schedules a huge batch of sims using ONE Slurm Array job per N-value.
# Core counts are adjusted dynamically by N to optimize scheduler packing.

# Configuration
INPUT_BASE="/n/home01/yjung/Github/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading"
OUTPUT_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/dynamics_nsc_array"
STEPS=200000
DT=0.0005
FRICTIONS="0.0,0.05,0.1,0.15,0.2,0.4,1.0"
SIGMA_V=0.1
LIMIT=0 # Submit all inside the seed limit. Set to 0.
STRIDE=1000
DRY_RUN=${DRY_RUN:-false}

# NSC solver parameters
NSC_ITERS=40
NSC_BETA=0.2
NSC_POS_ITERS=5
NSC_POS_PSOR=50

echo "Scanning $INPUT_BASE... (NSC / thermal init / Array Mode)"

for dir in "$INPUT_BASE"/N*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")

        N=${dirname#N}

        if ! [[ "$N" =~ ^[0-9]+$ ]]; then
            continue
        fi

        # Skip N=200 since it was already submitted
        if [ "$N" -eq 200 ]; then
            continue
        fi

        # Dynamically set CPU cores based on N
        if [ "$N" -lt 100 ]; then
            THREADS=4
        elif [ "$N" -le 200 ]; then
            THREADS=8
        elif [ "$N" -lt 1000 ]; then
            THREADS=16  # For N=500 and similar
        else
            THREADS=32  # For N=1000+
        fi

        JOB_NAME="dynamics_nsc_array_${dirname}_sweep"

        echo "---------------------------------------------------"
        echo "Submitting NSC thermal Array Job for N=$N | Cores=$THREADS"
        echo "Job Name: $JOB_NAME"
        
        mkdir -p "$OUTPUT_BASE"
        
        DRY_RUN_ARG=""
        if [ "$DRY_RUN" = true ]; then
            DRY_RUN_ARG="--dry-run"
        fi

        python3 parametric_study/submit_entangled_array.py \
            --n-rods "$N" \
            --input-root "$dir" \
            --job-name "$JOB_NAME" \
            --runs-root "$OUTPUT_BASE" \
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
            --threads "$THREADS" \
            $DRY_RUN_ARG
    fi
done

echo "Done generating Array sweeps."
