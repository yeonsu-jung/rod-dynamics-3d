#!/bin/bash
# iter_submit_n_single.sh
#
# Schedules one bundled Slurm job per N-value.
# Each bundled job runs the same subprocess commands that the array flow would
# have launched, but inside a single allocation. Core counts follow the same
# dynamic N-based policy as iter_submit_n_array.sh.

# Configuration
INPUT_BASE=${INPUT_BASE:-"/n/home01/yjung/Github/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading"}
OUTPUT_BASE=${OUTPUT_BASE:-"/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/dynamics_nsc_single"}
# STEPS=${STEPS:-200000}
# DT=${DT:-0.0005}
STEPS=${STEPS:-100000}
DT=${DT:-0.001}
FRICTIONS=${FRICTIONS:-"0.0,0.05,0.1,0.15,0.2,0.4,1.0"}
SIGMA_V=${SIGMA_V:-0.1}
SIGMA_W=${SIGMA_W:-}
LIMIT=${LIMIT:-0}
STRIDE=${STRIDE:-1000}
MAX_PARALLEL_SUBPROCESSES=${MAX_PARALLEL_SUBPROCESSES:-1}
DRY_RUN=${DRY_RUN:-false}

# Optional filters for focused debugging via environment variables.
# Example:
#   export N_FILTERS="200"
#   export SEED_FILTERS="945,12,381"
#   export AR_FILTERS="200"
#   export DRY_RUN=true
N_FILTERS_CSV=${N_FILTERS:-""}
SEED_FILTERS_CSV=${SEED_FILTERS:-""}
AR_FILTERS_CSV=${AR_FILTERS:-""}

N_FILTERS=()
SEED_FILTERS=()
AR_FILTERS=()

if [ -n "$N_FILTERS_CSV" ]; then
    IFS=';' read -r -a N_FILTERS <<< "$N_FILTERS_CSV"
fi

if [ -n "$SEED_FILTERS_CSV" ]; then
    IFS=';' read -r -a SEED_FILTERS <<< "$SEED_FILTERS_CSV"
fi

if [ -n "$AR_FILTERS_CSV" ]; then
    IFS=',' read -r -a AR_FILTERS <<< "$AR_FILTERS_CSV"
fi

# NSC solver parameters
NSC_ITERS=40
NSC_BETA=0.2
NSC_POS_ITERS=5
NSC_POS_PSOR=50

echo "Scanning $INPUT_BASE... (NSC / thermal init / Single Job Mode)"

for dir in "$INPUT_BASE"/N*; do
    if [ -d "$dir" ]; then
        dirname=$(basename "$dir")
        N=${dirname#N}

        if ! [[ "$N" =~ ^[0-9]+$ ]]; then
            continue
        fi

        if [ ${#N_FILTERS[@]} -gt 0 ]; then
            MATCHED_N=false
            for selected_n in "${N_FILTERS[@]}"; do
                if [ "$N" = "$selected_n" ]; then
                    MATCHED_N=true
                    break
                fi
            done
            if [ "$MATCHED_N" = false ]; then
                continue
            fi
        fi

        # Dynamically set CPU cores based on N
        if [ "$N" -lt 100 ]; then
            THREADS=4
        elif [ "$N" -le 200 ]; then
            THREADS=8
        elif [ "$N" -lt 1000 ]; then
            THREADS=16
        else
            THREADS=32
        fi

        JOB_NAME="dynamics_nsc_single_${dirname}_sweep"

        echo "---------------------------------------------------"
        TOTAL_CORES=$((THREADS * MAX_PARALLEL_SUBPROCESSES))
        echo "Submitting NSC thermal Single Job for N=$N | Cores per subprocess=$THREADS | Max parallel subprocesses=$MAX_PARALLEL_SUBPROCESSES | Total allocated cores=$TOTAL_CORES"
        echo "Job Name: $JOB_NAME"

        mkdir -p "$OUTPUT_BASE"

        CMD=(
            python3 parametric_study/submit_entangled_single_job.py
            --n-rods "$N"
            --input-root "$dir"
            --job-name "$JOB_NAME"
            --runs-root "$OUTPUT_BASE"
            --frictions "$FRICTIONS"
            --sigma-v "$SIGMA_V"
            --steps "$STEPS"
            --dt "$DT"
            --seed-limit "$LIMIT"
            --output-stride "$STRIDE"
            --perrod-stride 0
            --no-network
            --no-csv
            --ent-period "$STEPS"
            --nsc
            --nsc-iters "$NSC_ITERS"
            --nsc-beta "$NSC_BETA"
            --nsc-pos-iters "$NSC_POS_ITERS"
            --nsc-pos-psor "$NSC_POS_PSOR"
            --threads "$THREADS"
            --max-parallel-subprocesses "$MAX_PARALLEL_SUBPROCESSES"
        )

        if [ -n "$SIGMA_W" ]; then
            CMD+=(--sigma-w "$SIGMA_W")
        fi

        if [ "$DRY_RUN" = true ]; then
            CMD+=(--dry-run)
        fi

        if [ ${#SEED_FILTERS[@]} -gt 0 ]; then
            for seed in "${SEED_FILTERS[@]}"; do
                CMD+=(--seed-filter "$seed")
            done
        fi

        if [ ${#AR_FILTERS[@]} -gt 0 ]; then
            CMD+=(--ar)
            for ar in "${AR_FILTERS[@]}"; do
                CMD+=("$ar")
            done
        fi

        "${CMD[@]}"
    fi
done

echo "Done generating bundled single-job sweeps."