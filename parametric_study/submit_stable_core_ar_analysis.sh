#!/bin/bash
#SBATCH -p sapphire
#SBATCH -J stable_core_ar
#SBATCH -t 0-04:00:00
#SBATCH --mem=32G
#SBATCH -c 1
#SBATCH -o stable_core_ar_%j.out
#SBATCH -e stable_core_ar_%j.err
#SBATCH --mail-type=END

# Batch stable core vs AR analysis for a given N value
# Usage: sbatch parametric_study/submit_stable_core_ar_analysis.sh <input_dir> <N> <output_base>

module load python
mamba activate mujoco-env

INPUT_DIR="$1"
N_VALUE="$2"
OUTPUT_BASE="$3"

if [ -z "$INPUT_DIR" ] || [ -z "$N_VALUE" ] || [ -z "$OUTPUT_BASE" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch submit_stable_core_ar_analysis.sh <input_dir> <N> <output_base>"
    exit 1
fi

OUTPUT_DIR="${OUTPUT_BASE}/N${N_VALUE}"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Starting Stable Core vs AR Analysis for N=$N_VALUE"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Find all endpoints.csv files
RUNS=$(find "$INPUT_DIR" -name "endpoints.csv" | sort)
NUM_RUNS=$(echo "$RUNS" | wc -l)

echo "Found $NUM_RUNS runs with endpoints.csv"
echo ""

# Group runs by AR value and mu, analyze each combination separately
# Extract unique AR and mu values from directory names
AR_MU_PAIRS=$(find "$INPUT_DIR" -name "endpoints.csv" -exec dirname {} \; | \
    xargs -I {} basename {} | \
    grep -oP 'AR\K[0-9]+_Friction[\d.]+' | \
    sed 's/_Friction/_/' | sort -u)

# Alternative: extract AR and mu separately
AR_VALUES=$(find "$INPUT_DIR" -name "endpoints.csv" -exec dirname {} \; | \
    xargs -I {} basename {} | \
    grep -oP 'AR\K[0-9]+' | sort -u -n)

MU_VALUES=$(find "$INPUT_DIR" -name "endpoints.csv" -exec dirname {} \; | \
    xargs -I {} basename {} | \
    grep -oP 'Friction\K[\d.]+' | sort -u -n)

echo "Found AR values: $AR_VALUES"
echo "Found mu values: $MU_VALUES"
echo ""

# For each combination of AR and mu, collect runs and analyze
for AR in $AR_VALUES; do
    for MU in $MU_VALUES; do
        echo "Processing AR=$AR, mu=$MU..."
        
        # Find all runs for this AR and mu combination
        AR_MU_RUNS=$(find "$INPUT_DIR" -name "endpoints.csv" | grep "AR${AR}_" | grep "Friction${MU}_")
        
        if [ -z "$AR_MU_RUNS" ]; then
            echo "  No runs found for AR=$AR, mu=$MU"
            continue
        fi
        
        # Count runs
        NUM_AR_MU_RUNS=$(echo "$AR_MU_RUNS" | wc -l)
        echo "  Found $NUM_AR_MU_RUNS runs"
        
        # Collect run directories
        RUN_DIRS=""
        for ENDPOINTS_FILE in $AR_MU_RUNS; do
            RUN_DIR=$(dirname "$ENDPOINTS_FILE")
            RUN_DIRS="$RUN_DIRS $RUN_DIR"
        done
        
        # Run analysis for this AR and mu combination
        python3 study/analyze_stable_core_vs_n.py \
            $RUN_DIRS \
            --n-value "$N_VALUE" \
            --output "${OUTPUT_DIR}/stable_core_N${N_VALUE}_AR${AR}_mu${MU}.csv"
        
        echo "  Saved results for AR=$AR, mu=$MU"
    done
done

# Combine all results into one file with AR and mu columns
echo ""
echo "Combining results..."
python3 -c "
import pandas as pd
import glob
import os

output_dir = '${OUTPUT_DIR}'
csv_files = glob.glob(os.path.join(output_dir, 'stable_core_N${N_VALUE}_AR*_mu*.csv'))

if csv_files:
    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sort by mu, then AR
    if 'mu' in combined.columns and 'AR' in combined.columns:
        combined = combined.sort_values(['mu', 'AR'])
    elif 'AR' in combined.columns:
        combined = combined.sort_values('AR')
    
    combined.to_csv(os.path.join(output_dir, 'stable_core_vs_ar_N${N_VALUE}_all.csv'), index=False)
    print(f'Combined {len(csv_files)} files into stable_core_vs_ar_N${N_VALUE}_all.csv')
    print(f'Total rows: {len(combined)}')
    
    # Also create per-mu summary files
    if 'mu' in combined.columns:
        for mu in combined['mu'].unique():
            mu_data = combined[combined['mu'] == mu]
            mu_data.to_csv(os.path.join(output_dir, f'stable_core_vs_ar_N${N_VALUE}_mu{mu}.csv'), index=False)
            print(f'Created stable_core_vs_ar_N${N_VALUE}_mu{mu}.csv with {len(mu_data)} rows')
else:
    print('No files found to combine')
"

echo ""
echo "=========================================="
echo "AR analysis complete!"
echo "Results saved to: ${OUTPUT_DIR}/stable_core_vs_ar_N${N_VALUE}.csv"
echo "=========================================="
