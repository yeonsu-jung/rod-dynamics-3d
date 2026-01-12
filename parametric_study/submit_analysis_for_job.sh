#!/bin/bash
# Submit entanglement analysis for all seeds in a hierarchical job folder.
# Structure: JOB_DIR/N{n}/{seed}/...

JOB_DIR="$1"

if [ -z "$JOB_DIR" ]; then
    echo "Usage: $0 <JOB_DIR>"
    exit 1
fi

if [ ! -d "$JOB_DIR" ]; then
    echo "Error: $JOB_DIR is not a directory."
    exit 1
fi

echo "Scanning $JOB_DIR for seed directories..."

# Find all directory levels that look like seeds (children of N*)
# Strategy: Look for N* folders, then iterate their children.
for n_dir in "$JOB_DIR"/N*; do
    if [ -d "$n_dir" ]; then
        echo "Found N-folder: $n_dir"
        
        for seed_dir in "$n_dir"/*; do
            if [ -d "$seed_dir" ]; then
                echo "Submitting analysis for seed: $seed_dir"
                
                # Check if it has runs (subdirs)
                # Count subdirs
                count=$(find "$seed_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
                
                if [ "$count" -gt 0 ]; then
                    # Submit
                    # We utilize the existing SBATCH script.
                    # We pass the absolute path of the seed dir.
                    abs_seed_path=$(realpath "$seed_dir")
                    
                    # We assume the sbatch script is in the same dir as this (parametric_study)
                    SCRIPT_DIR=$(dirname "$0")
                    sbatch "$SCRIPT_DIR/submit_mujoco_entanglement_analysis.sh" "$abs_seed_path"
                else
                    echo "  No runs found in $seed_dir, skipping."
                fi
            fi
        done
    fi
done
