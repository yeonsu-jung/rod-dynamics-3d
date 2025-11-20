#!/bin/bash
# Run parametric sweep for pairwise metrics study
# n ∈ {6, 7, 8}, AR ∈ {50, 150, 500}

set -e

# Directories
BUILD_DIR="../build"
SCENE_DIR="../assets/scenes/pairwise_sweep"
OUTPUT_DIR="pairwise_sweep_csvs"
ANALYSIS_DIR="pairwise_analysis"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if build exists
if [ ! -f "$BUILD_DIR/rigidbody_viewer_3d" ]; then
    echo "Error: Executable not found at $BUILD_DIR/rigidbody_viewer_3d"
    echo "Please build the project first:"
    echo "  cd ../build && cmake .. && make"
    exit 1
fi

# Generate scene files
echo "=== Generating scene files ==="
python3 generate_pairwise_scenes.py

# Run simulations
echo ""
echo "=== Running simulations ==="

TOTAL_STEPS=5000
MAX_FRAMES=1000

for scene in "$SCENE_DIR"/*.json; do
    if [ ! -f "$scene" ]; then
        echo "No scenes found in $SCENE_DIR"
        exit 1
    fi
    
    name=$(basename "$scene" .json)
    csv_path="$OUTPUT_DIR/${name}.csv"
    
    echo "Running $name..."
    
    cd "$BUILD_DIR"
    ./rigidbody_viewer_3d \
        --scene "$scene" \
        --headless "$TOTAL_STEPS" \
        --perrod "../parametric_study/$csv_path" \
        --perrod-max "$MAX_FRAMES"
    cd - > /dev/null
    
    echo "  Output: $csv_path"
done

echo ""
echo "=== Simulations complete ==="
echo "CSV files saved to $OUTPUT_DIR/"
echo ""
echo "To analyze results:"
echo "  python3 pairwise_metrics.py --sweep $OUTPUT_DIR"
