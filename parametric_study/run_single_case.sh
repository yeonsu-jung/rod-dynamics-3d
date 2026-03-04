#!/bin/bash
# run_single_case.sh
#
# Create and (optionally) submit a single simulation case with full control
# over delta, friction, mode (gpu/cpu), and initial condition.
#
# Usage:
#   ./parametric_study/run_single_case.sh [options]
#
# Options:
#   --n        N           Number of rods            (default: 2000)
#   --ar       AR          Aspect ratio              (default: 1000)
#   --friction F           Friction coefficient      (default: 1.0)
#   --delta    D           Contact activation margin (default: 0.002)
#   --seed     SEED        Seed folder, e.g. "224,259,311" (default: first found)
#   --steps    S           Simulation steps          (default: 200000)
#   --dt       DT          Timestep                  (default: 0.0005)
#   --mode     gpu|cpu     Use GPU two-pass or CPU   (default: gpu)
#   --dry-run              Print config but don't submit
#   --k-scaler K           Spring stiffness scaler   (default: 100.0)
#
# Examples:
#   # GPU run with small delta to fix tunneling:
#   ./parametric_study/run_single_case.sh --n 2000 --ar 1000 --friction 1.0 --delta 0.002
#
#   # CPU reference with same delta for comparison:
#   ./parametric_study/run_single_case.sh --n 2000 --ar 1000 --friction 1.0 --delta 0.002 --mode cpu
#
#   # Dry run to inspect generated files:
#   ./parametric_study/run_single_case.sh --n 2000 --ar 1000 --delta 0.002 --dry-run

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
N=2000
AR=1000
FRICTION=1.0
DELTA=0.002
SEED=""
STEPS=200000
DT=0.0005
MODE="gpu"
DRY_RUN=false
K_SCALER=100.0

REPO=/n/home01/yjung/Github/rod-dynamics-3d
INPUT_BASE=/n/holylabs/mahadevan_lab/Users/yjung/relaxation/relaxation_large_N_gpuEntangle_cpuRelax
OUTPUT_BASE=/n/holylabs/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --n)        N="$2";        shift 2 ;;
        --ar)       AR="$2";       shift 2 ;;
        --friction) FRICTION="$2"; shift 2 ;;
        --delta)    DELTA="$2";    shift 2 ;;
        --seed)     SEED="$2";     shift 2 ;;
        --steps)    STEPS="$2";    shift 2 ;;
        --dt)       DT="$2";       shift 2 ;;
        --mode)     MODE="$2";     shift 2 ;;
        --k-scaler) K_SCALER="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true;  shift   ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Find seed folder
# ---------------------------------------------------------------------------
N_DIR="$INPUT_BASE/N${N}"
if [ ! -d "$N_DIR" ]; then
    echo "ERROR: Input directory not found: $N_DIR"
    exit 1
fi

if [ -z "$SEED" ]; then
    SEED=$(ls "$N_DIR" | head -1)
    echo "No --seed given; using first found: $SEED"
fi

INIT_CSV="$N_DIR/$SEED/x_relaxed_AR${AR}.txt"
if [ ! -f "$INIT_CSV" ]; then
    echo "ERROR: Initial CSV not found: $INIT_CSV"
    echo "Available AR files in $N_DIR/$SEED/:"
    ls "$N_DIR/$SEED/" 2>/dev/null || echo "  (directory not found)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Choose binary and SLURM settings
# ---------------------------------------------------------------------------
SEED_SLUG="${SEED//,/_}"   # "224,259,311" → "224_259_311"

if [ "$MODE" = "gpu" ]; then
    BIN="$REPO/build_cuda/rigidbody_viewer_3d"
    PARTITION="gpu_test"
    GRES="#SBATCH --gres=gpu:1"
    MEM="8G"
    THREADS=4
    USE_CUDA="true"
    MODE_LABEL="cuda"
    MODULES="module load cmake\nmodule load gcc/13.2.0-fasrc01\nmodule load cuda/12.9.1-fasrc01"
else
    BIN="$REPO/build_head/rigidbody_viewer_3d"
    PARTITION="seas_compute"
    GRES=""
    MEM="4G"
    THREADS=8
    USE_CUDA="false"
    MODE_LABEL="cpu"
    MODULES="module load gcc/13.2.0-fasrc01"
fi

JOB_NAME="${MODE_LABEL}_N${N}_AR${AR}_F${FRICTION}_d${DELTA}_${SEED_SLUG}"

# Run directory
RUNDIR="$OUTPUT_BASE/single_cases/${MODE_LABEL}_N${N}_AR${AR}_F${FRICTION}_delta${DELTA}/${SEED_SLUG}"
mkdir -p "$RUNDIR"

# ---------------------------------------------------------------------------
# Write scene.json
# ---------------------------------------------------------------------------
cat > "$RUNDIR/scene.json" << JSON
{
  "scene": {
    "periodic": {"enabled": false, "min": [-0.55,-0.55,-0.55], "max": [0.55,0.55,0.55], "cellSize": 2.0},
    "populate": {"count": ${N}, "mode": "nonoverlap", "spacingMul": 2.0, "seed": 12345,
                 "maxAttempts": 500000, "length": 1.0, "radius": 0.0333, "density": 2500.0},
    "randomInit": {"enabled": true, "vSigma": 0.1, "wSpeed": 0.01, "seed": 42},
    "randomForce": {"enabled": false, "fSigma": 0.1, "tauMag": 0.0, "seed": 123},
    "bodies": [{"length": 1.0, "diameter": 0.01, "density": 1000.0,
                "restitution": 1.0, "friction": ${FRICTION},
                "friction_s": ${FRICTION}, "friction_d": ${FRICTION}}]
  },
  "physics": {
    "dt": ${DT},
    "gravity": [0.0, 0.0, 0.0],
    "lin_damp": 0.0,
    "ang_damp": 0.0,
    "substeps": 1,
    "soft_contact": {
      "enabled": true,
      "delta": ${DELTA},
      "k_scaler": ${K_SCALER},
      "mu": ${FRICTION},
      "mu_static": ${FRICTION},
      "nu": 1e-09,
      "enable_friction": true,
      "verbose": false,
      "use_spatial_hash": false,
      "cell_size": 1.2,
      "use_aabb": true,
      "use_cuda": ${USE_CUDA}
    }
  }
}
JSON

# Symlink initial condition
ln -sf "$INIT_CSV" "$RUNDIR/x_relaxed.txt"

# ---------------------------------------------------------------------------
# Write Sbatch.sh
# ---------------------------------------------------------------------------
GRES_LINE=""
[ -n "$GRES" ] && GRES_LINE="$GRES"

cat > "$RUNDIR/Sbatch.sh" << SBATCH
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -p ${PARTITION}
${GRES_LINE:+$GRES_LINE}
#SBATCH -n 1
#SBATCH -c ${THREADS}
#SBATCH -N 1
#SBATCH -t 1-00:00:00
#SBATCH --mem=${MEM}
#SBATCH -o output_%j.out
#SBATCH -e errors_%j.err

set -euo pipefail

$(echo -e "$MODULES")

BIN=${BIN}

echo "======================================"
echo "Mode:         ${MODE}"
echo "N=${N}  AR=${AR}  friction=${FRICTION}  delta=${DELTA}"
echo "seed_folder:  ${SEED}"
echo "Binary:       \$BIN"
[ "$MODE" = "gpu" ] && echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "======================================"

cd ${RUNDIR}

\$BIN \\
    --headless \\
    --scene scene.json \\
    --init-csv x_relaxed.txt \\
    --output output.csv \\
    --steps ${STEPS} \\
    --dt ${DT} \\
    --threads ${THREADS} \\
    --no-csv \\
    --output-stride 1000 \\
    --entanglement \\
    --entanglement-period ${STEPS} \\
    --entanglement-threads 0 \\
    --entanglement-cutoff 1000.0

echo "Job complete."
SBATCH

chmod +x "$RUNDIR/Sbatch.sh"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "======================================"
echo "Run directory:  $RUNDIR"
echo "Mode:           $MODE"
echo "N=$N  AR=$AR  friction=$FRICTION  delta=$DELTA"
echo "Seed:           $SEED"
echo "Init CSV:       $INIT_CSV"
echo "Binary:         $BIN"
echo "======================================"
echo ""
echo "Files written:"
ls -lh "$RUNDIR"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] scene.json:"
    cat "$RUNDIR/scene.json"
    echo ""
    echo "[dry-run] Not submitting."
else
    if [ "$MODE" = "gpu" ] && [ ! -f "$BIN" ]; then
        echo "WARNING: GPU binary not found at $BIN"
        echo "         Build first: cd build_cuda && make -j4 rigidbody_viewer_3d"
    fi
    jobid=$(sbatch "$RUNDIR/Sbatch.sh" | awk '{print $NF}')
    echo "Submitted job: $jobid"
    echo "Monitor:  squeue -j $jobid"
    echo "Stdout:   $RUNDIR/output_${jobid}.out"
fi
