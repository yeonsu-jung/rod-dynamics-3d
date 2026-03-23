#!/bin/bash
# iter_submit_free_rod.sh
#
# Meta-runner: submits free-rod perturbation jobs for every
# (N, AR, seed, metric, friction) combination in extreme_rods_summary.csv.
#
# Sweep dimensions:
#   N      : from CSV  (10, 15, 20, 30, 50, 100, 200, 500, 1000, 1500, 2000)
#   AR     : from CSV  (10, 25, 50, 100, 150, 200, 300, 500, 1000)
#   metric : from CSV  (MinFSA, MaxFSA, MinFTA, MaxFTA)
#   mu     : 5 values  (0.0, 0.1, 0.2, 0.4, 1.0)
#
# Submission strategy (all CPU — broadphase is O(N) with one free rod):
#   N <  COMBINE_N_THRESHOLD  →  CPU combined,  THREADS_SMALL cores
#   N >= COMBINE_N_THRESHOLD  →  CPU combined,  THREADS_LARGE cores
#
# Usage:
#   bash parametric_study/iter_submit_free_rod.sh
#   DRY_RUN=true bash parametric_study/iter_submit_free_rod.sh
#
# Scope filters (space-separated values; empty = all):
#   FILTER_N="10 50" FILTER_AR="100 500" DRY_RUN=true bash ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$REPO_ROOT/parametric_study/submit_free_rod.py"
CSV="$REPO_ROOT/extreme_rods_summary.csv"

# ── Physics parameters ────────────────────────────────────────────────────────
FRICTIONS="0.0,0.1,0.2,0.4,1.0"
FRAMES=200000
DT=0.0005
KICK=0.1        # translational kick (vSigma)
WSPEED=0.2      # rotational kick (wSpeed)
PERROD_STRIDE=667    # 200000 / 667 ≈ 300 output frames

# ── Input data roots ─────────────────────────────────────────────────────────
INPUT_SMALL="$REPO_ROOT/initial-configs/relaxation_3rd_multithreading"
INPUT_LARGE="/n/holylabs/mahadevan_lab/Users/yjung/relaxation/relaxation_large_N_gpuEntangle_cpuRelax"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_BASE="/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs"
JOB_NAME="free_rod_sweep"

# ── N threshold and core counts ───────────────────────────────────────────────
COMBINE_N_THRESHOLD=500   # N < this → small tier;  N >= this → large tier
THREADS_SMALL=8           # cores for N < COMBINE_N_THRESHOLD
THREADS_LARGE=16          # cores for N >= COMBINE_N_THRESHOLD

# ── Scope filters (space-separated; empty = all) ──────────────────────────────
FILTER_N="${FILTER_N:-}"
FILTER_AR="${FILTER_AR:-}"
FILTER_METRIC="${FILTER_METRIC:-}"

# ── Dry-run flag ──────────────────────────────────────────────────────────────
DRY_RUN="${DRY_RUN:-false}"

# ─────────────────────────────────────────────────────────────────────────────

if [ ! -f "$CSV" ]; then
    echo "ERROR: CSV not found: $CSV"
    exit 1
fi

echo "========================================"
echo "Free-rod meta-runner"
echo "CSV:       $CSV"
echo "Frictions: $FRICTIONS"
echo "Frames:    $FRAMES  dt: $DT  vSigma: $KICK  wSpeed: $WSPEED"
echo "Job name:  $JOB_NAME"
echo "Strategy:  N<$COMBINE_N_THRESHOLD → bundle into 1 SLURM job ($THREADS_SMALL cores)"
echo "           N>=$COMBINE_N_THRESHOLD → per-entry combined ($THREADS_LARGE cores)"
echo "Filter N:  ${FILTER_N:-<all>}   AR: ${FILTER_AR:-<all>}"
echo "Dry run:   $DRY_RUN"
echo "========================================"

DRY_ARG=""
[ "$DRY_RUN" = true ] && DRY_ARG="--dry-run"

# Helper: intersect two space-separated integer lists
intersect() {
    python3 -c "
a=set(map(int,'$1'.split()))
b=set(map(int,'$2'.split()))
r=sorted(a&b)
print(' '.join(map(str,r)) if r else '')
"
}

# Collect N values in each tier from CSV
ALL_SMALL_N=$(python3 -c "
import csv
rows=list(csv.DictReader(open('$CSV')))
ns=sorted({int(r['N']) for r in rows if int(r['N'])<$COMBINE_N_THRESHOLD})
print(' '.join(map(str,ns)))
")

ALL_LARGE_N=$(python3 -c "
import csv
rows=list(csv.DictReader(open('$CSV')))
ns=sorted({int(r['N']) for r in rows if int(r['N'])>=$COMBINE_N_THRESHOLD})
print(' '.join(map(str,ns)))
")

# Apply FILTER_N
if [ -n "$FILTER_N" ]; then
    SMALL_N=$([ -n "$ALL_SMALL_N" ] && intersect "$FILTER_N" "$ALL_SMALL_N" || echo "")
    LARGE_N=$([ -n "$ALL_LARGE_N" ] && intersect "$FILTER_N" "$ALL_LARGE_N" || echo "")
else
    SMALL_N="$ALL_SMALL_N"
    LARGE_N="$ALL_LARGE_N"
fi

AR_ARG=$([ -n "$FILTER_AR" ] && echo "--filter-ar $FILTER_AR" || echo "")
MET_ARG=$([ -n "$FILTER_METRIC" ] && echo "--filter-metric $FILTER_METRIC" || echo "")

COMMON_ARGS="
    --extreme-rods-csv $CSV
    --input-root $INPUT_SMALL $INPUT_LARGE
    --job-name $JOB_NAME
    --runs-root $OUTPUT_BASE
    --frictions $FRICTIONS
    --frames $FRAMES
    --dt $DT
    --init-velocity-sigma $KICK
    --w-speed $WSPEED
    --perrod-stride $PERROD_STRIDE"

# ── Batch 1: small N  →  one bundle SLURM job ───────────────────────────────
echo ""
echo "--- Batch 1: N < $COMBINE_N_THRESHOLD → bundle (1 SLURM job, $THREADS_SMALL cores) ---"
if [ -n "$SMALL_N" ]; then
    echo "N values: $SMALL_N"
    python3 "$SCRIPT" $COMMON_ARGS \
        --threads $THREADS_SMALL \
        --filter-n $SMALL_N \
        $AR_ARG $MET_ARG \
        --bundle-all \
        $DRY_ARG
else
    echo "No N values for this tier."
fi

# ── Batch 2: large N  →  one job per entry (combined frictions) ──────────────
echo ""
echo "--- Batch 2: N >= $COMBINE_N_THRESHOLD ($THREADS_LARGE cores, per-entry) ---"
if [ -n "$LARGE_N" ]; then
    echo "N values: $LARGE_N"
    python3 "$SCRIPT" $COMMON_ARGS \
        --threads $THREADS_LARGE \
        --filter-n $LARGE_N \
        $AR_ARG $MET_ARG \
        --combine-frictions \
        $DRY_ARG
else
    echo "No N values for this tier."
fi

echo ""
echo "Done."
