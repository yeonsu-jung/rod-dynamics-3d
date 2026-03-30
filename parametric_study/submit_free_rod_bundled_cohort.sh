#!/usr/bin/env bash
set -euo pipefail

# Submit free-rod cohorts as a small number of bundled SLURM jobs.
#
# Default behavior: one bundle job per N value (9 total jobs for sweep grid),
# where each bundle job runs all matching AR x ID x metric x friction cases.
# This avoids submitting hundreds/thousands of tiny jobs for short trajectories.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMIT_PY="$ROOT_DIR/parametric_study/submit_free_rod.py"

EXTREME_CSV="$ROOT_DIR/extreme_rods_summary.csv"
JOB_NAME="free_rod_sweep4_bundled"
N_LIST="10,15,20,30,50,100,200,500,1000"
ALPHA_LIST="10,25,50,100,150,200,300,500,1000"
FRICTIONS="0.0,0.1,0.2,0.4,1.0"
FRAMES="20000"
ENDPOINT_MAX="300"
THREADS="8"
TIME_LIMIT=""
DRY_RUN="0"
EXTRA_ARGS=()

print_help() {
  cat <<'EOF'
Usage:
  bash parametric_study/submit_free_rod_bundled_cohort.sh [options]

Options:
  --extreme-rods-csv PATH   Input summary CSV (default: repo/extreme_rods_summary.csv)
  --job-name NAME           Job group name under runs/ (default: free_rod_sweep4_bundled)
  --n-list CSV              N values; one bundled SLURM job per N
  --alpha-list CSV          AR(alpha) values to include
  --frictions CSV           Friction values (default: 0.0,0.1,0.2,0.4,1.0)
  --frames N                Steps per simulation (default: 20000)
  --endpoint-max N          Max sampled endpoint rows (default: 300)
  --threads N               Threads per simulation process (default: 8)
  --time SLURM_TIME         Optional SLURM time limit override (e.g. 0-12:00:00)
  --dry-run                 Generate scripts but do not submit
  --help                    Show this help

Notes:
  - Submits one --bundle-all job per N value (9 jobs by default).
  - Each bundled job runs all matching rows for that N across selected AR/ID/metric
    and all selected frictions.
  - Extra args after '--' are forwarded to submit_free_rod.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      print_help
      exit 0 ;;
    --extreme-rods-csv)
      EXTREME_CSV="$2"; shift 2 ;;
    --job-name)
      JOB_NAME="$2"; shift 2 ;;
    --n-list)
      N_LIST="$2"; shift 2 ;;
    --alpha-list)
      ALPHA_LIST="$2"; shift 2 ;;
    --frictions)
      FRICTIONS="$2"; shift 2 ;;
    --frames)
      FRAMES="$2"; shift 2 ;;
    --endpoint-max)
      ENDPOINT_MAX="$2"; shift 2 ;;
    --threads)
      THREADS="$2"; shift 2 ;;
    --time)
      TIME_LIMIT="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN="1"; shift 1 ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1 ;;
  esac
done

if [[ ! -f "$SUBMIT_PY" ]]; then
  echo "submit script not found: $SUBMIT_PY" >&2
  exit 1
fi
if [[ ! -f "$EXTREME_CSV" ]]; then
  echo "extreme rods CSV not found: $EXTREME_CSV" >&2
  exit 1
fi

IFS=',' read -r -a NS <<< "$N_LIST"
IFS=',' read -r -a ARS <<< "$ALPHA_LIST"

echo "Submitting bundled cohort jobs"
echo "  job_name     : $JOB_NAME"
echo "  Ns           : ${NS[*]}"
echo "  ARs          : ${ARS[*]}"
echo "  frictions    : $FRICTIONS"
echo "  frames       : $FRAMES"
echo "  endpoint_max : $ENDPOINT_MAX"
echo "  threads      : $THREADS"
echo "  dry_run      : $DRY_RUN"

for N in "${NS[@]}"; do
  # Trim potential whitespace from CSV entries.
  N="${N// /}"
  [[ -z "$N" ]] && continue

  CMD=(
    python3 "$SUBMIT_PY"
    --extreme-rods-csv "$EXTREME_CSV"
    --job-name "$JOB_NAME"
    --bundle-all
    --frames "$FRAMES"
    --endpoint-max "$ENDPOINT_MAX"
    --threads "$THREADS"
    --frictions "$FRICTIONS"
    --filter-n "$N"
    --filter-alpha "${ARS[@]}"
  )

  if [[ -n "$TIME_LIMIT" ]]; then
    CMD+=(--time "$TIME_LIMIT")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    CMD+=(--dry-run)
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  echo "[submit-bundle] N=$N"
  "${CMD[@]}"
done

echo "Done. Submitted ${#NS[@]} bundled jobs (before filtering empty N entries)."
