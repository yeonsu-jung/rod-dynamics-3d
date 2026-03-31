#!/usr/bin/env bash
set -euo pipefail

# Submit ONE SLURM Array Job per N covering all realizations/frictions 
# filtered from the extreme rods summary. Applies dynamic core scaling.
#
# Usage:
#   bash parametric_study/iter_submit_free_rod_array.sh \
#     --extreme-rods-csv /n/home01/yjung/Github/rod-dynamics-3d/extreme_rods_summary.csv \
#     --n-list 50,100,200,500,1000 \
#     --alpha-list 10,25,50,100,150,200,300,500,1000 \
#     --job-name free_rod_array

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMIT_PY="$ROOT_DIR/parametric_study/submit_free_rod_array.py"

EXTREME_CSV="/n/home01/yjung/Github/rod-dynamics-3d/extreme_rods_summary.csv"
N_LIST=""
ALPHA_LIST=""
JOB_NAME="free_rod_array"
FRICTIONS="0.0,0.05,0.1,0.15,0.2,0.4,1.0"
FRAMES="200000"
ENDPOINT_STRIDE="1000"
ENDPOINT_MAX="0"
THREADS="8"
DRY_RUN="0"
IDS_PER_N="0"
ID_SELECT="first"
RANDOM_SEED="42"
ID_RANK_START="1"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extreme-rods-csv) EXTREME_CSV="$2"; shift 2 ;;
    --n-list) N_LIST="$2"; shift 2 ;;
    --alpha-list) ALPHA_LIST="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --frictions) FRICTIONS="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --endpoint-stride) ENDPOINT_STRIDE="$2"; shift 2 ;;
    --endpoint-max) ENDPOINT_MAX="$2"; shift 2 ;;
    --ids-per-n) IDS_PER_N="$2"; shift 2 ;;
    --id-select) ID_SELECT="$2"; shift 2 ;;
    --id-rank-start) ID_RANK_START="$2"; shift 2 ;;
    --random-seed) RANDOM_SEED="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift 1 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$EXTREME_CSV" ]]; then
  echo "--extreme-rods-csv is required" >&2
  exit 1
fi
if [[ -z "$N_LIST" || -z "$ALPHA_LIST" ]]; then
  echo "--n-list and --alpha-list are required" >&2
  exit 1
fi

# Split N_LIST into an array
IFS=',' read -ra N_ARRAY <<< "$N_LIST"

for N_VAL in "${N_ARRAY[@]}"; do
  # Trim whitespace
  N_VAL=$(echo "$N_VAL" | xargs)
  
  if ! [[ "$N_VAL" =~ ^[0-9]+$ ]]; then
      continue
  fi

  # Stop at N > 1000 per user instructions
  if [ "$N_VAL" -gt 1000 ]; then
      echo "Skipping N=$N_VAL (Restricting to N <= 1000)"
      continue
  fi

  echo "=========================================================="
  echo "Processing Free Rod Array for N=$N_VAL | Cores=$THREADS"
  
  # Extract IDs strictly for this N_VAL
  readarray -t ROWS < <(python3 - "$EXTREME_CSV" "$N_VAL" "$ALPHA_LIST" <<'PY'
import csv, sys
csv_path, n_val_s, alpha_list_s = sys.argv[1:4]
target_n = int(n_val_s)
alphas = {int(x.strip()) for x in alpha_list_s.split(',') if x.strip()}
with open(csv_path, newline='') as f:
    for r in csv.DictReader(f):
        n = int(r['N'])
        ar = int(r['AR'])
        if n == target_n and ar in alphas:
            print(f"{n}\t{ar}\t{r['ID']}\t{r['Metric']}")
PY
  )

  if (( IDS_PER_N > 0 )); then
    readarray -t ROWS < <(printf '%s\n' "${ROWS[@]}" | python3 -c '
import random, sys
ids_per_n, mode, seed, rank_start = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows, group = [], []
for line in sys.stdin:
    if line.strip():
        n, ar, key, metric = line.strip().split("\t")
        r = (int(n), int(ar), key, metric)
        rows.append(r)
        group.append(r)

rng = random.Random(seed)
keys = sorted({r[2] for r in group})
picked = set()
if mode == "random":
    picked = set(rng.sample(keys, min(ids_per_n, len(keys))))
else:
    start_idx = rank_start - 1
    picked = set(keys[start_idx:start_idx + ids_per_n])
for r in group:
    if r[2] in picked:
        print(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}")
' "$IDS_PER_N" "$ID_SELECT" "$RANDOM_SEED" "$ID_RANK_START")
  fi

  if [[ ${#ROWS[@]} -eq 0 ]]; then
    echo "No matching realizations for N=$N_VAL in alpha=[$ALPHA_LIST]"
    continue
  fi

  UNIQUE_IDS=()
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r N AR ID METRIC <<< "$row"
    UNIQUE_IDS+=("$ID")
  done

  # Deduplicate IDs
  mapfile -t UNIQUE_IDS < <(printf '%s\n' "${UNIQUE_IDS[@]}" | sort -u)

  echo "Submitting ${#UNIQUE_IDS[@]} configurations for N=$N_VAL to Array bundle..."
  
  CUR_JOB_NAME="${JOB_NAME}_N${N_VAL}"

  CMD=(
    python3 "$SUBMIT_PY"
    --extreme-rods-csv "$EXTREME_CSV"
    --job-name "$CUR_JOB_NAME"
    --frames "$FRAMES"
    --endpoint-stride "$ENDPOINT_STRIDE"
    --endpoint-max "$ENDPOINT_MAX"
    --threads "$THREADS"
    --filter-id "${UNIQUE_IDS[@]}"
    --bundle-all
  )

  if [[ -n "$FRICTIONS" ]]; then
    CMD+=(--frictions "$FRICTIONS")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    CMD+=(--dry-run)
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  "${CMD[@]}"
done

echo "Done submitting all Arrays."
