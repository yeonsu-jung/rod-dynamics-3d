#!/usr/bin/env bash
set -euo pipefail

# Submit one SLURM job per realization (row) for selected N and alpha(=AR).
# Each submitted job runs all frictions in sequence and writes endpoint-only CSVs.
#
# Usage:
#   bash parametric_study/iter_submit_free_rod.sh \
#     --extreme-rods-csv /path/to/extreme_rods.csv \
#     --n-list 1000,2000 \
#     --alpha-list 500,1000 \
#     --ids-per-n 3 \
#     --id-select first \
#     --id-rank-start 1 \
#     --random-seed 42 \
#     --frictions 0.0,0.2,0.4 \
#     --job-name free_rod_fast

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMIT_PY="$ROOT_DIR/parametric_study/submit_free_rod.py"

EXTREME_CSV="/n/home01/yjung/Github/rod-dynamics-3d/extreme_rods_summary.csv"
N_LIST=""
ALPHA_LIST=""
JOB_NAME="free_rod"
FRICTIONS="0,0.1,0.2,0.4"
FRAMES="300"
ENDPOINT_STRIDE="-1"
ENDPOINT_MAX="300"
THREADS="8"
DRY_RUN="0"
IDS_PER_N="0"
ID_SELECT="first"
RANDOM_SEED="42"
ID_RANK_START="1"
EXTRA_ARGS=()

print_help() {
  cat <<'EOF'
Usage:
  bash parametric_study/iter_submit_free_rod.sh \
    --extreme-rods-csv /path/to/extreme_rods.csv \
    --n-list 500,1000,2000 \
    --alpha-list 100,300,500,1000 \
    --ids-per-n 3 \
    --id-select first \
    --id-rank-start 1 \
    --frictions 0,0.1,0.2,0.4 \
    --job-name free_rod_fast

Options:
  --extreme-rods-csv PATH   Input summary CSV.
  --n-list CSV              N values, comma-separated.
  --alpha-list CSV          Alpha(AR) values, comma-separated.
  --ids-per-n N             Keep up to N unique IDs per N (0 = all).
  --id-select MODE          first|random. first = lexicographic-first IDs.
  --id-rank-start N         1-based start rank in sorted IDs (default: 1).
                            Example: ids-per-n=1, id-rank-start=2 picks 2nd key.
  --random-seed INT         Seed used only when --id-select random.
  --frictions CSV           Frictions passed to submit script.
  --frames N                Simulation steps.
  --endpoint-stride N       Endpoint sampling stride. Use <=0 to let the app auto-compute stride from --endpoint-max.
  --endpoint-max N          Max sampled endpoint rows.
  --threads N               Threads per job.
  --job-name NAME           Job group name.
  --dry-run                 Do not submit; print planned jobs.
  --help                    Show this help.

Notes:
  - One submitted SLURM job per realization row.
  - By default, frictions are combined within each submitted job.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      print_help
      exit 0 ;;
    --extreme-rods-csv)
      EXTREME_CSV="$2"; shift 2 ;;
    --n-list)
      N_LIST="$2"; shift 2 ;;
    --alpha-list)
      ALPHA_LIST="$2"; shift 2 ;;
    --job-name)
      JOB_NAME="$2"; shift 2 ;;
    --frictions)
      FRICTIONS="$2"; shift 2 ;;
    --frames)
      FRAMES="$2"; shift 2 ;;
    --endpoint-stride)
      ENDPOINT_STRIDE="$2"; shift 2 ;;
    --endpoint-max)
      ENDPOINT_MAX="$2"; shift 2 ;;
    --threads)
      THREADS="$2"; shift 2 ;;
    --ids-per-n)
      IDS_PER_N="$2"; shift 2 ;;
    --id-select)
      ID_SELECT="$2"; shift 2 ;;
    --id-rank-start)
      ID_RANK_START="$2"; shift 2 ;;
    --random-seed)
      RANDOM_SEED="$2"; shift 2 ;;
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

if [[ -z "$EXTREME_CSV" ]]; then
  echo "--extreme-rods-csv is required" >&2
  exit 1
fi
if [[ -z "$N_LIST" || -z "$ALPHA_LIST" ]]; then
  echo "--n-list and --alpha-list are required" >&2
  exit 1
fi
if [[ ! -f "$SUBMIT_PY" ]]; then
  echo "submit script not found: $SUBMIT_PY" >&2
  exit 1
fi

readarray -t ROWS < <(python3 - "$EXTREME_CSV" "$N_LIST" "$ALPHA_LIST" <<'PY'
import csv, sys
csv_path, n_list_s, alpha_list_s = sys.argv[1:4]
ns = {int(x.strip()) for x in n_list_s.split(',') if x.strip()}
alphas = {int(x.strip()) for x in alpha_list_s.split(',') if x.strip()}
with open(csv_path, newline='') as f:
    for r in csv.DictReader(f):
        n = int(r['N'])
        ar = int(r['AR'])
        if n in ns and ar in alphas:
            print(f"{n}\t{ar}\t{r['ID']}\t{r['Metric']}")
PY
)

if ! [[ "$ID_SELECT" == "first" || "$ID_SELECT" == "random" ]]; then
  echo "--id-select must be 'first' or 'random'" >&2
  exit 1
fi

if ! [[ "$IDS_PER_N" =~ ^[0-9]+$ ]]; then
  echo "--ids-per-n must be a nonnegative integer" >&2
  exit 1
fi

if ! [[ "$RANDOM_SEED" =~ ^-?[0-9]+$ ]]; then
  echo "--random-seed must be an integer" >&2
  exit 1
fi

if ! [[ "$ID_RANK_START" =~ ^[0-9]+$ ]] || (( ID_RANK_START < 1 )); then
  echo "--id-rank-start must be an integer >= 1" >&2
  exit 1
fi

if [[ "$ID_SELECT" == "random" && "$ID_RANK_START" != "1" ]]; then
  echo "--id-rank-start is only supported with --id-select first" >&2
  exit 1
fi

if (( IDS_PER_N > 0 )); then
  readarray -t ROWS < <(printf '%s\n' "${ROWS[@]}" | python3 -c '
import random
import sys

ids_per_n = int(sys.argv[1])
mode = sys.argv[2]
seed = int(sys.argv[3])
rank_start = int(sys.argv[4])

rows = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    n, ar, key, metric = line.split("\t")
    rows.append((int(n), int(ar), key, metric))

by_n = {}
for r in rows:
    by_n.setdefault(r[0], []).append(r)

selected_rows = []
rng = random.Random(seed)
for n in sorted(by_n):
    group = by_n[n]
    keys = sorted({r[2] for r in group})
    if mode == "random":
        if len(keys) > ids_per_n:
            picked = set(rng.sample(keys, ids_per_n))
        else:
            picked = set(keys)
    else:
        # Deterministic, lexicographic-first key selection.
        start_idx = rank_start - 1
        if start_idx >= len(keys):
            picked = set()
        else:
            picked = set(keys[start_idx:start_idx + ids_per_n])

    for r in group:
        if r[2] in picked:
            selected_rows.append(r)

for n, ar, key, metric in selected_rows:
    print(f"{n}\t{ar}\t{key}\t{metric}")
' "$IDS_PER_N" "$ID_SELECT" "$RANDOM_SEED" "$ID_RANK_START")
fi

if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo "No matching realizations for N in [$N_LIST], alpha in [$ALPHA_LIST]" >&2
  exit 1
fi

echo "Matched ${#ROWS[@]} realizations. Submitting one job per realization..."

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r N AR ID METRIC <<< "$row"

  CMD=(
    python3 "$SUBMIT_PY"
    --extreme-rods-csv "$EXTREME_CSV"
    --job-name "$JOB_NAME"
    --frames "$FRAMES"
    --endpoint-stride "$ENDPOINT_STRIDE"
    --endpoint-max "$ENDPOINT_MAX"
    --threads "$THREADS"
    --filter-n "$N"
    --filter-alpha "$AR"
    --filter-id "$ID"
    --filter-metric "$METRIC"
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

  echo "[submit] N=$N alpha=$AR id=$ID metric=$METRIC"
  "${CMD[@]}"
done

echo "Done."
