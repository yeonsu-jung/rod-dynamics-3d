# Free-Rod Batch Guide

This workflow runs endpoint-only free-rod tests from an extreme-rods CSV using the same solver/contact logic as the main simulator.

Script

- [parametric_study/submit_free_rod.py](/Users/yeonsu/GitHub/rod-dynamics-3d/parametric_study/submit_free_rod.py)

Inputs used here

- Extreme rods CSV: [/Users/yeonsu/Downloads/extreme_rods.csv](/Users/yeonsu/Downloads/extreme_rods.csv)
- Relaxed configurations root: [initial-configs/relaxation_3rd_multithreading](/Users/yeonsu/GitHub/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading)

## What It Produces

Each run directory contains:

- `rigidbody_viewer_3d`
- `x_relaxed.txt` as the original packing input
- `init_config.csv` as the converted simulator-ready endpoint CSV
- one or more `scene_*.json`
- endpoint CSVs such as `free_rod_endpoints_mu0p2.csv`
- `Sbatch.sh`

Local batch mode also writes:

- `local_task_manifest.csv` for bundled runs, or a timestamped manifest under the job root
- `local_run_summary.json` for queued/completed/failed task counts

For local runs, the launcher prefers `build/rigidbody_viewer_3d` before `build-headless/rigidbody_viewer_3d` because the former is the working binary for `--test-rod-endpoints` output in this repo.

## Recommended Local Launch

This setting is tuned to avoid oversubscribing a laptop or workstation: a few concurrent simulations, each with a small thread count.

```bash
/Users/yeonsu/anaconda3/bin/conda run -p /Users/yeonsu/anaconda3 --no-capture-output python \
  /Users/yeonsu/.vscode/extensions/ms-python.python-2026.4.0-darwin-arm64/python_files/get_output_via_markers.py \
  parametric_study/submit_free_rod.py \
  --extreme-rods-csv /Users/yeonsu/Downloads/extreme_rods.csv \
  --input-root /Users/yeonsu/GitHub/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading \
  --job-name free_rod_extreme_nsc \
  --local \
  --local-workers 4 \
  --threads 2 \
  --nsc \
  --frictions 0.0,0.1,0.2,0.4,1.0
```

Why this setup:

- `--local-workers 4` runs four subprocesses in parallel
- `--threads 2` gives each subprocess two simulator threads
- total target load is about eight simulation threads, which is safer than `4 x 8`

## Pilot Run

Use this first if you want a quick sanity check on one case.

```bash
/Users/yeonsu/anaconda3/bin/conda run -p /Users/yeonsu/anaconda3 --no-capture-output python \
  /Users/yeonsu/.vscode/extensions/ms-python.python-2026.4.0-darwin-arm64/python_files/get_output_via_markers.py \
  parametric_study/submit_free_rod.py \
  --extreme-rods-csv /Users/yeonsu/Downloads/extreme_rods.csv \
  --input-root /Users/yeonsu/GitHub/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading \
  --job-name free_rod_pilot_nsc \
  --local \
  --local-workers 2 \
  --threads 2 \
  --nsc \
  --frictions 0.2,1.0 \
  --filter-n 10 \
  --filter-ar 25 \
  --filter-id 278_868_121 \
  --filter-metric MinFSA
```

## SLURM Launch

The same script still supports cluster submission.

```bash
python parametric_study/submit_free_rod.py \
  --extreme-rods-csv /Users/yeonsu/Downloads/extreme_rods.csv \
  --input-root /Users/yeonsu/GitHub/rod-dynamics-3d/initial-configs/relaxation_3rd_multithreading \
  --job-name free_rod_extreme_nsc \
  --nsc \
  --frictions 0.0,0.1,0.2,0.4,1.0
```

## Choosing Contact Logic

NSC / hard-contact run:

- add `--nsc`
- optional solver knobs: `--nsc-iters`, `--nsc-beta`, `--nsc-cfm`, `--nsc-omega`, `--nsc-pos-iters`, `--nsc-pos-psor`

Soft-contact run:

- omit `--nsc`
- optional soft-contact knob: `--delta`

## Output Layout

Job root:

- [runs](/Users/yeonsu/GitHub/rod-dynamics-3d/runs)

Typical local job root:

- `runs/free_rod_extreme_nsc/`

Typical per-run directory:

- `runs/free_rod_extreme_nsc/2026..._N10_278_868_121_AR25_MinFSA_rod3/`

## Post-Run Checks

Check produced endpoint files:

```bash
find runs/free_rod_extreme_nsc -name 'free_rod_endpoints_mu*.csv' | wc -l
```

Look at the queued-task manifest:

```bash
ls runs/free_rod_extreme_nsc/*local_task_manifest.csv
```

Look at the batch summary:

```bash
ls runs/free_rod_extreme_nsc/*local_run_summary.json
```

Analyze outputs:

```bash
python parametric_study/analyze_free_rod_endpoints.py runs/free_rod_extreme_nsc
```

## Notes

- The launcher converts each `x_relaxed_AR*.txt` into `init_config.csv` before calling `--init-csv`.
- The script uses `FilePath` from the CSV as a hint, but it also resolves paths relative to the local relaxation-config root.
- Combined-friction mode is the default, so one run directory typically contains all requested `mu` values for a given extreme-rod entry.
- If rerun, completed directories are skipped when the last expected endpoint file already exists.
