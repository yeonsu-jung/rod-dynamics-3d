# Free-Rod Handoff 2026-04-03

This note captures the current state of the free-rod NSC batch so work can resume on another machine without re-deriving the setup.

## Current State

- Job root: `runs/free_rod_extreme_nsc`
- Analysis output dir: `analysis_outputs/free_rod_extreme_nsc`
- Input CSV: `/Users/yeonsu/Downloads/extreme_rods.csv`
- Relaxed-config root: `initial-configs/relaxation_3rd_multithreading`
- Contact mode used here: NSC with friction list `0.0,0.1,0.2,0.4,1.0`
- Batch status at handoff: complete

Verified counts from the completed run:

- `1620` run directories complete
- `8100` endpoint CSVs present
- all requested `mu` values present for every run

## Important Implementation Notes

The launcher and analysis flow already include the key fixes needed for this dataset:

- `parametric_study/submit_free_rod.py` converts each `x_relaxed_AR*.txt` into a simulator-ready `init_config.csv`
- local runs must prefer `build/rigidbody_viewer_3d` over `build-headless/rigidbody_viewer_3d`
- `parametric_study/analyze_free_rod_endpoints.py` now writes both full per-simulation and grouped summary tables

Why the binary preference matters:

- on this machine, `build-headless/rigidbody_viewer_3d` completed runs but did not emit `free_rod_endpoints_mu*.csv`
- `build/rigidbody_viewer_3d` did emit endpoint CSVs correctly

## Output Files To Use

Primary outputs under `analysis_outputs/free_rod_extreme_nsc`:

- `health_summary.json`
- `individual_simulation_results.csv`
- `summary_by_mu_metric.csv`
- `mean_displacement_vs_time.png`
- `mean_path_vs_time.png`
- `mean_orientation_change_vs_time.png`
- `final_displacement_boxplot.png`
- `final_path_boxplot.png`
- `final_orientation_change_boxplot.png`

Table meanings:

- `individual_simulation_results.csv`: one row per `(run_name, mu)` with final and max displacement, path length, and orientation-change values
- `summary_by_mu_metric.csv`: grouped aggregates by `(mu, metric)` with count, mean, std, median, min, and max

## How To Reproduce The Analysis

From the repo root:

```bash
/Users/yeonsu/anaconda3/bin/conda run -p /Users/yeonsu/anaconda3 --no-capture-output python \
  /Users/yeonsu/.vscode/extensions/ms-python.python-2026.4.0-darwin-arm64/python_files/get_output_via_markers.py \
  parametric_study/analyze_free_rod_endpoints.py \
  runs/free_rod_extreme_nsc \
  --out-dir analysis_outputs/free_rod_extreme_nsc
```

Quick completeness checks:

```bash
find runs/free_rod_extreme_nsc -name 'free_rod_endpoints_mu*.csv' | wc -l
```

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path('analysis_outputs/free_rod_extreme_nsc/health_summary.json')
data = json.loads(path.read_text())
print(data['complete_runs'], data['incomplete_runs'])
print(data['per_mu_file_counts'])
PY
```

## Recommended Next Work

The most natural next steps are:

1. Inspect `summary_by_mu_metric.csv` for trends split by `mu` and `metric`
2. Slice `individual_simulation_results.csv` by `N` or `AR` to identify which extreme cases dominate the large `mu=0` tails
3. Repeat the same batch with a soft-contact configuration for solver comparison
4. If moving to a new machine, verify the local binary used for endpoint output before launching a full rerun

## If You Need To Relaunch

Use the guide in `parametric_study/FREE_ROD_BATCH_GUIDE.md` for the full launcher syntax.

The exact local NSC batch form used for this completed dataset is:

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

## Commit Scope

For this handoff commit, only the free-rod analysis changes and outputs should be included. There are unrelated local modifications elsewhere in the repo that should stay out of this commit.