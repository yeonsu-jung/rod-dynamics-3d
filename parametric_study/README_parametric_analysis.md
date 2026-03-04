# Parametric Analysis Scripts for Stable Core vs N/AR

This directory contains scripts for analyzing stable core size, delta C, and average pair distance as functions of N (number of rods) or AR (aspect ratio).

## Overview

The analysis pipeline consists of:
1. **Analysis scripts** - Compute metrics from simulation data
2. **Iteration scripts** - Submit batch jobs across multiple N values
3. **Plotting scripts** - Visualize results

## Analysis Scripts

### 1. Stable Core vs N/AR Analysis

**Script**: `study/analyze_stable_core_vs_n.py`

Analyzes stable core size and related metrics for multiple simulation runs.

**Usage**:
```bash
python study/analyze_stable_core_vs_n.py <input_dirs_or_csvs> --output stable_core_vs_n.csv
```

**Arguments**:
- `input_dirs`: Directories containing `endpoints_formatted.csv` or direct paths to CSV files
- `--output`: Output CSV file (default: `stable_core_vs_n.csv`)
- `--frame-initial`: Initial frame to analyze (default: 0)
- `--frame-final`: Final frame to analyze (default: last frame)

**Output**: CSV file with columns:
- `N`: Number of rods
- `core_size`: Number of rods in stable core
- `core_fraction`: Fraction of rods in stable core
- `n_changes`: Number of changed vorticity triples
- `avg_dist`: Average pairwise distance in stable core
- `AR`, `mu`: Extracted from directory names if available

### 2. Time Series Analysis

**Script**: `study/analyze_timeseries.py`

Analyzes delta C, average pair distance, and stable core size over time.

**Usage**:
```bash
python study/analyze_timeseries.py <endpoints_csv> --output timeseries.csv --stride 10 --plot
```

**Arguments**:
- `input_file`: Path to `endpoints_formatted.csv`
- `--output`: Output CSV file (default: `timeseries_analysis.csv`)
- `--frames`: Specific frames to analyze (e.g., `--frames 0 50 100`)
- `--stride`: Analyze every N-th frame (e.g., `--stride 10`)
- `--plot`: Generate plots automatically

**Output**: CSV file with columns:
- `frame`: Frame number
- `total_chirality`: Total chirality C
- `delta_C`: |C - C_initial|
- `core_size`: Stable core size
- `core_fraction`: Stable core fraction
- `avg_pair_distance`: Average pairwise distance between all rods
- `n_changed_triples`: Number of changed vorticity triples

## Batch Submission Scripts

### 1. Stable Core Analysis Across N Values

**Iteration Script**: `parametric_study/iter_analyze_stable_core_vs_n.sh`

Submits analysis jobs for all N values found in the runs directory.

**Usage**:
```bash
bash parametric_study/iter_analyze_stable_core_vs_n.sh
```

**Configuration**: Edit the script to set:
- `RUNS_BASE`: Base directory containing `relax3rd_N*_sweep` folders
- `OUTPUT_BASE`: Output directory for results

**Submit Script**: `parametric_study/submit_stable_core_analysis.sh`

SLURM script that processes all runs for a given N value.

### 2. Time Series Analysis Across Runs

**Iteration Script**: `parametric_study/iter_analyze_timeseries.sh`

Submits time series analysis jobs for all runs.

**Usage**:
```bash
bash parametric_study/iter_analyze_timeseries.sh
```

**Configuration**: Edit the script to set:
- `RUNS_BASE`: Base directory containing simulation runs
- `OUTPUT_BASE`: Output directory for results

**Submit Script**: `parametric_study/submit_timeseries_analysis.sh`

SLURM script that analyzes a single run's time series.

## Plotting Scripts

### 1. Plot Stable Core vs N/AR

**Script**: `study/plot_stable_core_vs_n.py`

Creates plots of stable core metrics vs N or AR with error bars.

**Usage**:
```bash
python study/plot_stable_core_vs_n.py stable_core_N*.csv --output stable_core_vs_n
```

**Output**: PNG file with 4 subplots:
- Stable core size vs N/AR
- Stable core fraction vs N/AR
- Average distance in stable core vs N/AR
- Number of changed triples vs N/AR

### 2. Plot Time Series

**Script**: `study/plot_timeseries.py`

Creates time series plots for delta C, average pair distance, and stable core metrics.

**Usage**:
```bash
# Individual plot
python study/plot_timeseries.py run1_timeseries.csv --individual

# Overlay multiple runs
python study/plot_timeseries.py *_timeseries.csv --overlay --output overlay.png
```

**Arguments**:
- `--overlay`: Create overlay plot of all files
- `--individual`: Create individual plots for each file
- `--output`: Output file name

## Example Workflow

### Analyze Stable Core vs N

```bash
# 1. Submit batch analysis jobs
bash parametric_study/iter_analyze_stable_core_vs_n.sh

# 2. Wait for jobs to complete, then plot results
python study/plot_stable_core_vs_n.py \
    /path/to/stable_core_analysis/N*/stable_core_N*.csv \
    --output study/stable_core_vs_n
```

### Analyze Time Series

```bash
# 1. Submit time series analysis jobs
bash parametric_study/iter_analyze_timeseries.sh

# 2. Wait for jobs to complete, then create overlay plot
python study/plot_timeseries.py \
    /path/to/timeseries_analysis/N200/*_timeseries.csv \
    --overlay --output study/timeseries_N200_overlay.png
```

## Notes

- All scripts use the topology analysis utilities from `study/compute_topology.py` and `study/find_stable_core.py`
- For large N values (N > 300), pairwise distance calculations use sampling to keep computation tractable
- Time series analysis with `--stride 10` analyzes every 10th frame to balance detail and computation time
- SLURM scripts are configured for the `seas_compute` partition with appropriate memory and time limits
