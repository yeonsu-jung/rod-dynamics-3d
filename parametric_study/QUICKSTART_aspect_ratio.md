# Aspect Ratio Study - Quick Start Guide

## Overview

This study examines how the length-to-diameter ratio (L/D) affects rod dynamics. Rod length is fixed at 1.0, and diameter varies to achieve different aspect ratios.

## Quick Start

### 1. Test with dry run
```bash
cd parametric_study
python3 submit_aspect_ratio.py --job-name aspect_test --dry-run
```

### 2. Submit jobs
```bash
python3 submit_aspect_ratio.py --job-name aspect_ratio_study
```

This creates 5 jobs (one per aspect ratio):
- L/D = 10 → diameter = 0.100
- L/D = 50 → diameter = 0.020
- L/D = 100 → diameter = 0.010
- L/D = 200 → diameter = 0.005
- L/D = 500 → diameter = 0.002

### 3. Monitor jobs
```bash
squeue -u $USER
```

### 4. Analyze results (after completion)
```bash
python3 post_analyze_aspect_ratio.py \
    --job-name aspect_ratio_study \
    --make-plots \
    --outdir analysis_aspect_ratio_study
```

## Outputs

The analysis produces:
- `summary_table.csv` - Statistics for each aspect ratio
- `ke_traces_by_aspect_ratio.png` - KE evolution over time
- `statistics_vs_aspect_ratio.png` - Mean KE and growth rate vs L/D
- `contacts_vs_aspect_ratio.png` - Contact count vs L/D

## Parameter Space

**Fixed parameters:**
- Rod length: 1.0
- Number of rods: 200
- Periodic box: [-1, 1]³
- Friction coefficient: 0.2
- Noise amplitude: 1e-3
- Steps: 100,000

**Variable parameter:**
- Aspect ratio (L/D): [10, 50, 100, 200, 500]

## Modifying Parameters

### Add more friction values

Edit `submit_aspect_ratio.py` line 31:
```python
# Change from:
FRICTION_COEFFS = [0.2]

# To:
FRICTION_COEFFS = [0.0, 0.1, 0.2, 0.4]
```

This creates a 2D sweep: 5 aspect ratios × 4 friction values = 20 jobs

### Change aspect ratio values

Edit `submit_aspect_ratio.py` line 28:
```python
ASPECT_RATIOS = [10, 50, 100, 200, 500]  # Change these values
```

### Change noise level

Edit `submit_aspect_ratio.py` line 34:
```python
NOISE_AMPLITUDE = 1e-3  # Adjust this value
```

## Expected Run Time

- Per job: ~5-15 minutes
- Total (all 5 jobs in parallel): ~5-15 minutes
- Sequential: ~25-75 minutes

## Physical Expectations

Higher aspect ratios (thinner rods):
- Lower volume fraction (less crowded)
- Possibly more contacts (easier to pack)
- Different rotational dynamics
- Modified energy dissipation rates

## File Locations

**Scripts:**
- `parametric_study/submit_aspect_ratio.py` - Job submission
- `parametric_study/post_analyze_aspect_ratio.py` - Analysis

**Data:**
- Run directories: `/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/{job_name}/`
- Individual outputs: `{run_dir}/profile.csv`, `{run_dir}/figs/`
- Analysis: `{outdir}/` (specified in analysis command)

## Troubleshooting

### Jobs failing
```bash
# Check error logs
cd /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/{job_name}/
cat */errors_*.err
```

### Resubmit individual job
```bash
cd /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/{job_name}/{run_dir}
sbatch Sbatch.sh
```

### Analysis fails
```bash
# Check if all jobs completed
cd /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/{job_name}/
ls -l */profile.csv  # Should see all 5 files
```
