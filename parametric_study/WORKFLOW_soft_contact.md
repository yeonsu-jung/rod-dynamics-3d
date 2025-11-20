# Soft Contact Parametric Sweep - Quick Reference

## Complete Workflow

### 1. Build the executable (one time)
```bash
cd /n/home01/yjung/Github/rod-dynamics-3d
mkdir -p build && cd build
cmake -DBUILD_HEADLESS=ON ..
make -j8
```

### 2. Test submission (recommended)
```bash
cd /n/home01/yjung/Github/rod-dynamics-3d/parametric_study
python3 submit_soft_contact.py --job-name test_dry --dry-run
```

This creates the directory structure without submitting jobs.

### 3. Submit jobs
```bash
python3 submit_soft_contact.py --job-name soft_contact_sweep
```

This will:
- Create 25 run directories under `/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/soft_contact_sweep/`
- Submit 25 SLURM jobs (one per parameter combination)
- Each job runs independently and generates its own outputs

### 4. Monitor jobs
```bash
# Check job status
squeue -u $USER

# Check specific job output (while running or after completion)
cd /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/soft_contact_sweep/
ls -lt | head -10  # See recent directories

# View output from a specific run
cd 20251120-HHMMSS_RUN_soft_mu0.20_noise1.0e-03_soft_contact_sweep/
cat output_*.out  # SLURM stdout
cat errors_*.err  # SLURM stderr
```

### 5. Check individual run results
Each run directory contains:
```
profile.csv        # Simulation data (frame, KE, n_contacts, etc.)
figs/ke.png        # Per-run KE plot
analysis.txt       # Per-run statistics
```

### 6. Aggregate analysis (after all jobs complete)
```bash
cd /n/home01/yjung/Github/rod-dynamics-3d/parametric_study
python3 post_analyze_soft_contact.py \
    --job-name soft_contact_sweep \
    --make-plots \
    --outdir analysis_soft_contact_sweep
```

### 7. View results
```bash
cd analysis_soft_contact_sweep
ls -lh *.png *.csv

# View on local machine (from your laptop)
scp -r username@login.rc.fas.harvard.edu:/path/to/analysis_soft_contact_sweep ./
```

## Parameter Space

- **Friction coefficients (μ):** 0.0, 0.05, 0.1, 0.2, 0.4
- **Noise amplitudes (σ):** 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
- **Total runs:** 25 (5×5 grid)

## Expected Run Time

- **Per job:** ~2-8 hours (depends on system complexity)
- **Total wall time:** ~2-8 hours (with parallel execution)
- **Sequential time:** ~50-200 hours (if run sequentially)

## Troubleshooting

### Jobs not starting
```bash
# Check partition availability
sinfo -p seas_compute

# Check your job priority
squeue -u $USER --start
```

### Jobs failing
```bash
# Find failed run directories
cd /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/soft_contact_sweep/
grep -l "error\|Error\|ERROR" */errors_*.err

# Check error messages
cat <failed_run_dir>/errors_*.err
```

### Resubmit individual jobs
```bash
# Go to specific run directory
cd /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/soft_contact_sweep/<run_dir>

# Resubmit
sbatch Sbatch.sh
```

### Analysis fails to find runs
```bash
# Check if run_dirs.txt exists
cat /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/soft_contact_sweep/run_dirs.txt

# If missing, analysis will scan for directories with pattern '_RUN_soft_'
```

## File Locations

**Code:**
- Submission script: `parametric_study/submit_soft_contact.py`
- Analysis script: `parametric_study/post_analyze_soft_contact.py`

**Data:**
- Run directories: `/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/{job_name}/`
- Individual outputs: `{run_dir}/profile.csv`, `{run_dir}/figs/`
- Aggregated analysis: `{outdir}/` (specified in analysis command)

## Quick Commands Reference

```bash
# Submit jobs
python3 submit_soft_contact.py --job-name my_sweep

# Check status
squeue -u $USER

# Cancel all jobs
scancel -u $USER

# Cancel specific job pattern
scancel -u $USER -n soft_mu*

# Analyze results
python3 post_analyze_soft_contact.py --job-name my_sweep --make-plots --outdir analysis_my_sweep

# Check disk usage
du -sh /n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/my_sweep/
```
