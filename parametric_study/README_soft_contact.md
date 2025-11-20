# Soft Contact Model Parametric Sweep

This directory contains scripts for running a parametric sweep of the soft contact model, varying friction coefficients and noise amplitudes.

## Parameters

**Friction coefficients (μ):** 0.0, 0.05, 0.1, 0.2, 0.4

**Noise amplitudes (σ):** 1e-5, 1e-4, 1e-3, 1e-2, 1e-1

**System configuration:**
- Number of rods: 200
- Periodic box: [-1, 1]³
- Rod length: 1.5
- Rod diameter: 0.05
- Density: 1000.0

**Simulation parameters:**
- Time step: 0.001
- Total steps: 100,000
- Output interval: 100
- Soft contact model: k_scaler = 1e1, delta = 0.0002, nu = 0.1

**Total runs:** 5 × 5 = 25 simulations

## Usage

### Running the sweep

#### Option 1: Submit individual jobs to cluster (RECOMMENDED)
```bash
cd parametric_study
python3 submit_soft_contact.py --job-name soft_contact_sweep
```

This creates 25 separate SLURM jobs, each with its own run directory containing:
- Scene file with specific parameters
- Executable copy
- Individual sbatch script
- Output files

#### Option 2: Direct execution (local, sequential)
```bash
cd parametric_study
python3 sweep_soft_contact.py
```

#### Option 3: Submit single job for all runs
```bash
cd parametric_study
sbatch submit_soft_contact_sweep.sh
```

**Dry run test (recommended first):**
```bash
python3 submit_soft_contact.py --job-name test --dry-run
```

### Analyzing results

After all jobs complete, analyze the aggregated results:

#### For individual job submissions (Option 1):
```bash
cd parametric_study
python3 post_analyze_soft_contact.py \
    --job-name soft_contact_sweep \
    --make-plots \
    --outdir analysis_soft_contact_sweep
```

This will:
1. Scan all run directories
2. Extract KE data and parameters from each run
3. Compute statistics (mean KE, std, growth rate)
4. Generate comprehensive plots and summary table

#### For local/single job execution (Options 2-3):
```bash
cd parametric_study
python3 analyze_soft_contact.py
```

### Generated outputs

The analysis produces:
- **summary_table.csv** - Tabular data with statistics for each parameter combination
- **heatmap_mean_ke.png** - Mean kinetic energy vs friction and noise
- **heatmap_growth_rate.png** - Energy growth/decay rate vs parameters
- **ke_traces_by_friction.png** - Time traces grouped by friction coefficient
- **ke_traces_by_noise.png** - Time traces grouped by noise amplitude
- **growth_rate_vs_noise.png** - Growth rate dependence on noise for each friction level

## Output Structure

### Individual job submission (Option 1)
```
/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/{job_name}/
├── run_dirs.txt                           # List of all run directories
├── parameter_summary.json                 # Parameter sweep configuration
├── combined_analysis_command.txt          # Command to run combined analysis
└── YYYYMMDD-HHMMSS_RUN_soft_mu*_noise*/  # Individual run directories
    ├── scene.json                         # Scene configuration
    ├── rigidbody_viewer_3d                # Executable copy
    ├── Sbatch.sh                          # SLURM submission script
    ├── README.txt                         # Run documentation
    ├── options.txt                        # Parameter values
    ├── profile.csv                        # Simulation output
    ├── output_*.out                       # SLURM stdout
    ├── errors_*.err                       # SLURM stderr
    ├── figs/                              # Analysis plots
    │   ├── ke.png
    │   └── contacts.png
    └── analysis.txt                       # Per-run analysis
```

### Local/single job execution (Options 2-3)
```
parametric_study/
├── scenes_soft_contact/          # Generated scene files
│   └── soft_contact_mu*_noise*.json
├── analysis_soft_contact/        # Simulation outputs
│   ├── mu*_noise*_ke.csv         # Kinetic energy data
│   ├── mu*_noise*_pos.csv        # Position data
│   ├── analysis_results.npz      # Processed results
│   ├── sweep_summary.txt         # Run summary
│   └── plots/                    # Analysis plots
│       ├── ke_heatmap.png
│       ├── growth_rate_heatmap.png
│       ├── ke_traces_by_friction.png
│       └── ke_traces_by_noise.png
```

## Scripts

- `submit_soft_contact.py` - **RECOMMENDED**: Creates and submits individual SLURM jobs for each parameter combination
- `sweep_soft_contact.py` - Sequential sweep script that generates scenes and runs simulations locally
- `analyze_soft_contact.py` - Analysis script that processes results and creates plots
- `submit_soft_contact_sweep.sh` - Single SLURM job submission script (runs all 25 sequentially)

## Expected Behavior

The soft contact model uses a penalty-based approach with:
- Repulsive spring force proportional to overlap
- Damping proportional to relative velocity
- Coulomb friction in tangential direction

Expected trends:
1. **Friction effects:** Higher friction should dissipate energy faster, potentially stabilizing the system
2. **Noise effects:** Higher noise amplitudes should drive energy growth, potentially counteracting dissipation
3. **Combined effects:** Competition between friction dissipation and noise injection

The analysis will quantify:
- Steady-state kinetic energy levels
- Energy growth/decay rates
- Parameter regimes where system equilibrates vs grows

## Notes

- Each simulation takes approximately 5-15 minutes depending on system complexity
- Total sweep time: ~2-6 hours for 25 runs
- Memory usage: ~1-2 GB per simulation
- Disk usage: ~100-500 MB per run for output files
