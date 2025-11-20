# Aspect Ratio Parametric Sweep

This study examines the effect of rod aspect ratio (length-to-diameter ratio) on the dynamics of entangled rod systems.

## Parameters

**Aspect ratios (L/D):** 10, 50, 100, 200, 500

**Rod length (fixed):** 1.0

**Rod diameters (varies):** 0.1, 0.02, 0.01, 0.005, 0.002

**Friction coefficient:** 0.2 (default, can be changed to sweep multiple values)

**Noise amplitude:** 1e-3

**System configuration:**
- Number of rods: 200
- Periodic box: [-1, 1]³
- Density: 1000.0

**Simulation parameters:**
- Time step: 0.001
- Total steps: 100,000
- Output interval: 100
- Soft contact model: k_scaler = 1e1, delta = 0.0002, nu = 0.1

**Total runs:** 5 (one for each aspect ratio, with single friction value)

## Usage

### Running the sweep

#### Submit individual jobs to cluster (RECOMMENDED)
```bash
cd parametric_study
python3 submit_aspect_ratio.py --job-name aspect_ratio_test
```

**Dry run test (recommended first):**
```bash
python3 submit_aspect_ratio.py --job-name test --dry-run
```

### Analyzing results

After all jobs complete:
```bash
cd parametric_study
python3 post_analyze_aspect_ratio.py \
    --job-name aspect_ratio_test \
    --make-plots \
    --outdir analysis_aspect_ratio_test
```

## Generated outputs

The analysis produces:
- **summary_table.csv** - Statistics for each aspect ratio
- **ke_traces_by_aspect_ratio.png** - KE time evolution for all aspect ratios
- **statistics_vs_aspect_ratio.png** - Mean KE and growth rate vs L/D
- **contacts_vs_aspect_ratio.png** - Average contact count vs L/D

## Output Structure

```
/n/holylabs/LABS/mahadevan_lab/Users/yjung/rod-dynamics-3d/runs/{job_name}/
├── run_dirs.txt                           # List of all run directories
├── parameter_summary.json                 # Parameter sweep configuration
├── combined_analysis_command.txt          # Command to run analysis
└── YYYYMMDD-HHMMSS_RUN_aspect_LD*_mu*/   # Individual run directories
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

## Expected Behavior

As the aspect ratio increases (rods become thinner relative to their length):

1. **Increased flexibility:** Higher L/D ratios lead to more flexible rods (though this code uses rigid rods, the effective "flexibility" comes from easier rotation and reorientation)

2. **Contact dynamics:** Thinner rods may have:
   - More contacts (easier to pack)
   - More complex entanglement patterns
   - Different collision frequencies

3. **Energy dissipation:** 
   - Friction effects may scale differently with rod geometry
   - Contact duration and overlap characteristics change
   - Rotational vs translational energy balance shifts

4. **Volume fraction:**
   - For fixed rod length, thinner rods have lower volume fraction
   - This affects the degree of confinement and crowding
   - May lead to different dynamical regimes

The analysis quantifies:
- How mean kinetic energy scales with L/D
- Whether energy grows, decays, or equilibrates
- Contact count dependence on aspect ratio
- Parameter regimes for different dynamical behaviors

## Modifying the sweep

To test multiple friction coefficients with aspect ratios, edit `submit_aspect_ratio.py`:

```python
# Change this line:
FRICTION_COEFFS = [0.2]  # Single value

# To this:
FRICTION_COEFFS = [0.0, 0.1, 0.2, 0.4]  # Multiple values
```

This will create a 2D parameter sweep: 5 aspect ratios × 4 friction values = 20 runs.

## Notes

- Each simulation takes approximately 5-15 minutes
- Memory usage: ~1-2 GB per simulation
- Disk usage: ~100-500 MB per run
- Very high aspect ratios (L/D=500) may require longer simulation times to reach steady state
- Contact detection becomes more important for thin rods
