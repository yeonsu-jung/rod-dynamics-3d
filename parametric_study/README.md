# Parametric Study Analysis Tools

This directory contains tools for analyzing rod ensemble dynamics from the simulation.

## Analysis Modules

### 1. Rigid Motion Decomposition (`rigid_motion_decomposition.py`)

Implements the theory from `RodMotionDecomposition.md`:

**Key Features:**
- **Twist fitting**: Computes best-fit global rigid motion (ω, v₀) from velocity field
- **Energy decomposition**: T_total = T_global + T_def (exact orthogonal decomposition)
- **Kabsch algorithm**: Position alignment between frames
- **Screw motion parameters**: Axis, rotation angle, and pitch

**Usage:**
```bash
python3 rigid_motion_decomposition.py <per_rod_csv>
```

**Output:**
- `rigid_decomposition.csv` - Time series of twist parameters and energies
- Energy decomposition plots (total, global, deformational)
- Angular and translational velocity magnitudes
- Screw motion parameters
- Fit quality (residual RMS)

**Physical Interpretation:**
- **Global KE**: Energy in bulk translation + rotation of entire ensemble
- **Deformational KE**: Energy in relative rod motions (collisions, rearrangements, entanglement)
- **Residual**: How well the ensemble behaves as a rigid body (0 = perfect rigid motion)

### 2. Pairwise Metrics (`pairwise_metrics.py`)

Computes structural and dynamic properties of rod pairs:

**Metrics:**
1. **Average pairwise distance**: ⟨d_ij⟩ between rod centers
2. **Average pairwise angle**: ⟨θ_ij⟩ between rod axes (in degrees)
3. **Orientation autocorrelation**: Temporal correlation of cos(θ(t, t+δt))
4. **Distance autocorrelation**: Temporal correlation of d_ij(t)

**Usage:**
```bash
# Single file analysis
python3 pairwise_metrics.py <per_rod_csv>

# Parametric sweep analysis
python3 pairwise_metrics.py --sweep <csv_directory>
```

**Output:**
- `pairwise_metrics.csv` - Summary metrics for each configuration
- Plots vs n (number of rods) and AR (aspect ratio)
- Heatmaps of metrics in (n, AR) space

### 3. Combined Analysis (`combined_analysis.py`)

Unified analysis combining rigid decomposition and pairwise metrics:

**Features:**
- Rigid motion decomposition at each timestep
- Time-resolved pairwise distances and angles
- Energy-structure phase space diagrams
- Comprehensive multi-panel visualizations

**Usage:**
```bash
python3 combined_analysis.py <per_rod_csv> [--output-dir DIR]
```

**Output:**
- `combined_analysis.csv` - All time series in one file
- Energy + distance correlation plots
- Rotation + angle correlation plots
- Phase space: energy fractions vs structural order
- 4-panel summary dashboard

**Key Insights:**
- Relates global motion (rigid) to local structure (pairwise)
- Identifies regimes: rigid-like vs fluid-like behavior
- Tracks structural evolution alongside energy dissipation

## Confined System Study (`confined_study.py`)

Study small confined systems with kinetic energy tracking.

**Features:**
- Adjustable number of rods (default: 10)
- Adjustable confinement (box size as fraction of rod length, default: 0.7)
- Automatic packing fraction calculation
- Initial kinetic energy control
- Detailed KE evolution analysis (total, linear, rotational)

**Usage:**
```bash
# Generate and run confined system
python3 confined_study.py --n-rods 10 --box-factor 0.7 --steps 5000

# Custom parameters
python3 confined_study.py --n-rods 15 --box-factor 0.5 --initial-ke 0.2 --steps 10000

# Generate scene only (no simulation)
python3 confined_study.py --n-rods 10 --box-factor 0.8 --generate-only

# Analyze existing data
python3 confined_study.py --analyze ../build/confined_n10_box0.70.csv
```

**Output:**
- Scene JSON in `../assets/scenes/`
- Per-rod trajectory CSV in `../build/`
- Analysis directory with:
  - `ke_summary.csv` - Time series of total/linear/rotational KE
  - `ke_evolution.png` - KE components over time
  - `ke_fractions.png` - Linear vs rotational partition
  - `ke_loss.png` - Cumulative energy dissipation
  - `ke_evolution_log.png` - Log scale view

**Physics:**
- No gravity, damping, or friction (clean collision dynamics)
- Elastic collisions (restitution = 1.0)
- Periodic boundary conditions
- High packing fractions → frequent collisions → rapid KE decay

## Scene Generation

### Scene Configuration (`generate_pairwise_scenes.py`)

Creates JSON scene files for parametric studies:

**Usage:**
```bash
python3 generate_pairwise_scenes.py
```

**Default sweep:** n ∈ {6, 7, 8}, AR ∈ {50, 150, 500}

**Output:** `../assets/scenes/pairwise_sweep/pairwise_n{n}_ar{ar}.json`

**Physics settings:**
- Periodic boundary conditions
- No gravity, damping, or friction (for clean dynamics)
- Elastic collisions (restitution = 1.0)
- Random initial positions/orientations/velocities
- Box size scaled by packing fraction (~0.15 for dilute system)

### Rod Placement Optimization (`optimize_rod_placement.py`)

Optimizes initial rod positions to avoid collisions while maintaining specific patterns:

**Methods:**
- Segment-segment distance computation for capsules
- Differential evolution optimization
- Constraint satisfaction (min clearance, max spread)

**Usage:** Used by scene generation scripts

### Choreographed Motion (`analyze_cone_overlap.py`)

Designs phase-shifted rotations for collision-free tumbling:

**Features:**
- Cone sweep geometry analysis
- Phase shift calculation for N rods
- Overlap verification
- Rotation choreography design

## Simulation Execution

### Parametric Sweep (`run_pairwise_sweep.sh`)

Automated script to run all configurations:

**Workflow:**
1. Generate scene files
2. Run headless simulations (5000 steps)
3. Export per-rod trajectories (1000 samples)
4. Save to `pairwise_sweep_csvs/`

**Usage:**
```bash
cd parametric_study
./run_pairwise_sweep.sh
```

**After completion:**
```bash
# Analyze all results
python3 pairwise_metrics.py --sweep pairwise_sweep_csvs

# Or analyze individual files
python3 combined_analysis.py pairwise_sweep_csvs/pairwise_n6_ar50.csv
```

## Example Workflows

### Quick Test on Existing Data
```bash
# Combined analysis on existing perrod.csv
python3 combined_analysis.py ../perrod.csv

# Output: combined_analysis_perrod/
#   - combined_analysis.csv
#   - combined_energy_distance.png
#   - combined_rotation_angles.png
#   - phase_space.png
#   - summary_4panel.png
```

### Full Parametric Study
```bash
# 1. Generate scenes
python3 generate_pairwise_scenes.py

# 2. Run simulations
./run_pairwise_sweep.sh

# 3. Analyze each configuration
for csv in pairwise_sweep_csvs/*.csv; do
    python3 combined_analysis.py "$csv"
done

# 4. Compare across parameter space
python3 pairwise_metrics.py --sweep pairwise_sweep_csvs
```

### Custom Analysis
```python
from rigid_motion_decomposition import load_trajectory_csv, fit_twist
from pairwise_metrics import compute_pairwise_distances

# Load data
positions, velocities, ang_vel, orientations, frames, masses = load_trajectory_csv('data.csv')

# Analyze specific frame
frame_idx = 500
omega, v0, residual = fit_twist(positions[frame_idx], velocities[frame_idx], masses)
dists = compute_pairwise_distances(positions[frame_idx])

print(f"Global rotation: |ω| = {np.linalg.norm(omega):.4f} rad/s")
print(f"Mean distance: {dists[dists > 0].mean():.4f}")
```

## Theory References

See `../RodMotionDecomposition.md` for mathematical foundations:
- Chasles' theorem (screw motions)
- SE(3) Lie group and se(3) algebra
- Twist representation
- Energy orthogonality proof
- Kabsch algorithm derivation

## Output Interpretation

### Typical Values (500 rods, AR~20, periodic box)
- **Global KE fraction**: ~2-10% (most energy in deformational motion)
- **Mean pairwise distance**: ~1-2 (depends on box size and packing)
- **Mean pairwise angle**: ~57° (random orientation → 45°, aligned → 0°, perpendicular → 90°)
- **Orientation autocorr**: ~0.97 (high = persistent tumbling)
- **Distance autocorr**: ~0.99 (high = stable structure)
- **Residual RMS**: ~0.1-0.5 (depends on collision intensity)

### Interpreting Phase Space Plots
- **High global fraction + low angles**: Coherent rotation of aligned bundle
- **Low global fraction + high angles**: Chaotic rearrangements
- **Trajectory over time**: Shows evolution from initial state to steady state

## Dependencies

```bash
pip install numpy scipy pandas matplotlib
```

## File Format

Per-rod CSV from simulator (`--perrod` flag):
```
frame,rod,px,py,pz,vx,vy,vz,wx,wy,wz,qw,qx,qy,qz,KE_lin,KE_rot,KE_total
0,0,0.123,0.456,0.789,...
0,1,-0.234,0.567,-0.890,...
...
```

Each frame contains N_rods entries (one per rod).
