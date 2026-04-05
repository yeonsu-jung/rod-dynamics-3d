# Frictional reptation (Mar 30)

Goal: simulating a single rod that reptates through a infinitely long cylinder. Want to get sliding length, collision frequency, etc.

## Setting

- A long test rod will be placed in the cylinder.
- Test rod specs: length and diameter will be given in inputs. density is fixed = 1.
- Initial condition: translational and rotational velocities will be given in inputs.
-- A discussion point: should we include this simulation within our current binary (./rigidbody_viewer) or making a new one? Somehow I want to make it easy to put different velocities outside of the binary. Then we will need a new cli option like "./rigidbody_viewer (...) --fix-every-except 1 --init-vel 1,1,1,1,1 (...)" Which I think it's okay. But want to discuss on this. Maybe it's time to discuss on trimming CLI options and trimming json files now...
-- Question: what will be the most natural probability distribution of velocities living in the tangent space of configuration space (R^3 x S^2)?
- Boundary condition: no collision in the axial direction of the containing cylinder. If the test rod hit the tube (the radial position of one of the tips >= cylinder radius), then the collision handling activated.
-- Question: can we handle the case where the test rod hit the cylinder wall in parallel way -- the rod is aligned with the continer.
- Collision handling: We will use hard contact model (nsc) with different mus.

## Outcomes to export

The end goal is draw plots including:

- Sliding length before it stops as a function of the container tube radius, friction coefficient, initial velocity (statistics), etc.

For that we will need this data:

- Sliding length before it stops.
- Collision statistics (waiting time, positions).
- Entire trajectory (optional)

## Current workflow notes

- `scripts/sweep_reptation.py` supports headless early stopping by axial slide speed with
  `--stop-slide-vel-threshold <V>` and `--stop-slide-vel-min-steps <N>`.
- When that threshold is provided, the sweep writes those settings into the combined CSV so
  post-processing can track which stopping criterion was used.
- `scripts/analyze_reptation.py` carries those stop columns through the per-run table together
  with `final_y`, making it easier to compare KE-based stops against axial-slide stops.
- For validation runs where the stopping metric itself is under study, do not pass any early-stop
  flags to the binary. Instead, record per-frame rod state with `--perrod` and determine stopping
  from postprocessed tangential velocity so the solver does not terminate the trajectory early.
- For the current reptation sweeps, the default analysis path is postprocessed first-crossing only:
  run the full trajectory with `--no-stop-ke --perrod`, then analyze with
  `scripts/analyze_reptation_tangent_stop.py --mode first`.
- Gap inputs are in rod-length units, but the sweep driver can interpret them in two ways:
  `--gap-radius-basis radius` means `R_cyl = rod_radius + gap`, while
  `--gap-radius-basis diameter` means `R_cyl = rod_diameter + gap`.
  Use the diameter basis when the requested tube radius is explicitly `rod diameter + gap`.
  In both cases, output tags and postprocessed summaries should keep the requested gap inputs,
  not the derived wall-clearance value.
- Non-thermal reptation sweeps now support four useful families:
  `--fixed-reptation` for constant `(vn, vt, va, w)`,
  `--init-mode gaussian-axial-transverse` for component-wise Gaussian samples in `(vn, vt, va, w)`,
  `--init-mode gaussian-isotropic` for i.i.d. Gaussian samples in all Cartesian velocity and angular-velocity components,
  and `--thermal` for native C++ Maxwell-Boltzmann initialization.

Example:

```bash
python scripts/sweep_reptation.py \
  --exe ./build/rigidbody_viewer_3d \
  --scene assets/scenes/reptation.json \
  --out-dir results/reptation_ar200_thermal \
  --thermal --nsc \
  --rod-length 1.0 --aspect-ratio 200 \
  --gaps 0.001 0.01 0.1 \
  --mus 0.0 0.1 0.2 0.4 1.0 \
  --perrod --perrod-stride 100 \
  --stop-slide-vel-threshold 1e-5 \
  --stop-slide-vel-min-steps 5000

python scripts/analyze_reptation.py \
  --out-dir results/reptation_ar200_thermal \
  --combined-out results/reptation_ar200_thermal/final_y_runs.csv \
  --summary-out results/reptation_ar200_thermal/final_y_summary.csv
```

## Sweep process for first-stop reptation studies

Use this process when the objective is stopping statistics rather than solver-side early termination.

1. Choose rod geometry from `--rod-length` and `--aspect-ratio`.
2. Choose the tube convention.
   For the recent multi-AR sweeps, use `--gap-radius-basis diameter` so
   `R_cyl = rod_diameter + gap`.
3. Run the full sweep with `--no-stop-ke --perrod --perrod-stride 100 --perrod-max 25000`.
4. Analyze only the first stop with `scripts/analyze_reptation_tangent_stop.py --mode first`.
5. Plot sliding length from `tangent_stop_summary.csv` only. Do not generate sustained-stop outputs unless explicitly requested.

Template:

```bash
python scripts/sweep_reptation.py \
  --exe ./build/rigidbody_viewer_3d \
  --scene assets/scenes/reptation.json \
  --out-dir <OUT_DIR> \
  --nsc \
  --rod-length 1.0 \
  --aspect-ratio <AR> \
  --gaps 0.001 0.002 0.005 0.01 0.02 0.05 0.1 \
  --gap-radius-basis diameter \
  --mus 0.01 0.01668100537200059 0.027825594022071243 0.046415888336127774 0.0774263682681127 0.1291549665014884 0.21544346900318834 0.3593813663804626 0.5994842503189409 1.0 \
  --trials 20 \
  --jobs 4 \
  --no-stop-ke \
  --perrod \
  --perrod-stride 100 \
  --perrod-max 25000

python scripts/analyze_reptation_tangent_stop.py \
  --input-dir <OUT_DIR> \
  --output <OUT_DIR>/tangent_stop_summary.csv \
  --threshold 1e-5 \
  --dt 0.001 \
  --mode first \
  --window 1

python scripts/plot_sliding_length_vs_gap_over_mu.py \
  --input <OUT_DIR>/tangent_stop_summary.csv \
  --scatter-output <OUT_DIR>/sliding_length_vs_gap_over_mu_scatter.png \
  --summary-output <OUT_DIR>/sliding_length_vs_gap_over_mu_summary.png \
  --csv-output <OUT_DIR>/sliding_length_vs_gap_over_mu_summary.csv
```

Initialization presets used in the current sweep set:

```bash
# 1. Constant: vn=0.1, vt=0, va=0.1, w=0
--fixed-reptation --fixed-vn 0.1 --fixed-vt 0.0 --fixed-va 0.1 --fixed-w 0.0

# 2. Constant: vn=0.1, vt=0, va=0.1, w=0.2
--fixed-reptation --fixed-vn 0.1 --fixed-vt 0.0 --fixed-va 0.1 --fixed-w 0.2

# 3. Gaussian reptation coordinates: sigma_vn=0.1, sigma_vt=0, sigma_va=0, sigma_w=0.2
--init-mode gaussian-axial-transverse --sigma-vn 0.1 --sigma-vt 0.0 --sigma-va 0.0 --sigma-w-reptation 0.2
```

## Soft-contact first-stop sweep process

To repeat the same first-stop matrix with the implemented soft-contact model, use
`assets/scenes/reptation_soft.json` and do not pass `--nsc`.

Requested initialization families for the soft-contact rerun:

```bash
# 1. Constant reptation kick: vn=0.1, va=0.1, vt=0, w=0
--fixed-reptation --fixed-vn 0.1 --fixed-vt 0.0 --fixed-va 0.1 --fixed-w 0.0

# 2. Isotropic Gaussian: all v and omega components i.i.d. Gaussian
--init-mode gaussian-isotropic --sigma-v 0.1 --sigma-w 0.2
```

The convenience launcher for this matrix is:

```bash
./scripts/run_reptation_gapdiam_first_sweeps_soft.sh
```

Output directory naming convention:

- `results/reptation_ar<AR>_soft_const_vn0p1_vt0_va0p1_w0_gapdiam_mugeom10_first`
- `results/reptation_ar<AR>_soft_isogi_sv0p1_sw0p2_gapdiam_mugeom10_first`

<!-- round 0 -->

## Addition 1

- Overall it's great.
- Strongly agree with externalizing inputs with python runners.
- Strongly agree with sampling three points (endpoints + mid)
- Gravity off as default (but make it adjuatable).
- Restitution = 1 as default (but make it adjuatable).
- Do not have to consider wall friction. Make it equal to the rod's one; or remove it from the property.
- Yes in the beginning the test rod and tube is at the origin.
- We actually has no $\omega_\parallel$ -- do we have this? I thought we are only dealing with the one without azimuthal spin.

<!-- round 1 -->