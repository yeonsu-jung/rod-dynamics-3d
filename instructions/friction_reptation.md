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