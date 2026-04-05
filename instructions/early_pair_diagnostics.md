
# Early Pair Diagnostics

## Goal

Add early-time diagnostics that let us inspect:

- active contacting pairs and their relative contact kinematics
- all-pair signed distances or gaps
- behavior over a limited startup window, typically steps 100 to 10000

The diagnostics should be usable while switching between NSC and soft-contact solvers.

## Scope

This note is about debug instrumentation, not a new physics model.

The output should answer two questions:

1. Which pairs are geometrically close during early dynamics?
2. Which of those pairs are treated as active contacts by the current solver, and with what relative velocity?

## Existing Hooks

### Contact Geometry

The codebase already exposes detected contact geometry in multiple solvers:

- `SoftContactSolver::getContacts()` returns `ContactPrimitive`
- `MujocoContactSolver::getContacts()` returns `MujocoContactSolver::Contact`
- `NscContactSolver` exposes `getManifolds()`, and internally reuses soft-contact detection

### Pair Distance Logic

`App::minPairGap()` already contains robust pair-gap geometry for sphere-sphere,
capsule-capsule, and sphere-capsule combinations.

That logic should be generalized into a reusable helper that returns per-pair gap,
not only the minimum gap.

## Required Outputs

Use two separate streams.

### 1. Active Contact Pair Diagnostics

Suggested file:

- `pair_contact_velocity_early.csv`

Suggested columns:

- `frame`
- `solver`
- `body_a`
- `body_b`
- `point_ax`, `point_ay`, `point_az`
- `point_bx`, `point_by`, `point_bz`
- `normal_x`, `normal_y`, `normal_z`
- `signed_gap`
- `surface_limit`
- `distance`
- `v_rel_x`, `v_rel_y`, `v_rel_z`
- `v_n`
- `v_t`

Optional NSC-only extension columns:

- `lambda_n`
- `lambda_t1`
- `lambda_t2`
- `v_n_post`
- `v_t_post`

### 2. All-Pair Distance Diagnostics

Suggested file:

- `pair_distance_early.csv`

Suggested columns:

- `frame`
- `body_a`
- `body_b`
- `signed_gap`
- `distance_metric`
- `surface_limit`
- `pair_type`

For dense runs this may need to become binary or stride-subsampled.

## Sampling Policy

The diagnostics should be gated by:

- `enabled`
- `start_step`
- `end_step`
- `stride`

Default debug window:

- `start_step = 100`
- `end_step = 10000`
- `stride = 1`

### Example JSON

```json
{
  "diagnostics": {
    "early_pairs": {
      "enabled": true,
      "start_step": 100,
      "end_step": 10000,
      "stride": 1,
      "contact_output_path": "pair_contact_velocity_early.csv",
      "pair_distance_output_path": "pair_distance_early.csv",
      "pair_distance_cutoff": 0.05,
      "binary_pair_distance_output": false
    }
  }
}
```

CLI overrides are also available for the same fields. JSON is the better
default when the diagnostics should travel with a scene configuration.

## Sampling Phase

For solver comparisons, sample at consistent phases.

### Contact Diagnostics

Capture immediately after contact detection and before force or impulse solve.

This keeps the meaning of relative contact velocity comparable across:

- soft penalty contact
- MuJoCo-style penalty contact
- NSC hard contact

### Pair Distance Diagnostics

Capture after positions are finalized for the step.

This makes the distance stream represent the actual evolved configuration at
that frame, not a partially updated state.

## Implementation Plan

### 1. Add App-Owned Diagnostics State

Near the existing diagnostic flags in `App`, add:

- `earlyPairDiagnosticsEnabled`
- `earlyPairStartStep`
- `earlyPairEndStep`
- `earlyPairStride`
- output paths for contact and pair-distance streams
- optional distance cutoff or binary mode for dense pair output

### 2. Introduce Shared Pair-Gap Geometry Helpers

Refactor the geometry currently embedded in `minPairGap()` into helpers that can:

- compute a signed gap for one pair
- classify the pair type
- iterate over all `(i, j)` pairs

This layer should remain independent from any contact solver.

### 3. Add Shared Relative-Kinematics Helper

Given:

- body A
- body B
- contact points on A and B
- contact normal
- optional PBC shift for B

compute:

- relative contact velocity
- normal component
- tangential component magnitude

This shared helper should be used by all solver adapters.

### 4. Add Solver Adapters

For each solver, convert native detected-contact data into a common debug row.

- soft contact: adapt `ContactPrimitive`
- MuJoCo contact: adapt `MujocoContactSolver::Contact`
- NSC: prefer adapting the underlying detected geometry rather than manifold-only state

### 5. Log Active Contacts

At the end of the detection stage, if the current frame is inside the configured
window, write one row per active detected contact.

### 6. Log All-Pair Distances

After positions are finalized, iterate over all body pairs and write signed gap
rows for the configured window.

## Performance Considerations

All-pair distance logging is expensive.

For `N` bodies and `T` sampled steps, row count is approximately:

`N * (N - 1) / 2 * T`

Examples:

- `N = 200`, `T = 10000` gives about 199 million rows
- `N = 1000`, `T = 10000` is too large for CSV in normal practice

Mitigations:

- allow `stride > 1`
- allow a gap cutoff for pair logging
- estimate output size up front and warn
- support a compact binary format for dense pair streams

## Recommended Rollout

### Phase 1

Implement contact diagnostics plus all-pair distances for NSC runs.

This is the lowest-risk path because NSC already computes useful per-contact
kinematics and exposes stable detection points.

### Phase 2

Move the contact diagnostics to a solver-agnostic adapter layer so the same CSV
schema works for:

- NSC
- soft contact
- MuJoCo contact

### Phase 3

Add dense-run protections:

- row-count estimate
- stride control
- optional binary output

## Validation

First validate on a short headless run, such as steps 100 to 300.

Check:

- row counts are as expected
- body-pair ordering is stable
- PBC handling matches solver detection logic
- contact rows appear only for active detected contacts
- signed-gap rows agree with near-contact events

## Relationship To Other Notes

This note focuses on the diagnostics feature.

The architectural split between contact detection and contact response is
described in:

- `instructions/contact_detection_common_geometry.md`
