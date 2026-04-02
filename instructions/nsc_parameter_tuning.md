# NSC Parameter Tuning Guide

This note summarizes practical guidance for tuning the hard-contact NSC solver in this repo.

The main NSC parameters are:

- `velocity_iters`: number of PSOR iterations in the velocity solve
- `beta`: Baumgarte stabilization factor for penetration correction
- `cfm`: constraint force mixing, acting like compliance/regularization
- `omega`: SOR relaxation factor for each iterative update
- `mu`: Coulomb friction coefficient

## What They Do

### `velocity_iters`

Controls how fully the coupled normal and friction constraints converge each step.

- Too low: weak friction, excess slip, solver-order dependence, more penetration drift.
- Higher: more consistent contact impulses, but more cost.

### `beta`

Controls how strongly penetration depth contributes to the normal correction bias.

- Too low: contacts may remain overlapped longer.
- Too high: dynamics become solver-driven; can look overly repulsive or artificially dissipative.

### `cfm`

Adds regularization/compliance to constraints.

- `0.0`: hardest contact.
- Small positive values: often improve robustness in dense systems.
- Too large: contact network becomes mushy and allows too much overlap/slip.

### `omega`

Scales each PSOR update.

- `1.0`: standard Gauss-Seidel-style update.
- `< 1.0`: under-relaxation, usually slower but more robust.
- `> 1.0`: over-relaxation, can help convergence, but can also overshoot or add noise.

## Suggested Defaults

### Single-Rod Reptation In A Tube

Recommended baseline:

```json
"nsc": {
  "enabled": true,
  "mu": 0.1,
  "velocity_iters": 80,
  "beta": 0.05,
  "cfm": 1e-6,
  "omega": 1.0
}
```

Reasoning:

- low enough `beta` to keep penetration correction from dominating the wall dynamics
- enough iterations that wall friction is reasonably converged
- tiny `cfm` for robustness without visibly softening the tube wall

### Dense Entangled NSC Runs

Recommended baseline:

```json
"nsc": {
  "enabled": true,
  "mu": 0.6,
  "velocity_iters": 200,
  "beta": 0.08,
  "cfm": 1e-6,
  "omega": 0.95
}
```

Reasoning:

- dense frictional networks usually need many iterations
- `beta = 0.2` is often too aggressive for production physics studies
- tiny `cfm` helps avoid pathological convergence in large coupled systems
- `omega = 0.95` is a good conservative alternative to `1.0`

## How To Tune In Practice

Do not sweep `mu` first. First establish that solver parameters are not dominating the result.

### Reptation Workflow

1. Convergence check

- Fix `mu = 0.1`, `beta = 0.05`, `cfm = 1e-6`, `omega = 1.0`
- Sweep `velocity_iters = 20, 40, 80, 160`

1. Stabilization sensitivity

- Fix `mu`, converged `velocity_iters`, `cfm`, `omega`
- Sweep `beta = 0.0, 0.02, 0.05, 0.1, 0.2`

1. Regularization sensitivity

- Sweep `cfm = 0, 1e-7, 1e-6, 1e-5`

1. Relaxation sensitivity

- Sweep `omega = 0.8, 0.9, 1.0, 1.05`

1. Friction study

- After the solver baseline is stable, sweep `mu = 0.0, 0.05, 0.1, 0.3, 0.6, 1.0`

### Dense Entangled Workflow

1. Sweep `velocity_iters = 50, 100, 200, 400`
1. Sweep `beta = 0.02, 0.05, 0.08, 0.15`
1. Sweep `cfm = 0, 1e-7, 1e-6, 1e-5`
1. Sweep `omega = 0.85, 0.95, 1.0`
1. Only then sweep `mu`

## How To Tell Physics From Numerics

Treat a friction result as physically meaningful only if it is relatively insensitive to `velocity_iters`, `beta`, `cfm`, and `omega` over a reasonable range.

Rules of thumb:

- If changing `mu` changes the outcome much more than changing the solver parameters, the effect is likely physical.
- If changing `iters`, `beta`, `cfm`, or `omega` changes the outcome as much as `mu`, the result is solver-sensitive.

## Recommended Observables

### Reptation

- net axial displacement
- total axial path length
- final kinetic energy
- wall contact count
- contact pre/post normal and tangential relative speeds
- penetration statistics

### Dense Entangled Systems

- total KE vs time
- contact count vs time
- penetration statistics
- topology/network statistics
- entanglement metrics

## Important Initialization Note

Be explicit about `randomInit.mode` in scene files.

If `randomInit.enabled = true` but `mode` is omitted, the config loader falls back to the default mode, which is currently `"thermal"` with default `kBT = 1.0`.

That can inject much larger initial kinetic energy than expected if the scene was written assuming legacy `vSigma` / `wSpeed` behavior.

If a scene is intended to use legacy bounded random velocities, set:

```json
"randomInit": {
  "enabled": true,
  "mode": "uniform",
  "vSigma": 0.01,
  "wSpeed": 0.02,
  "seed": 42,
  "projectParallelSpin": true
}
```
