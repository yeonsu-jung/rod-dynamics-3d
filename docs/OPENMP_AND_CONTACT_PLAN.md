# OpenMP Reporting And Contact Plan

## Immediate Changes

- Make startup logging report the OpenMP runtime default, the requested `--threads` limit, and the effective runtime max after the limit is applied.
- Add one-time diagnostics inside shared OpenMP contact regions so a run reports the actual team size used by the runtime, not just the machine maximum.

## Why

- The previous startup banner printed `omp_get_max_threads()` before the CLI thread limit had been pushed into the OpenMP runtime.
- That made logs such as `Max threads: 24` look like the solver ignored `--threads 4`, even when later parallel regions were actually capped.
- A one-time in-region diagnostic removes ambiguity by reporting the real team size seen inside OpenMP work-sharing regions.

## Current Architecture Note

- `NscContactSolver` currently reuses `SoftContactSolver` for broadphase and narrowphase detection.
- That coupling is convenient, but it mixes contact detection with one specific response model.

## Follow-Up Refactor Plan

- Extract contact detection into a dedicated shared detector layer.
- Move broadphase, capsule geometry, AABB helpers, and contact primitive generation into that detector.
- Make both soft-contact and NSC consume the same detector output without either solver owning the other.
- Keep penalty-force computation in `SoftContactSolver` only.
- Keep manifold construction and PSOR in `NscContactSolver` only.

## Expected Outcome

- Thread logs become trustworthy for batch runs.
- NSC and soft contact remain behaviorally unchanged in the short term.
- The later detector split can remove the current architectural dependency without duplicating collision code.

## Refactor Status

- The shared detector split is now implemented as `ContactDetector`.
- Broadphase, narrowphase, AABB helpers, and contact primitive generation now live in the detector layer.
- `SoftContactSolver` now delegates detection to `ContactDetector` and keeps only force, damping, friction, and history logic.
- `NscContactSolver` now depends on `ContactDetector` directly instead of depending on `SoftContactSolver`.