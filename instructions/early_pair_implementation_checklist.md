# Early Pair Implementation Checklist

## Goal

Translate the design notes into a concrete implementation sequence with clear
file targets and minimal disruption to existing solver behavior.

## Phase 1: Common Geometry Scaffolding

### Phase 1 Files

- `include/physics/contact_geometry.hpp`
- `src/physics/contact_geometry.cpp`
- `include/physics/contact_geometry_adapters.hpp`
- `src/physics/contact_geometry_adapters.cpp`

### Phase 1 Tasks

- add `CommonContactGeometry`
- add `ContactKinematics`
- add shared relative-contact-kinematics helper
- add adapter from `ContactPrimitive`
- add adapter from `MujocoContactSolver::Contact`
- expose NSC detector contacts through the existing soft-contact detector path

## Phase 2: Config Surface

### Phase 2 Files

- `include/config/config.hpp`
- `src/config/config.cpp`
- `src/app/main.cpp`

### Phase 2 Tasks

- add `EarlyPairDiagnosticsCfg`
- add `DiagnosticsCfg`
- load `diagnostics.early_pairs` from JSON
- add CLI overrides for start, end, stride, contact path, distance path,
  cutoff, and binary mode
- store resolved config inside `App`

## Phase 3: App Integration Scaffolding

### Phase 3 Files

- `src/app/main.cpp`

### Phase 3 Tasks

- add app-owned early-pair diagnostics state
- add `configureEarlyPairDiagnostics()`
- add `shouldSampleEarlyPairDiagnostics()`
- prepare stable hook points for detection-phase and post-step sampling

## Phase 4: Active Contact Logging

### Phase 4 Files

- `src/app/main.cpp`

### Phase 4 Tasks

- log active detected contacts immediately after detection
- compute kinematics with shared helper
- use common CSV schema across soft contact and NSC

## Phase 5: All-Pair Distance Logging

### Phase 5 Files

- `src/app/main.cpp`
- optional future extraction to shared geometry utilities

### Phase 5 Tasks

- generalize `minPairGap()` into per-pair helpers
- iterate over all pairs after positions are finalized
- support stride and optional cutoff to control output size

## Validation

- build the project after each phase
- verify CLI help includes the new flags
- test a short headless run with JSON-only config
- test the same run with CLI overrides
- verify NSC and soft-contact paths both resolve the common contact geometry
