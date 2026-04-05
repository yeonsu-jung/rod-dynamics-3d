# Contact Detection and Common Geometry Layer

## Goal

Separate contact detection geometry from contact response so that:

- NSC and soft-contact solvers can share the same detected-contact description
- diagnostics can compare solvers without being biased by solver internals
- future solver experiments can reuse contact geometry without duplicating code

This note is about architecture, not just logging.

## Problem

Right now the codebase has three overlapping contact paths:

- `SoftContactSolver`
- `MujocoContactSolver`
- `NscContactSolver`

They all need contact geometry, but they package it differently:

- soft contact exposes `ContactPrimitive`
- MuJoCo contact exposes its own `Contact`
- NSC exposes `NscManifold`, which includes solver-specific constraint state

That makes it easy to compare final behavior but harder to compare the geometry
that each solver started from.

## Design Principle

The shared layer should represent detected contact geometry, not solver state.

This means:

- include body ids, points, normals, gaps, and optional PBC shift
- exclude solver-specific fields such as impulses, friction history, or PSOR state

Solver-specific response data can be added later as an optional augmentation.

## Proposed Layering

### 1. Pair Geometry Layer

Responsible for:

- closest-point geometry
- contact points on both bodies
- contact normal
- signed distance or signed gap
- surface limit or equivalent geometric threshold
- PBC-aware shifted coordinates when needed

This layer should not know whether the downstream solver is penalty-based or
impulse-based.

### 2. Contact Detection Layer

Responsible for deciding whether a pair should produce a detected contact event.

This layer can still differ between solvers if necessary, but the event it emits
should be convertible into the same common geometry record.

### 3. Contact Response Layer

Responsible for applying:

- penalty forces
- smooth friction forces
- hard-constraint impulses
- stabilization or projection

This is where NSC and soft contact intentionally diverge.

## Common Contact Geometry Record

Suggested shared struct:

```cpp
struct CommonContactGeometry {
  int bodyA = -1;
  int bodyB = -1;

  glm::vec3 pointA{0.0f};
  glm::vec3 pointB{0.0f};
  glm::vec3 normal{0.0f};

  double distance = 0.0;
  double surfaceLimit = 0.0;
  double signedGap = 0.0;

  glm::vec3 shiftB{0.0f};
  bool isWall = false;
};
```

Notes:

- `distance` is the raw geometric distance returned by the detector
- `surfaceLimit` is the sum of radii or equivalent threshold
- `signedGap = distance - surfaceLimit`
- `shiftB` preserves PBC-consistent lever arms and relative velocity calculations

## Why This Boundary Works

### Soft Contact

`ContactPrimitive` already contains almost all required fields and is very close
to the proposed common record.

### MuJoCo Contact

The MuJoCo-style contact record also already contains the essential geometry,
with only minor differences in naming and no explicit PBC shift.

### NSC

NSC manifolds contain both geometry and solver state. They are useful for the
response layer but are too specialized to serve as the common abstraction.

Since NSC already reuses soft-contact detection, the better shared boundary is
the detected contact geometry before manifold construction.

## Relative Kinematics Should Be Shared Too

Relative contact kinematics should be computed from bodies plus contact geometry,
not stored as solver-owned truth.

Suggested helper inputs:

- body A
- body B
- contact point on A
- contact point on B
- contact normal
- optional `shiftB`

Suggested outputs:

- relative contact velocity vector
- normal relative velocity
- tangential relative velocity magnitude

This lets diagnostics compare soft contact and NSC using the same definition of
contact velocity.

## Recommended Refactor Path

### Phase 1: Non-Invasive Adapters

Do not rewrite solvers yet.

Instead, add adapter functions:

- `toCommonContactGeometry(const ContactPrimitive&)`
- `toCommonContactGeometry(const MujocoContactSolver::Contact&)`
- `toCommonContactGeometry(...)` for the NSC detection stage

This is enough to support solver-agnostic diagnostics.

### Phase 2: Shared Utilities

Move these helpers into a neutral location:

- signed-gap geometry helpers
- relative-contact-kinematics helper
- pair classification helper

At this stage, diagnostics and analysis tools can stop depending on solver
implementation details.

### Phase 3: Optional Stronger Architectural Split

If future work justifies it, introduce a dedicated detector output type and let:

- soft contact consume it to compute forces
- NSC consume it to build manifolds
- MuJoCo contact consume it to compute penalty forces

That is a larger refactor and should not block diagnostics.

## Benefits

- makes solver comparison meaningful at the geometry level
- reduces duplicate contact-geometry logic
- gives diagnostics a single stable schema
- makes it easier to debug disagreements between solvers
- keeps solver-specific response details where they belong

## Risks

- over-abstracting too early could slow down solver changes
- a forced common detector may accidentally erase solver-specific heuristics
- dense diagnostics built on the common layer can still be expensive

The right immediate move is an adapter-based common layer, not a full solver
rewrite.

## Immediate Recommendation

Use this note as the architectural basis for early pair diagnostics.

Implement:

1. a common detected-contact geometry record
2. adapter functions per solver
3. shared relative-contact-kinematics computation
4. all-pair gap logging outside the solver

Do not couple the diagnostics format to NSC manifold fields.
