# Hard Contact / NSC Friction Implementation Plan (Rods Only)

## Goal

Add a **Non-Smooth Contact (NSC)** solver with Coulomb friction for **rod (capsule) bodies** alongside the existing soft (penalty) contact pipeline. Users switch between them via a single CLI flag (`--nsc`) and can benchmark both approaches on the same scene.

> **Scope**: Capsule-capsule contacts only. No sphere support in this solver.

---

## Why Spheres Are Not Needed

The PSOR solver is **shape-agnostic** — it operates on contact manifolds (normal, gap, contact points), not geometry. The existing `SoftContactSolver` narrowphase already produces `ContactPrimitive` for all three capsule contact types:

| ContactType | Geometry |
|-------------|----------|
| `EDGE_TO_EDGE` | Two capsule axes are closest at interior points |
| `EDGE_TO_POINT` | One capsule endpoint is closest to the other's axis |
| `POINT_TO_POINT` | Two capsule endpoints are closest to each other |

Each `ContactPrimitive` provides `normal`, `distance`, `point_a`, `point_b` — exactly what the NSC solver consumes as constraints. No intermediate sphere step is necessary.

---

## Architecture Overview

```
Current:  physicsStep()
          └── Soft Contact     (rods/capsules, penalty Verlet)

Proposed: physicsStep()
          ├── Soft Contact     (rods/capsules, penalty Verlet)       ← default
          └── NSC Contact      (rods/capsules, impulse-based PSOR)   ← --nsc
```

The NSC path uses a **different time-stepping scheme** from Velocity Verlet — it follows Project Chrono's `ChTimestepperEulerImplicitProjected` pattern:

```
1. v_free = v_old + M⁻¹ · f_ext · dt          (free flight prediction)
2. Detect capsule-capsule contacts at current positions
3. Solve velocity-level PSOR with friction cones → v_new
4. x_new = x_old + v_new · dt                  (position update)
5. Position projection (optional stabilization) (normal-only PSOR on positions)
```

This is the algorithm from `tests/demo_CH_friction_timestepping.py`, extended to 3D rigid bodies with 6 DOF (translation + rotation).

---

## Files to Create / Modify

### New files

| File | Purpose |
|------|---------|
| `include/physics/nsc_contact.hpp` | `NscContactSolver` class + `NscContactCfg` struct |
| `src/physics/nsc_contact.cpp` | PSOR velocity solver, friction cone projection, position stabilization |

### Modified files

| File | Change |
|------|--------|
| `include/config/config.hpp` | Add `NscContactCfg` struct to `PhysicsCfg` |
| `src/config/config.cpp` | JSON loading for NSC parameters |
| `src/app/main.cpp` | New `physicsStep()` branch, CLI flags, solver init |
| `CMakeLists.txt` | Add `nsc_contact.cpp` to sources |

---

## Phase 1: Data Structures & Config

### `NscContactCfg` (in `config.hpp`)

```cpp
struct NscContactCfg {
    bool enabled = false;

    // Solver parameters
    float mu = 0.3f;                // Coulomb friction coefficient
    float beta = 0.2f;              // Baumgarte stabilization factor (velocity level)
    float cfm = 0.0f;               // Constraint Force Mixing (regularization)
    float omega = 1.0f;             // SOR relaxation factor (1.0 = Gauss-Seidel)
    int velocity_iters = 40;        // PSOR iterations for velocity solve

    // Position stabilization
    bool position_stabilization = true;
    int position_iters = 5;         // Outer loops of position projection
    int position_psor_iters = 50;   // Inner PSOR solves per position loop
    float slop = 1e-4f;             // Allowed penetration before correction

    // Broadphase (reuses SoftContactSolver infrastructure)
    bool use_spatial_hash = true;
    double cell_size = -1.0;        // Auto if <=0
    bool use_aabb = true;
};
```

### `NscManifold` (internal to nsc_contact.hpp)

Each active contact stores Jacobian-derived quantities + accumulated impulses:

```cpp
struct NscManifold {
    int body_a, body_b;
    glm::vec3 normal;               // A→B
    glm::vec3 t1, t2;              // Tangent basis on contact plane
    glm::vec3 r_a, r_b;            // Lever arms: contact_point - body_center
    float phi;                      // Signed gap (negative = penetrating)

    // Cached diagonal of J·M⁻¹·Jᵀ (avoid recomputing each iteration)
    float g_n;                      // Normal direction
    float g_t1, g_t2;              // Tangent directions

    // Accumulated impulses (warm-starting candidate)
    float lambda_n = 0.0f;          // Normal impulse (≥ 0)
    float lambda_t1 = 0.0f;         // Tangent impulse direction 1
    float lambda_t2 = 0.0f;         // Tangent impulse direction 2
};
```

### Tangent Basis Construction

3D friction requires two tangent directions. Given contact normal `n`, construct an orthonormal basis:

```cpp
inline void buildTangentBasis(const glm::vec3& n, glm::vec3& t1, glm::vec3& t2) {
    // Pick the axis least aligned with n to avoid degeneracy
    if (std::abs(n.x) < 0.9f)
        t1 = glm::normalize(glm::cross(n, glm::vec3(1, 0, 0)));
    else
        t1 = glm::normalize(glm::cross(n, glm::vec3(0, 1, 0)));
    t2 = glm::cross(n, t1);
}
```

---

## Phase 2: Solver Implementation (`nsc_contact.cpp`)

### 2a. Contact Detection — Reuse Existing Capsule Narrowphase

The NSC solver **reuses** the `SoftContactSolver` broadphase + narrowphase. The `ContactPrimitive` already contains everything:

```
ContactPrimitive:
  .type         → EDGE_TO_EDGE / EDGE_TO_POINT / POINT_TO_POINT
  .body_a, .body_b
  .normal       → A→B (unit vector)
  .distance     → signed (negative = overlap)
  .point_a      → contact point on body A (world space)
  .point_b      → contact point on body B (world space)
  .surface_limit → r_a + r_b (sum of radii)
```

**Strategy**: `NscContactSolver` holds a `SoftContactSolver` by composition for broadphase/narrowphase, then converts `ContactPrimitive` → `NscManifold` for the constraint solver.

### 2b. Jacobian Structure (3D Capsule Rigid Bodies, 6 DOF)

For body A at contact point `p` with direction `d` (normal or tangent):

```
r_a = point_a - x_a                    (lever arm from COM to contact)
r_b = point_b - x_b

Impulse application (translational + rotational):
  A.v -= d * delta_lambda / m_a
  A.w -= I_a_w_inv * (r_a × d) * delta_lambda
  B.v += d * delta_lambda / m_b
  B.w += I_b_w_inv * (r_b × d) * delta_lambda
```

The key difference from the 2D Python demo is the **rotational terms**: capsules are long, so contact-induced torques are large and critical for realistic behavior.

### 2c. Diagonal Computation

```
g_n = (1/m_a + 1/m_b)
    + (r_a×n)ᵀ · I_a⁻¹ · (r_a×n)
    + (r_b×n)ᵀ · I_b⁻¹ · (r_b×n)
    + cfm
```

For capsules with high aspect ratio, the rotational contribution dominates because:
- Moment of inertia about the long axis is small (easy to spin)
- Moment about transverse axes is large (hard to tumble)
- Cross products `r × n` couple these differently depending on where on the capsule the contact occurs

This makes the rotational terms non-negotiable — a translation-only solver would be qualitatively wrong for rods.

### 2d. PSOR Velocity Solve

```cpp
void NscContactSolver::solveVelocities(
    std::vector<RigidBody>& bodies,
    float dt)
{
    for (int iter = 0; iter < cfg_.velocity_iters; ++iter) {
        for (auto& m : manifolds_) {
            auto& A = bodies[m.body_a];
            auto& B = bodies[m.body_b];

            // Relative velocity at contact point
            glm::vec3 v_rel = (B.v + glm::cross(B.w, m.r_b))
                            - (A.v + glm::cross(A.w, m.r_a));

            // --- Normal (unilateral: λ_n ≥ 0) ---
            float vn = glm::dot(v_rel, m.normal);
            float b_n = cfg_.beta * m.phi / dt;
            float w_n = vn + b_n + cfg_.cfm * m.lambda_n;
            float delta_n = -(cfg_.omega / m.g_n) * w_n;
            float old_n = m.lambda_n;
            m.lambda_n = std::max(0.0f, old_n + delta_n);
            float true_dn = m.lambda_n - old_n;
            if (true_dn != 0.0f)
                applyImpulse(A, B, m, m.normal, true_dn);

            // --- Tangent 1 (box friction: |λ_t1| ≤ μ·λ_n) ---
            float vt1 = glm::dot(v_rel, m.t1);  // recompute after normal
            float w_t1 = vt1 + cfg_.cfm * m.lambda_t1;
            float delta_t1 = -(cfg_.omega / m.g_t1) * w_t1;
            float old_t1 = m.lambda_t1;
            float max_f = cfg_.mu * m.lambda_n;
            m.lambda_t1 = std::clamp(old_t1 + delta_t1, -max_f, max_f);
            float true_dt1 = m.lambda_t1 - old_t1;
            if (true_dt1 != 0.0f)
                applyImpulse(A, B, m, m.t1, true_dt1);

            // --- Tangent 2 (box friction: |λ_t2| ≤ μ·λ_n) ---
            float vt2 = glm::dot(v_rel, m.t2);
            float w_t2 = vt2 + cfg_.cfm * m.lambda_t2;
            float delta_t2 = -(cfg_.omega / m.g_t2) * w_t2;
            float old_t2 = m.lambda_t2;
            m.lambda_t2 = std::clamp(old_t2 + delta_t2, -max_f, max_f);
            float true_dt2 = m.lambda_t2 - old_t2;
            if (true_dt2 != 0.0f)
                applyImpulse(A, B, m, m.t2, true_dt2);
        }
    }
}
```

### 2e. Impulse Application (with rotational DOFs)

```cpp
inline void applyImpulse(RigidBody& A, RigidBody& B,
                         const NscManifold& m,
                         const glm::vec3& dir, float lambda) {
    // Body A: receives negative impulse
    A.v -= dir * (lambda * A.invMass);
    A.w -= A.IworldInv() * glm::cross(m.r_a, dir) * lambda;
    // Body B: receives positive impulse
    B.v += dir * (lambda * B.invMass);
    B.w += B.IworldInv() * glm::cross(m.r_b, dir) * lambda;
}
```

### 2f. Position Stabilization (Normal-Only)

After `x += v·dt`, run a normal-only PSOR on positions to remove residual penetration:

```cpp
int NscContactSolver::projectPositions(
    std::vector<RigidBody>& bodies)
{
    for (int outer = 0; outer < cfg_.position_iters; ++outer) {
        // Re-detect contacts at current positions (capsule-capsule)
        detector_.detectContacts(bodies);
        const auto& contacts = detector_.getContacts();

        // Filter to only penetrating beyond slop
        // Build normal-only manifolds
        // Run PSOR: correct positions (no friction, no velocity change)
        // Apply dx, dq directly to bodies
    }
}
```

---

## Phase 3: Integration into `physicsStep()`

The NSC path is a **peer** of the soft contact branch. When `--nsc` is passed:
- `soft_contact.enabled` stays true (for its broadphase/narrowphase)
- `nsc.enabled` controls which **time-stepper + solver** runs

```cpp
void App::physicsStep() {
    // ... apply external + random forces ...

    if (settings.physics.nsc.enabled) {
        // ===== NSC (Hard Contact) Semi-Implicit Euler for Rods =====

        // 1) Free-flight velocity prediction
        for (auto& rb : rods) {
            if (rb.invMass > 0) {
                rb.v += (rb.f * rb.invMass + gravity) * dt;
                // Angular: w += I_w⁻¹ · (tau - w×(I_w·w)) · dt
                glm::mat3 Iinv = rb.IworldInv();
                glm::vec3 gyro = glm::cross(rb.w, rb.R() * rb.I_body * glm::transpose(rb.R()) * rb.w);
                rb.w += Iinv * (rb.tau - gyro) * dt;
            }
        }

        // 2) Detect capsule-capsule contacts (reuses broadphase)
        nscSolver.detectAndBuildManifolds(rods);

        // 3) Solve velocity constraints with friction
        nscSolver.solveVelocities(rods, dt);

        // 4) Update positions + orientations
        for (auto& rb : rods) {
            rb.x += rb.v * dt;
            glm::quat wq(0, rb.w);
            rb.q += 0.5f * dt * wq * rb.q;
            rb.q = glm::normalize(rb.q);
        }

        // Apply damping
        for (auto& rb : rods) {
            rb.v *= (1.0f - g_lin_damp * dt);
            rb.w *= (1.0f - g_ang_damp * dt);
        }

        // 5) Position stabilization
        nscSolver.projectPositions(rods);

        // 6) Clear forces for next step
        for (auto& rb : rods) {
            rb.f = glm::vec3(0);
            rb.tau = glm::vec3(0);
        }

        lastHitCount = nscSolver.getNumContacts();
        keAfterIntegrate = totalKE();

    } else if (settings.physics.soft_contact.enabled) {
        // ... existing soft contact Velocity Verlet path (unchanged) ...
    }

    // ... sleep update, diagnostics, etc (shared) ...
}
```

---

## Phase 4: CLI & Config

### New CLI flags

```
--nsc                           Enable NSC (hard) contact solver for rods
--nsc-iters <N>                 Velocity PSOR iterations (default: 40)
--nsc-beta <f>                  Baumgarte factor (default: 0.2)
--nsc-cfm <f>                   Constraint regularization (default: 0.0)
--nsc-omega <f>                 SOR relaxation (default: 1.0)
--nsc-mu <f>                    Friction coefficient (default: 0.3)
--nsc-pos-iters <N>             Position stabilization outer iters (default: 5)
--nsc-pos-psor <N>              Position stabilization inner PSOR iters (default: 50)
--no-nsc-pos                    Disable position stabilization
```

### JSON scene support

```json
{
  "physics": {
    "nsc": {
      "enabled": true,
      "mu": 0.3,
      "velocity_iters": 40,
      "beta": 0.2,
      "position_stabilization": true
    }
  }
}
```

### Benchmark workflow

```bash
# Soft contact baseline
./rigidbody_viewer_3d --scene test.json --headless --steps 10000 --csv soft.csv

# Hard contact (NSC)
./rigidbody_viewer_3d --scene test.json --headless --steps 10000 --nsc --csv nsc.csv

# Compare energy, contacts, timing
python compare_scenes_ke.py soft.csv nsc.csv
```

---

## Phase 5: Implementation Order

### Step 1: Scaffold (compiles, no-op) ✅
- [x] Add `NscContactCfg` to `config.hpp`
- [x] Create `nsc_contact.hpp` with class stub
- [x] Create `nsc_contact.cpp` with empty implementations
- [x] Add to `CMakeLists.txt`
- [x] Add `--nsc` CLI flag + related flags
- [x] Add dispatch branch in `physicsStep()` (calls empty solver)
- **Verify**: builds and runs, `--nsc` flag accepted, no crashes ✅

### Step 2: Manifold builder + velocity solver ✅
- [x] Implement `detectAndBuildManifolds()`: call `SoftContactSolver::detectContacts()`, convert `ContactPrimitive` → `NscManifold`
- [x] Implement tangent basis construction
- [x] Implement diagonal `g_n`, `g_t1`, `g_t2` computation (with rotational terms)
- [x] Implement `solveVelocities()` PSOR with normal + 2 tangent directions
- [x] Implement `applyImpulse()` with translational + rotational terms (inlined in solver loop)
- [x] Wire up semi-implicit Euler time-stepper in `physicsStep()`
- **Verify**: two crossing rods collide and bounce apart, energy doesn't explode ✅

### Step 3: Position stabilization ✅
- [x] Implement `projectPositions()` (normal-only PSOR on capsule contacts with 3D rotational corrections)
- **Verify**: rods under gravity settle without drifting through each other ✅

### Step 4: Tuning & benchmarks ✅
- [x] Test with existing rod scenes (confined, PBC, right-angle collision)
- [x] Add warm-starting (reuse previous frame's λ values via body-pair hash map)
- [x] Add solver residual + impulse sums (jn_sum, jt_sum, nsc_residual) to CSV output
- [x] Profile and compare: soft contact vs NSC (see benchmark below)

### Benchmark results (300 rods, PBC, dt=0.001, 2000 frames)

Scene: `benchmark_nsc_vs_soft.json` — 300 rods (L=0.50, D=0.08), PBC box 3×2×3,
random init, gravity=[0,-2,0], NSC: 40 PSOR iters, mu=0.3, beta=0.2.

| Metric                | Soft (penalty)  | NSC (PSOR)      |
|-----------------------|-----------------|-----------------|
| integrate (ms/frame)  | 0.223           | 0.084           |
| broadphase (ms/frame) | 3.528           | 1.561           |
| solve (ms/frame)      | 0.046           | 1.974           |
| **total phys (ms/frame)** | **3.80**    | **3.62**        |
| contacts/frame (mean) | 36.5            | 109.8           |
| contacts/frame (max)  | 63              | 164             |
| KE (start → end)      | 39.4 → 39.3    | 39.4 → 62.2    |
| soft PE (mean)        | 1.17            | 0.00            |

**Key takeaways**:
- NSC solve cost ~43× higher than soft (PSOR iterations vs simple force accumulation)
- Total per-frame cost is comparable because broadphase dominates in the soft pathway
- NSC detects ~3× more contacts (all overlapping capsule pairs generate manifolds)
- Under gravity+PBC, NSC allows more KE accumulation; soft contact stores energy elastically

---

## Rod-Specific Considerations

### Why rods make the rotational solver essential

| Property | Value | Implication |
|----------|-------|-------------|
| Aspect ratio L/(2r) | 5–50 | Inertia tensor is highly anisotropic |
| I_long / I_transverse | ~0.01–0.1 | Easy to spin about axis, hard to tumble |
| Contact lever arm | up to L/2 | Large torques from end contacts |
| Multiple contacts per rod | common | Edge-to-edge along parallel rods |

A translation-only solver would miss the torque-dominated dynamics entirely. The full 6-DOF `applyImpulse` with `I_world_inv` is required.

### Capsule-specific manifold notes

- **Edge-to-edge**: Most common for entangled rods. Contact normal is perpendicular to both rod axes. The lever arms `r_a`, `r_b` are large → strong torque coupling.
- **Edge-to-point**: One rod's endcap touches another's shaft. The endcap body has a short lever arm, the shaft body has a potentially long one.
- **Point-to-point**: Two endcaps touching. Nearly spherical locally. Lever arms are both approximately `half_height * axis + radius * normal`.

The existing narrowphase handles all three and returns correct `point_a`, `point_b` in world space. No modifications needed.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Friction cone approximation | Box (decouple t1, t2) | Standard in Chrono/Bullet; exact cone needs inner iteration |
| Broadphase | Reuse `SoftContactSolver`'s spatial hash | Avoid duplicating broadphase code; already handles PBC |
| Time integrator | Semi-implicit Euler (not Verlet) | Required for impulse-based contacts; Verlet assumes smooth forces |
| Tangent basis | Compute fresh each step | No warm-starting of tangent directions (simpler, robust) |
| Position correction | Separate PSOR pass | Matches Chrono pattern; cleanly separates velocity/position levels |
| Rotational DOFs in solver | Full 3D (6 DOF per body) | Critical for rods — torques dominate contact response |
| Shape support | Capsules only (for now) | Matches the simulation's purpose; spheres can be added later if needed |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| PSOR convergence too slow for dense rod packing | Warm-starting + increase iters; Chrono handles 10k+ contacts |
| Energy gain from Baumgarte | Use conservative `beta=0.2`; position stabilization absorbs drift |
| Anisotropic inertia causes instability | Angular velocity clamping (existing `g_w_max`) acts as safety net |
| Multiple contacts per rod pair (edge-to-edge is extended) | One contact per narrowphase pair is sufficient for rigid capsules |
| PBC image contacts | Existing broadphase handles PBC; `ContactPrimitive.shift_b` carries image offset |

---

## Reference

- Python prototype: `tests/demo_CH_friction_timestepping.py` (2D version of this algorithm)
- Project Chrono `ChTimestepperEulerImplicitProjected`: semi-implicit Euler + PSOR contact solver
- The `SolverConfig` struct in `types.hpp` was already partially set up for this (Baumgarte, velIters, splitImpulse)
