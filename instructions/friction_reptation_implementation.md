# Frictional Reptation — Implementation Plan (Mar 30)

This document discusses design decisions and concrete implementation steps for the frictional reptation simulation described in [friction_reptation.md](friction_reptation.md).

---

## 1. Should we make a separate binary or extend `rigidbody_viewer`?

**Recommendation: extend the existing binary.**

Reasons:
- The existing binary already has `--fix-every-except`, `--set-velocity`, `--perturb-rod`, `--nsc-mu`, per-rod CSV, network CSV, and headless mode — all of which this experiment needs.
- A separate binary would duplicate the entire physics loop, contact solver, and output pipeline.
- The only truly new thing is the **cylindrical boundary** (see §3). Everything else is configuration.

What we *do* need to add/change:
1. A **cylinder boundary** collision primitive.
2. A **reptation scene JSON** that defines the tube + test rod.
3. A few **new CLI options** (or JSON fields) for cylinder geometry and initial angular velocity.
4. A **stopping condition** (KE threshold) to halt the simulation when the rod stops.
5. A **summary output** (sliding length, contact count) at the end.

This keeps the codebase unified and avoids bit-rot from maintaining a fork.

---

## 2. CLI & JSON additions

### 2.1 JSON scene: cylinder boundary

Add a new section under `scene`:

```json
"cylinder": {
  "enabled": true,
  "axis": [0, 1, 0],
  "radius": 0.3
}
```

- `axis` — unit vector along the cylinder's long direction (default Y). The cylinder is infinite in this direction.
- `radius` — inner radius of the tube.
- No separate wall friction — the cylinder uses the same $\mu$ as the rod (from `--nsc-mu` or the rod body's friction field). This avoids an extra parameter and matches the physics: friction is a property of the rod-wall pair, and we control it via the global `mu`.

Implementation in `config.hpp`:
```cpp
struct CylinderCfg {
    bool enabled = false;
    glm::vec3 axis{0, 1, 0};
    float radius = 0.3f;
};
```
Add it to `SceneCfg` and parse in `loadScene()`.

### 2.2 CLI: angular velocity + stopping condition

Existing `--set-velocity <ID> <vx> <vy> <vz>` sets linear velocity only. We need angular velocity too.

**Option A** (minimal): add `--set-ang-velocity <ID> <wx> <wy> <wz>` — mirrors the linear one.

**Option B** (combined): `--set-velocity <ID> <vx> <vy> <vz> <wx> <wy> <wz>` — 7 args after ID. This is a breaking change.

**Recommendation: Option A.** Avoids breaking existing scripts. Both flags can be combined.

Stopping condition:
```
--stop-ke <threshold>   Stop simulation when total KE drops below threshold (default: off).
```
This would let us do:
```bash
./rigidbody_viewer --headless --steps 1000000 \
  --scene reptation.json \
  --nsc --nsc-mu 0.5 \
  --set-velocity 0 0 1.0 0 \
  --set-ang-velocity 0 0 0 0.5 \
  --stop-ke 1e-8 \
  --perrod perrod.csv
```

---

## 3. Cylinder boundary collision

This is the core new feature. We need `collideCapsuleCylinder()`.

### 3.1 Geometry

The cylinder wall is the *inner surface* of an infinite tube of radius $R$ centered at the origin, with axis direction $\hat{a}$.

For a capsule with center $\mathbf{c}$ and half-axis direction $\hat{u}$ (body-frame Y rotated to world), half-length $h$, radius $r$:
- The two endpoints are $\mathbf{p}_0 = \mathbf{c} - h\hat{u}$ and $\mathbf{p}_1 = \mathbf{c} + h\hat{u}$.
- The capsule "axis segment" is the line segment $[\mathbf{p}_0, \mathbf{p}_1]$.

**Contact condition**: the capsule hits the cylinder wall when the *radial* distance from any point on the capsule axis to the cylinder axis exceeds $R - r$.

For each endpoint $\mathbf{p}_i$:
1. Project onto the plane perpendicular to $\hat{a}$: $\mathbf{p}_\perp = \mathbf{p}_i - (\mathbf{p}_i \cdot \hat{a})\hat{a}$.
2. Radial distance: $d_\perp = |\mathbf{p}_\perp|$.
3. Penetration: $\delta = d_\perp + r - R$. If $\delta > 0$, we have contact.
4. Contact normal (pointing inward, toward axis): $\hat{n} = -\mathbf{p}_\perp / d_\perp$.
5. Contact point: $\mathbf{p}_\perp \cdot (R / d_\perp)$ (on the wall surface), projected back along $\hat{a}$.

For accurate handling we should also sample the capsule mid-axis (to catch the case where the rod is bowing outward in the middle — though for rigid rods this is just the geometric midpoint). Two-endpoint + midpoint sampling covers the common cases well.

**Edge case — rod parallel to wall**: When the rod axis lies exactly in the radial plane and is pressed against the wall, we can get a line contact. Sampling 2–3 points along the axis handles this adequately for rigid bodies.

### 3.2 Code location

- **Declaration**: `include/physics/collision.hpp` — add `Contact collideCapsuleCylinder(const RigidBody& C, float cylRadius, const glm::vec3& cylAxis);` (returns up to 3 contact points, or use a vector).
- **Implementation**: `src/physics/collision.cpp`.
- **Integration into contact loop**: In the main simulation step (around where `collideCapsuleFloor` is called), add a check `if (settings.scene.cylinder.enabled)` and loop over all (non-static) bodies.

### 3.3 NSC integration

The contact goes into the same `ContactPrimitive` list used by the NSC solver:
- `body_a` = rod index, `body_b` = -1 (wall sentinel, same as floor).
- `normal` = inward radial direction.
- `distance` = negative of penetration.
- The NSC solver already handles infinite-mass walls (floor) via `body_b == -1`; reuse this.

### 3.4 No axial collision

Per the spec: "no collision in the axial direction." This means:
- We do **not** apply PBC in the cylinder-axis direction.
- We do **not** cap the cylinder at finite length.
- The rod can slide freely along $\hat{a}$.

This is automatically satisfied by the formulation above — the collision only acts in the radial direction.

---

## 4. Initial velocity distribution on $\mathbb{R}^3 \times S^2$

> "What is the most natural probability distribution of velocities living in the tangent space of configuration space ($\mathbb{R}^3 \times S^2$)?"

The configuration space of a rigid rod (no azimuthal spin) is $\mathbb{R}^3 \times S^2$. The tangent space at a point $(x, \hat{u})$ is $\mathbb{R}^3 \times T_{\hat{u}}S^2 \cong \mathbb{R}^3 \times \mathbb{R}^2$. So the velocity DOF is 5: 3 translational + 2 tumbling.

The **Maxwell–Boltzmann** distribution is the natural choice:

- **Translational**: each component $v_i \sim \mathcal{N}(0, k_BT/m)$, giving speed $|v| \sim$ Maxwell distribution.
- **Rotational (tumbling only)**: angular velocity perpendicular to the rod axis $\omega_\perp \sim \mathcal{N}(0, k_BT/I_\perp)$ (2 components). No $\omega_\parallel$ — see §4.1.

For the reptation experiment, a practical approach:
- Draw $v_\parallel$ (velocity component along the tube axis) from $\mathcal{N}(0, \sigma_v)$.
- Draw $\omega_\perp$ (tumbling angular velocity) from $\mathcal{N}(0, \sigma_\omega)$ in the 2D plane perpendicular to the rod.
- Set $v_\perp = 0$ initially (the rod starts centered in the tube).
- Set $\omega_\parallel = 0$ always.

All initial conditions handled externally in the Python driver script (§6).

### 4.1 Axial Spin and Rolling Friction (Two Phases)

**Phase 1: No azimuthal spin ($\omega_\parallel = 0$)**
For initial experiments, we want to completely ignore spinning around the rod's axis. An axisymmetric rod spinning around its own axis has no observable geometric effect in a perfect tube, and in the $S^2$ configuration space the parallel spin is not a degree of freedom.

**Implementation for Phase 1**: Add a per-step projection that zeros out the rod-axis component of $\omega$:
```cpp
// After integration/collision, project out parallel spin
glm::vec3 u = rod_axis_world(body);  // body-frame Y rotated to world
body.w -= glm::dot(body.w, u) * u;
```
This ensures clean 5-DOF dynamics.

**Phase 2: Adding Rolling Friction**
Later, we will want to investigate the effects of rolling friction. When the rod twists and contacts the walls, tangental friction will naturally induce axial spin. 
To support this:
1. **Disable the $\omega$ projection:** Allow the rod to accumulate axial angular velocity.
2. **Apply Rolling Resistance:** Introduce a rolling friction coefficient (e.g., `--rolling-mu`). During contact resolution, calculate the normal impulse $J_n$. The rolling resistance applies an opposing torque proportional to the normal force and the axial spin: $\tau_{roll} = - \mu_r |J_n / \Delta t| \hat{\omega}_\parallel$.
3. **CLI/Configuration:** Add a flag `--enable-spin` and `--rolling-mu <val>` to easily toggle between Phase 1 and Phase 2.

---

## 5. Output & data export

### 5.1 Existing infrastructure we reuse

| Need | Existing tool |
|------|---------------|
| Full trajectory | `--perrod perrod.csv` (pos, vel, ω, quat, KE per frame) |
| Contact events | `--network network.csv` (rod i/j, contact point, normal, force) |
| Summary stats | `--csv profile.csv` (KE, contacts per frame) |

### 5.2 New: reptation summary

At simulation end (when `--stop-ke` triggers or `--steps` reached), print a one-line summary and optionally append to a summary CSV:

```
--reptation-summary <path>
```

Columns:
```
mu, R_cyl, L_rod, d_rod, v0_lin, v0_ang, sliding_length, n_contacts, total_time, final_KE
```

Where:
- `sliding_length` = total arc-length traveled by the rod COM along the tube axis $\hat{a}$: $\sum_i |(\mathbf{x}_{i+1} - \mathbf{x}_i) \cdot \hat{a}|$.
  - Alternatively, just the net displacement: $|(\mathbf{x}_\text{final} - \mathbf{x}_0) \cdot \hat{a}|$.
- `n_contacts` = cumulative number of contact events with the wall.
- `total_time` = simulation time at stop.

This makes parameter sweeps trivial: a bash loop over `mu` values appends to the same CSV.

### 5.3 Sliding length computation

Two definitions worth tracking:
1. **Net displacement** $\Delta s = |(\mathbf{x}_f - \mathbf{x}_0) \cdot \hat{a}|$ — how far it got.
2. **Total path length** $L_{\text{path}} = \sum_i |(\mathbf{x}_{i+1} - \mathbf{x}_i) \cdot \hat{a}|$ — accounts for back-and-forth.

Both are cheap to compute. Store a running accumulator in the simulation loop.

---

## 6. Python driver script

A lightweight Python script `scripts/sweep_reptation.py` to:

1. Generate initial velocities from the MB distribution (§4).
2. Launch `rigidbody_viewer` with appropriate flags for each `(mu, R, v0)` combination.
3. Collect the reptation summary CSVs.
4. Produce plots (sliding length vs. mu, etc.).

Sketch:
```python
import subprocess, itertools, numpy as np

mus = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
radii = [0.2, 0.3, 0.5]
n_trials = 20
sigma_v = 1.0
sigma_w = 0.5

for mu, R in itertools.product(mus, radii):
    for trial in range(n_trials):
        rng = np.random.default_rng(seed=trial)
        v0 = rng.normal(0, sigma_v)          # axial velocity
        w0 = rng.normal(0, sigma_w, size=2)  # tumbling
        
        cmd = [
            "./build-headless/rigidbody_viewer",
            "--headless", "--steps", "2000000",
            "--scene", "assets/scenes/reptation.json",
            "--nsc", "--nsc-mu", str(mu),
            "--set-velocity", "0", "0", str(v0), "0",
            "--set-ang-velocity", "0", str(w0[0]), "0", str(w0[1]),
            "--stop-ke", "1e-10",
            "--reptation-summary", f"results/rept_mu{mu}_R{R}_t{trial}.csv",
            "--perrod", f"results/perrod_mu{mu}_R{R}_t{trial}.csv",
        ]
        subprocess.run(cmd)
```

---

## 7. Implementation order

| Step | Task | Files touched | Effort |
|------|------|---------------|--------|
| 1 | Add `CylinderCfg` to config + JSON parsing | `include/config/config.hpp`, `src/config/config.cpp` | S |
| 2 | Implement `collideCapsuleCylinder()` | `include/physics/collision.hpp`, `src/physics/collision.cpp` | M |
| 3 | Integrate cylinder contacts into simulation loop | `src/app/main.cpp` (contact detection section) | M |
| 4 | Add `--set-ang-velocity` CLI flag | `src/app/main.cpp` | S |
| 5 | Add `--stop-ke` stopping condition | `src/app/main.cpp` | S |
| 6 | Add reptation summary output | `src/app/main.cpp` | S |
| 7 | Create `reptation.json` scene file | `assets/scenes/reptation.json` | S |
| 8 | Write Python sweep driver | `scripts/sweep_reptation.py` | S |
| 9 | Test single-rod-in-tube by hand | — | S |
| 10 | Run sweep & make plots | `scripts/plot_reptation.py` | M |

Steps 1–3 are the critical path. Steps 4–6 are incremental CLI additions. Steps 7–10 are usage/analysis.

---

## 8. Resolved decisions (from Addition 1)

These were open questions, now settled:

1. **Multi-point sampling**: 2 endpoints + midpoint. ✅ Agreed.
2. **Gravity**: Off by default (`"gravity": [0,0,0]` in the reptation scene JSON). Adjustable via `--gravity` CLI. ✅
3. **Damping**: Zero (`lin_damp = 0`, `ang_damp = 0`) in the reptation scene. Energy dissipation comes only from friction.
4. **Restitution**: $\varepsilon = 1$ (perfectly elastic) as default for cylinder wall contacts. Adjustable via a new JSON/CLI field `"restitution"` in the NSC config. ✅
5. **Wall friction**: No separate wall friction parameter. The cylinder wall uses the rod's $\mu$ (from `--nsc-mu`). Removes a degree of freedom. ✅
6. **Cylinder center**: At origin. Rod also starts at origin. ✅
7. **$\omega_\parallel$**: Projected out every step (see §4.1). The engine currently allows it, but for reptation we enforce 5-DOF by zeroing the rod-axis spin component. ✅

## 9. Resolved (from Addition 2)

8. **Restitution location**: `nsc` block → `"restitution": 1.0`. It's a contact model parameter, not geometry. ✅
9. **Stopping condition**: Use a 5-frame rolling average of KE. Stop when the average drops below threshold. Cheap (just a small ring buffer), avoids false stops mid-bounce. ✅
10. **Sliding length**: Export **both** net displacement and total path length. Both are cheap running accumulators. ✅

---

All design questions are now resolved. Ready to implement (see §7 for the task list).
