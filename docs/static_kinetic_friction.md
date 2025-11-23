# Static and Kinetic Friction Implementation

## Summary

Successfully implemented **velocity-dependent friction** for the Hertz-Mindlin contact model, distinguishing between static (μ_s) and kinetic (μ_k) friction coefficients. This enhancement enables realistic sphere packing simulations with proper stiction behavior.

## Implementation Details

### 1. Configuration Parameters (config.hpp)

Added three new parameters to `HertzMindlinCfg`:

```cpp
double friction_coeff = 0.3;              // Kinetic friction μ_k
double friction_static_coeff = 0.5;       // Static friction μ_s [typically > μ_k]
double friction_transition_vel = 1e-3;    // Velocity threshold v_c (m/s)
```

### 2. Friction Model (hertz_mindlin.cpp)

Implemented smooth velocity-dependent transition:

```cpp
μ(v) = μ_k + (μ_s - μ_k) · exp(-|v_t|/v_c)
```

Where:
- **μ_k** = kinetic (dynamic) friction coefficient
- **μ_s** = static friction coefficient  
- **v_c** = transition velocity threshold
- **v_t** = tangential velocity magnitude

**Behavior:**
- At **v_t ≈ 0** (near-static): μ(v) ≈ μ_s (high friction, stiction)
- At **v_t >> v_c** (sliding): μ(v) ≈ μ_k (lower kinetic friction)
- Smooth exponential transition prevents discontinuities

### 3. Coulomb Limit

The effective friction coefficient is used in the Coulomb limit:

```cpp
|F_t| ≤ μ(v) · |F_n|
```

When the elastic tangential force exceeds this limit, the contact transitions from sticking to sliding.

### 4. JSON Configuration

Example scene configuration:

```json
{
  "physics": {
    "hertz_mindlin": {
      "enabled": true,
      "friction_coeff": 0.3,           // μ_k
      "friction_static_coeff": 0.5,    // μ_s
      "friction_transition_vel": 0.001 // v_c (m/s)
    }
  },
  "scene": {
    "randomForce": {
      "enabled": true,
      "fSigma": 1e-6  // Gaussian thermal noise
    }
  }
}
```

## Answers to Your Questions

### 1. ✅ Gaussian Noise on Centroids

**YES** - Already implemented via `RandomForceCfg`:

```json
"randomForce": {
  "enabled": true,
  "fSigma": 1e-6,    // Translational noise (N)
  "tauMag": 0.0,     // Optional rotational noise
  "seed": 42
}
```

**Formula:** `f += √(dt) · σ_f · N(0,1)³`

This applies white Gaussian noise to simulate thermal fluctuations, enabling equilibration in dense packings.

### 2. ✅ Dry Friction to Dissipate Energy

**YES** - Multiple dissipation mechanisms:

1. **Tangential damping:** C_t · v_t in Mindlin force
2. **Normal damping:** Computed from restitution coefficient
3. **Rolling friction:** τ_r = -μ_r · R · |F_n| · ω̂
4. **Velocity-dependent friction:** Dissipates energy during sliding

**Tuning dissipation:**
- Lower `restitution_coeff` (e.g., 0.3) → more inelastic collisions
- Higher `friction_coeff` → more tangential dissipation
- Increase `rolling_friction_coeff` → resist rotations

### 3. ⚠️ Static Friction Implementation

**Previously:** Single coefficient μ with implicit stiction via tangential spring

**Now:** Separate μ_s and μ_k with smooth velocity-dependent transition

**How it works:**
- **Sticking regime** (v_t → 0): Spring accumulates displacement, μ ≈ μ_s
- **Transition** (v_t ≈ v_c): Exponential blend between μ_s and μ_k
- **Sliding regime** (v_t >> v_c): Spring saturates, μ ≈ μ_k

This is **more physically realistic** than a sharp Coulomb cutoff and prevents numerical issues from friction discontinuities.

## Test Scenes

### Small Scale Test
`friction_test_small.json` - 5 spheres falling onto ground:
- Demonstrates static/kinetic transition
- Shows thermal noise effects
- Low energy dissipation test

### Realistic Packing
`realistic_packing.json` - 50 spheres with thermal agitation:
- Dense packing simulation
- Equilibration with noise + friction
- Suitable for jamming studies (Makse et al. methodology)

## Running Tests

```bash
cd build-debug

# Small test (5 spheres)
./rigidbody_viewer_3d --scene ../assets/scenes/friction_test_small.json

# Realistic packing (50 spheres)  
./rigidbody_viewer_3d --scene ../assets/scenes/realistic_packing.json

# Headless mode
./rigidbody_viewer_3d --headless --steps 5000 --scene ../assets/scenes/friction_test_small.json
```

## Key Parameters for Granular Packings

Based on literature (Makse, O'Hern, Silbert et al.):

### Material Properties
- **Young's modulus:** E = 7×10¹⁰ Pa (glass)
- **Poisson ratio:** ν = 0.25
- **Restitution:** e = 0.3–0.5 (inelastic for energy dissipation)

### Friction Coefficients
- **Static friction:** μ_s = 0.4–0.6 (prevents slip at rest)
- **Kinetic friction:** μ_k = 0.2–0.4 (lower during sliding)
- **Rolling friction:** μ_r = 0.01–0.02 (resist rolling)
- **Transition velocity:** v_c = 10⁻³ m/s

### Thermal Noise
- **Force noise:** σ_f = 10⁻⁷–10⁻⁵ N (tune for equilibration time)
- **Purpose:** Explore configuration space, escape local minima

### Simulation Parameters
- **Timestep:** dt = 10⁻⁴ s (captures contact dynamics)
- **Duration:** 5–10 s (allow settling)
- **Boundaries:** Confining walls or periodic

## Physical Interpretation

The velocity-dependent friction model captures:

1. **Stiction at rest:** High μ_s prevents motion under small perturbations
2. **Breakaway force:** Requires finite force to overcome static friction
3. **Kinetic regime:** Lower μ_k during continuous sliding
4. **Smooth transition:** No force discontinuities at v_t = 0

This is more realistic than:
- Sharp Coulomb cutoff (discontinuous)
- Single friction coefficient (no stiction distinction)
- Stribeck model (more complex, overkill for granular DEM)

## Future Enhancements

Potential improvements (not currently needed):
- ✅ Velocity-dependent μ(v) - **DONE**
- ⬜ Stribeck friction: μ(v) = μ_k + (μ_s - μ_k)·exp(-(v/v_s)^α) + μ_v·v
- ⬜ Temperature-dependent properties for thermal simulations
- ⬜ Adhesion forces (JKR, DMT models) for fine particles

## References

1. **Hertz-Mindlin DEM:** Cundall & Strack (1979), Silbert et al. (2001)
2. **Granular packings:** O'Hern et al. (2003), Makse et al. (1999)
3. **Friction models:** Coulomb (1785), Stribeck (1902), Rabinowicz (1995)
4. **Velocity Verlet:** Swope et al. (1982), Allen & Tildesley (1987)

## Commit Message

```
feat: Add static/kinetic friction distinction to Hertz-Mindlin

Implement velocity-dependent friction coefficient μ(v) = μ_k + (μ_s - μ_k)·exp(-|v|/v_c)
to distinguish between static (stiction) and kinetic (sliding) friction regimes.

Changes:
- Add friction_static_coeff and friction_transition_vel to HertzMindlinCfg
- Modify computeMindlinForce() to use smooth velocity-dependent μ(v)
- Update JSON parsing for new parameters with backward compatibility
- Create test scenes: friction_test_small.json, realistic_packing.json

Benefits:
- Realistic stiction behavior for sphere packings
- Smooth force transition (no discontinuities)
- Enables granular jamming studies with thermal noise

Fixes integration bug: integrateHalfVel → integrateSecondHalf
```
