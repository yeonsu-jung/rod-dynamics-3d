# Soft Contact Model Design (from DisMech)

## Overview
This document outlines how to implement a penalty-based soft contact model in rod-dynamics-3d, learning from the [DisMech](https://github.com/StructuresComp/dismech-rods) discrete differential geometry simulator.

**Key insight**: Instead of hard impulse-based contacts, use **penalty forces derived from smooth potential energy functions** that:
- Allow small penetrations (tunable via stiffness)
- Provide continuous forces and torques
- Preserve energy better (implicit integration + Jacobians)
- Enable implicit time stepping with Newton-Raphson

---

## Current vs. Proposed Approaches

### Current (rod-dynamics-3d): Hard Impulse-Based Contact
```cpp
// Sequential impulse solver (Gauss-Seidel)
for each contact:
    compute impulse to resolve penetration
    apply impulse to velocities
    iterate until convergence
```

**Issues observed**:
- 81-85% energy loss in confined systems despite perfect settings
- Sequential impulse solver has inherent numerical damping
- Cannot perfectly conserve energy with many simultaneous contacts
- Prioritizes constraint satisfaction over energy conservation

### Proposed (DisMech): Soft Penalty-Based Contact
```cpp
// Penalty method with potential energy
for each contact:
    compute signed distance d
    compute penalty force: F = -k * ∇U(d)  (gradient of potential)
    compute Jacobian: J = -k * ∇²U(d)     (Hessian for implicit)
    add to force vector and Jacobian matrix
```

**Advantages**:
- Smooth, continuous forces (no discrete impulses)
- Can use implicit integration (backward Euler, implicit midpoint)
- Newton-Raphson with exact Jacobians → quadratic convergence
- Energy better conserved (small controlled penetrations vs. iterative corrections)
- Friction naturally integrated via velocity-dependent forces

---

## DisMech Contact Model Details

### 1. Three Contact Primitives

DisMech handles rod-rod contact by detecting three geometric configurations:

#### a) Point-to-Point (P2P)
```
Rod 1: x1s ----------- x1e
                ⋮
Rod 2:         x2s
```
**Distance**: `d = ||x2s - x1s||`

**Potential (non-penetrated)**: 
$$U_{p2p} = \left(\frac{1}{K_1} \log\left(1 + e^{K_1(h - d)}\right)\right)^2$$

**Potential (penetrated)**: 
$$U_{p2p}^{pen} = (h - d)^2$$

Where:
- `h = r1 + r2` (sum of radii = surface limit)
- `δ` (delta) = transition width
- `K1 = 15/δ` = stiffness parameter

**Gradient**: 6D vector `∇U = [∂U/∂x1s, ∂U/∂x2s]` (3+3 components)
**Hessian**: 6×6 matrix for implicit integration

#### b) Edge-to-Point (E2P)
```
Rod 1: x1s -----●----- x1e  (edge)
                ⋮
Rod 2:         x2s          (point)
```
**Distance**: Perpendicular distance from point `x2s` to line segment `x1s--x1e`
$$d = \frac{||(x1s - x2s) \times (x1e - x2s)||}{||x1e - x1s||}$$

**Gradient**: 9D vector `[∂U/∂x1s, ∂U/∂x1e, ∂U/∂x2s]`
**Hessian**: 9×9 matrix

#### c) Edge-to-Edge (E2E)
```
Rod 1: x1s ----------- x1e
         ⋮           
Rod 2:    x2s ----------- x2e
```
**Distance**: Minimum distance between two line segments (skew lines)
$$d = \left|\frac{(x1s - x2s) \cdot (e_1 \times e_2)}{||e_1 \times e_2||}\right|$$

Where `e1 = x1e - x1s`, `e2 = x2e - x2s`

**Gradient**: 12D vector `[∂U/∂x1s, ∂U/∂x1e, ∂U/∂x2s, ∂U/∂x2e]`
**Hessian**: 12×12 matrix

### 2. Piecewise Potential Functions

The potential has two regimes to handle both approaching and penetrating contacts:

```
       U
       │     
       │   ┌─────  (h - d)²  [quadratic penalty]
       │  ╱
       │ ╱   log-barrier smoothing
h - δ ─┼──────────────────────────── d (distance)
       │              h + δ
```

- **Non-penetrated** (`d > h - δ`): Smooth log-barrier
  - `U = (1/K1 * log(1 + exp(K1*(h - d))))²`
  - Provides smooth repulsive force as surfaces approach
  - Vanishes exponentially when `d > h + δ`

- **Penetrated** (`d ≤ h - δ`): Quadratic penalty
  - `U = (h - d)²`
  - Strong repulsion proportional to penetration depth
  - Linear force: `F ∝ (h - d)`

**Parameters**:
- `δ` (delta): Transition width, typically `0.01 * rod_length`
- `K1 = 15/δ`: Controls smoothness of transition
- `k_scaler`: Global stiffness multiplier (tunable)

### 3. Friction Model

DisMech implements a **smooth friction model** that transitions between sticking and sliding:

```cpp
// Compute relative tangential velocity
Vector3d v_rel = v1 - v2;  // relative velocity at contact point
Vector3d n = contact_force.normalized();  // contact normal
Vector3d v_tan = v_rel - (v_rel · n) * n;  // tangential component
double v_tan_mag = ||v_tan||;

// Smooth friction coefficient using tanh-like function
if (v_tan_mag == 0) {
    friction = 0;
} else if (v_tan_mag > nu) {
    // Sliding: γ = 1
    friction = μ * v_tan / v_tan_mag;  
} else {
    // Sticking: γ smoothly transitions 0→1
    γ = (2 / (1 + exp(-K2 * v_tan_mag))) - 1;
    friction = μ * γ * v_tan / v_tan_mag;
}
```

Where:
- `μ` = friction coefficient (Coulomb)
- `ν` (nu) = velocity threshold for sliding (e.g., 1e-3 m/s)
- `K2 = 15/ν` = smoothness parameter
- `γ` = smooth sticking-to-sliding transition function

**Key features**:
- No discontinuous jumps (unlike classic Coulomb friction)
- Friction force depends on normal force magnitude
- Friction Jacobian computed via symbolic differentiation for implicit stepping

### 4. Symbolic Differentiation (SymEngine)

DisMech uses **SymEngine** (symbolic math library) to:

1. **Define potential energy symbolically**:
   ```cpp
   RCP<const Basic> E_p2p = pow(mul(div(one, K1), 
                                 log(add(one, exp(mul(K1, sub(h2, dist_p2p)))))), 2);
   ```

2. **Compute gradient automatically**:
   ```cpp
   jacobian(E_p2p_potential, nodes_p2p, E_p2p_gradient);
   // ∇U = [∂U/∂x1s_x, ∂U/∂x1s_y, ∂U/∂x1s_z, ∂U/∂x2s_x, ∂U/∂x2s_y, ∂U/∂x2s_z]
   ```

3. **Compute Hessian automatically**:
   ```cpp
   jacobian(E_p2p_gradient, nodes_p2p, E_p2p_hessian);
   // ∇²U = 6×6 matrix of second derivatives
   ```

4. **Compile to LLVM functions**:
   ```cpp
   E_p2p_gradient_func.init(func_p2p_inputs, E_p2p_gradient.as_vec_basic(), 
                            symbolic_cse, opt_level);
   ```
   - `symbolic_cse`: Common subexpression elimination
   - `opt_level = 3`: Maximum optimization
   - Result: Fast JIT-compiled evaluation functions

**Runtime usage**:
```cpp
// Evaluate gradient at specific positions
p2p_input << x1s_x, x1s_y, x1s_z, x2s_x, x2s_y, x2s_z, K1, h2;
E_p2p_gradient_func.call(p2p_gradient.data(), p2p_input.data());
// p2p_gradient now contains 6-element force vector
```

---

## Implementation Strategy for rod-dynamics-3d

### Phase 1: Manual Gradient Implementation (Simpler Start)

Before diving into symbolic differentiation, implement gradients by hand:

```cpp
// include/physics/soft_contact.hpp
struct SoftContactConfig {
    double delta = 0.005;           // Transition width
    double k_scaler = 1.0;          // Stiffness multiplier
    double mu = 0.5;                // Friction coefficient
    double nu = 1e-3;               // Sticking velocity threshold
};

class SoftContactSolver {
public:
    void detectContacts(std::vector<RigidBody>& bodies);
    void computeForces(std::vector<RigidBody>& bodies, double dt);
    
private:
    double K1, K2;
    
    // Point-to-point contact
    void computeP2PForce(const glm::vec3& x1s, const glm::vec3& x2s,
                         double h, glm::vec3& f1s, glm::vec3& f2s);
    
    // Edge-to-point contact  
    void computeE2PForce(const glm::vec3& x1s, const glm::vec3& x1e,
                         const glm::vec3& x2s, double h,
                         glm::vec3& f1s, glm::vec3& f1e, glm::vec3& f2s);
    
    // Edge-to-edge contact
    void computeE2EForce(const glm::vec3& x1s, const glm::vec3& x1e,
                         const glm::vec3& x2s, const glm::vec3& x2e, double h,
                         glm::vec3& f1s, glm::vec3& f1e,
                         glm::vec3& f2s, glm::vec3& f2e);
};
```

**Point-to-Point gradient (hand-derived)**:
```cpp
void SoftContactSolver::computeP2PForce(const glm::vec3& x1s, const glm::vec3& x2s,
                                         double h, glm::vec3& f1s, glm::vec3& f2s) {
    glm::vec3 diff = x2s - x1s;
    double d = glm::length(diff);
    glm::vec3 n = diff / d;  // unit normal from x1s to x2s
    
    double potential_deriv;
    if (d > h - delta) {
        // Non-penetrated: dU/dd from log-barrier
        double arg = K1 * (h - d);
        double exp_arg = std::exp(arg);
        double log_term = std::log(1.0 + exp_arg);
        potential_deriv = -2.0 / (K1 * K1) * log_term * exp_arg / (1.0 + exp_arg);
    } else {
        // Penetrated: dU/dd from quadratic
        potential_deriv = -2.0 * (h - d);
    }
    
    // Force = -k * dU/dx = -k * (dU/dd) * (dd/dx)
    // dd/dx = n for x2s, -n for x1s
    double force_mag = -k_scaler * potential_deriv;
    f1s = -force_mag * n;  // Push x1s away from x2s
    f2s = force_mag * n;   // Push x2s away from x1s
}
```

### Phase 2: Integration with Existing Simulator

Modify the main simulation loop:

```cpp
// src/app/main.cpp - in simulation loop

// OPTION A: Replace hard contacts with soft contacts
SoftContactSolver softContact(config);
for (int step = 0; step < totalSteps; ++step) {
    // 1. Apply external forces (gravity, random forces)
    applyExternalForces();
    
    // 2. Integrate positions (predictor)
    for (auto& rb : rods) {
        integrate(rb, gravity, dt);
    }
    
    // 3. Compute soft contact forces
    softContact.detectContacts(rods);
    softContact.computeForces(rods, dt);
    
    // 4. Update velocities with contact forces (corrector)
    for (auto& rb : rods) {
        rb.v += (rb.f / rb.mass) * dt;
        rb.w += rb.I_body_inv * rb.tau * dt;
    }
}

// OPTION B: Hybrid - soft for normal forces, impulse for friction
// (keep your existing solver, add soft normal forces)
```

### Phase 3: Collision Detection

DisMech uses FCL (Flexible Collision Library) for broad phase. For rod-dynamics-3d:

```cpp
// Adapt your existing capsule collision detection
struct ContactPrimitive {
    enum Type { POINT_TO_POINT, EDGE_TO_POINT, EDGE_TO_EDGE };
    Type type;
    int body_a, body_b;      // Rod indices
    int vertex_a, vertex_b;  // Vertex/segment indices
    double distance;
};

std::vector<ContactPrimitive> detectCapsuleCapsuleContact(
    const glm::vec3& a1, const glm::vec3& a2, double r1,  // Capsule A
    const glm::vec3& b1, const glm::vec3& b2, double r2)  // Capsule B
{
    // Compute closest points on two line segments
    // Classify as P2P, E2P, or E2E based on parameters
    // ...
}
```

### Phase 4: Friction Integration

```cpp
void SoftContactSolver::computeFriction(RigidBody& body_a, RigidBody& body_b,
                                        const ContactPrimitive& contact,
                                        const glm::vec3& normal_force, double dt) {
    // Get contact point positions and velocities
    glm::vec3 p_a = getContactPoint(body_a, contact);
    glm::vec3 p_b = getContactPoint(body_b, contact);
    
    glm::vec3 v_a = body_a.v + glm::cross(body_a.w, p_a - body_a.x);
    glm::vec3 v_b = body_b.v + glm::cross(body_b.w, p_b - body_b.x);
    
    glm::vec3 v_rel = v_a - v_b;
    glm::vec3 n = glm::normalize(normal_force);
    glm::vec3 v_tan = v_rel - glm::dot(v_rel, n) * n;
    
    double v_tan_mag = glm::length(v_tan);
    if (v_tan_mag < 1e-10) return;
    
    // Smooth friction model
    double gamma;
    if (v_tan_mag > nu) {
        gamma = 1.0;  // Sliding
    } else {
        gamma = 2.0 / (1.0 + std::exp(-K2 * v_tan_mag)) - 1.0;  // Sticking
    }
    
    double fn_mag = glm::length(normal_force);
    glm::vec3 friction_force = -mu * gamma * fn_mag * (v_tan / v_tan_mag);
    
    // Apply to bodies
    applyForceAtPoint(body_a, friction_force, p_a);
    applyForceAtPoint(body_b, -friction_force, p_b);
}
```

---

## Parameter Tuning Guidelines

Based on DisMech's approach:

### Contact Stiffness (`k_scaler`)
- **Too low**: Large penetrations, unrealistic deformations
- **Too high**: Stiff system, requires smaller timesteps, numerical instability
- **Recommended**: Start with `k_scaler = E * A / L` where:
  - `E` = Young's modulus of material
  - `A` = cross-sectional area (`π * r²`)
  - `L` = characteristic length (rod length)
- **For rod-dynamics-3d**: Try `k_scaler = 100` to `10000` with `dt = 1/600`

### Transition Width (`δ`)
- Controls smoothness of contact force
- **Recommended**: `δ = 0.01 * rod_length` to `0.05 * rod_length`
- Smaller `δ` → sharper transition, may need smaller timestep
- Larger `δ` → smoother, more forgiving, but activates farther from contact

### Friction Parameters
- `μ` (friction coefficient): `0.1` (slippery) to `0.9` (sticky)
- `ν` (sticking threshold): `1e-4` to `1e-2` m/s
- `K2 = 15/ν`: Automatically computed

### Time Stepping
- **Explicit (current)**: `dt ≤ sqrt(m / k)` (stable timestep)
  - For `m = 1 kg`, `k = 1000 N/m`: `dt ≤ 0.03 s`
  - Your current `dt = 1/600 ≈ 0.0017 s` should be safe
  
- **Implicit (DisMech-style)**: Can use much larger timesteps
  - Requires Newton-Raphson solver with Jacobian matrices
  - More complex but allows `dt = 0.01` to `0.1 s`

---

## Expected Improvements

Switching from hard impulse to soft penalty contacts should:

1. **Better energy conservation**: ~5-10% loss instead of 80%
   - Small controlled penetrations instead of iterative corrections
   - Smooth forces instead of discrete impulses
   
2. **More physical behavior**:
   - Rods can deform slightly at contacts (realistic)
   - Continuous force application (no velocity jumps)
   
3. **Tunable stiffness**:
   - Can model different materials
   - Trade-off between realism and stability

4. **Smoother dynamics**:
   - No chattering from impulse iterations
   - Better for visualization and analysis

**Caveat**: Will see small penetrations (controlled by `k_scaler` and `dt`). This is a feature, not a bug!

---

## References

1. **DisMech Paper**: 
   - Choi et al., "DisMech: A Discrete Differential Geometry-Based Physical Simulator for Soft Robots and Structures", *IEEE Robotics and Automation Letters*, 2024
   - DOI: 10.1109/LRA.2024.3365292

2. **Discrete Elastic Rods**:
   - Bergou et al., "Discrete Elastic Rods", *ACM SIGGRAPH*, 2008
   - Foundation for DisMech's rod model

3. **Contact Mechanics**:
   - IPC (Incremental Potential Contact): Li et al., *ACM TOG*, 2020
   - Continuous collision detection with barrier functions

4. **Symbolic Differentiation**:
   - SymEngine: https://github.com/symengine/symengine
   - Automatic differentiation for exact gradients/Hessians

---

## Next Steps

1. **Implement P2P contact manually** (simplest case)
   - Test on 2-rod collision
   - Verify energy conservation improves
   
2. **Add E2E contact** (most relevant for parallel rods)
   - Compare with current capsule-capsule solver
   
3. **Integrate friction model**
   - Start without friction, add later
   
4. **Benchmark energy conservation**
   - Re-run confined_n10_box0.70 test
   - Target: <10% energy loss over 1000 frames

5. **(Optional) Add symbolic differentiation**
   - For implicit time stepping
   - Requires SymEngine dependency
