# Sphere Shape Implementation Design

## Goal
Add support for spherical particles to enable jammed packing studies (e.g., Makse et al. 1999, 2004).

## Current Architecture

### Shape System
- **Location**: `include/physics/shape.hpp`
- **Current shapes**:
  ```cpp
  enum class ShapeType { Box, Capsule };
  struct Box { float hx, hy, hz; };
  struct Capsule { float r, h; };
  ```

### RigidBody
- **Location**: `include/physics/rigid_body.hpp`
- **Current structure**:
  ```cpp
  struct RigidBody {
      ShapeType type;
      Box box{};
      Capsule cap{};
      // ... physics state (x, q, v, w, mass, I_body, friction, restitution)
      
      // Factory methods:
      static RigidBody makeCapsule(...);
      static RigidBody makeRodLD(...);
      static RigidBody makeStaticFloor(...);
  };
  ```

### Contact Detection
- **Capsule-Capsule**: Implemented via segment-segment distance
- **Location**: Likely in collision detection files

## Proposed Changes

### 1. Shape Definition (`include/physics/shape.hpp`)
```cpp
enum class ShapeType { Box, Capsule, Sphere };  // Add Sphere

struct Sphere {
    float r;  // radius
};
```
**Estimated lines**: +5

### 2. RigidBody Structure (`include/physics/rigid_body.hpp`)
```cpp
struct RigidBody {
    ShapeType type;
    Box box{};
    Capsule cap{};
    Sphere sphere{};  // Add sphere union member
    
    // Add factory method:
    static RigidBody makeSphere(float radius, float density, 
                                 const glm::vec3& position, 
                                 float friction = 0.2f, 
                                 float restitution = 0.5f);
};

// Implementation in src/physics/rigid_body.cpp or inline:
inline RigidBody RigidBody::makeSphere(float radius, float density, 
                                        const glm::vec3& position, 
                                        float friction, float restitution) {
    RigidBody rb;
    rb.type = ShapeType::Sphere;
    rb.sphere.r = radius;
    
    // Mass: m = (4/3) π r³ ρ
    float volume = (4.0f / 3.0f) * M_PI * radius * radius * radius;
    rb.mass = density * volume;
    rb.invMass = 1.0f / rb.mass;
    
    // Inertia: I = (2/5) m r² (diagonal for sphere)
    float I_diag = 0.4f * rb.mass * radius * radius;
    rb.I_body = glm::mat3(I_diag, 0, 0,
                          0, I_diag, 0,
                          0, 0, I_diag);
    rb.I_body_inv = glm::inverse(rb.I_body);
    
    rb.x = position;
    rb.q = glm::quat(1, 0, 0, 0);  // Identity rotation
    rb.v = glm::vec3(0);
    rb.w = glm::vec3(0);
    
    rb.friction = friction;
    rb.restitution = restitution;
    
    return rb;
}
```
**Estimated lines**: +40

### 3. Collision Detection

#### Broadphase (if separate file exists)
Add sphere bounding box computation:
```cpp
glm::vec3 getAABBMin(const RigidBody& rb) {
    switch (rb.type) {
        case ShapeType::Sphere:
            return rb.x - glm::vec3(rb.sphere.r);
        // ... existing cases
    }
}

glm::vec3 getAABBMax(const RigidBody& rb) {
    switch (rb.type) {
        case ShapeType::Sphere:
            return rb.x + glm::vec3(rb.sphere.r);
        // ... existing cases
    }
}
```
**Estimated lines**: +10

#### Narrowphase Contact Detection
Location: Likely in `src/physics/collision.cpp` or similar

**Sphere-Sphere** (simplest case):
```cpp
bool detectSphereSphere(const RigidBody& a, const RigidBody& b, 
                         ContactManifold& contact) {
    glm::vec3 delta = b.x - a.x;
    float dist2 = glm::dot(delta, delta);
    float r_sum = a.sphere.r + b.sphere.r;
    
    if (dist2 >= r_sum * r_sum) return false;  // No contact
    
    float dist = std::sqrt(dist2);
    glm::vec3 normal = delta / dist;  // From a to b
    float overlap = r_sum - dist;
    
    contact.normal = normal;
    contact.depth = overlap;
    contact.pointA = a.x + normal * a.sphere.r;
    contact.pointB = b.x - normal * b.sphere.r;
    
    return true;
}
```
**Estimated lines**: +20

**Sphere-Capsule** (more complex, but needed for mixed systems):
```cpp
bool detectSphereCapsule(const RigidBody& sphere, const RigidBody& capsule,
                          ContactManifold& contact) {
    // Get capsule axis endpoints
    glm::vec3 cap_axis = glm::rotate(capsule.q, glm::vec3(0, 1, 0));
    glm::vec3 p0 = capsule.x - cap_axis * capsule.cap.h;
    glm::vec3 p1 = capsule.x + cap_axis * capsule.cap.h;
    
    // Find closest point on capsule axis to sphere center
    glm::vec3 v = p1 - p0;
    float t = glm::dot(sphere.x - p0, v) / glm::dot(v, v);
    t = glm::clamp(t, 0.0f, 1.0f);
    glm::vec3 closest = p0 + t * v;
    
    // Check distance
    glm::vec3 delta = sphere.x - closest;
    float dist2 = glm::dot(delta, delta);
    float r_sum = sphere.sphere.r + capsule.cap.r;
    
    if (dist2 >= r_sum * r_sum) return false;
    
    float dist = std::sqrt(dist2);
    glm::vec3 normal = delta / dist;
    
    contact.normal = normal;
    contact.depth = r_sum - dist;
    contact.pointA = sphere.x - normal * sphere.sphere.r;
    contact.pointB = closest + normal * capsule.cap.r;
    
    return true;
}
```
**Estimated lines**: +30

**Dispatch in collision detector**:
```cpp
// In main collision detection function:
if (a.type == ShapeType::Sphere && b.type == ShapeType::Sphere) {
    return detectSphereSphere(a, b, contact);
} else if (a.type == ShapeType::Sphere && b.type == ShapeType::Capsule) {
    return detectSphereCapsule(a, b, contact);
} else if (a.type == ShapeType::Capsule && b.type == ShapeType::Sphere) {
    bool result = detectSphereCapsule(b, a, contact);
    if (result) contact.normal = -contact.normal;  // Flip normal
    return result;
}
// ... existing capsule-capsule, etc.
```
**Estimated lines**: +15

### 4. Contact Model Abstraction

**Issue**: Papers use Hertz-Mindlin forces, current code may use different model.

**Proposed**: Create contact model interface

Create `include/physics/contact_model.hpp`:
```cpp
#pragma once
#include <glm/glm.hpp>

struct ContactPoint {
    glm::vec3 pointA;      // On body A
    glm::vec3 pointB;      // On body B
    glm::vec3 normal;      // From A to B
    float depth;           // Penetration depth
    glm::vec3 relVel;      // Relative velocity at contact
};

class ContactModel {
public:
    virtual ~ContactModel() = default;
    
    // Compute contact force given contact geometry and material properties
    virtual glm::vec3 computeNormalForce(const ContactPoint& cp,
                                          float stiffness,
                                          float damping) const = 0;
    
    virtual glm::vec3 computeTangentForce(const ContactPoint& cp,
                                           float friction,
                                           float normalForce) const = 0;
};

// Current soft contact model (penalty-based)
class SoftContactModel : public ContactModel {
public:
    glm::vec3 computeNormalForce(const ContactPoint& cp,
                                  float kn, float gamma_n) const override {
        // F_n = k_n * δ + γ_n * v_n
        float vn = glm::dot(cp.relVel, cp.normal);
        return (kn * cp.depth - gamma_n * vn) * cp.normal;
    }
    
    glm::vec3 computeTangentForce(const ContactPoint& cp,
                                    float mu, float Fn_mag) const override {
        // Coulomb friction with tangent velocity
        glm::vec3 vt = cp.relVel - glm::dot(cp.relVel, cp.normal) * cp.normal;
        float vt_mag = glm::length(vt);
        if (vt_mag < 1e-9f) return glm::vec3(0);
        return -(mu * Fn_mag / vt_mag) * vt;  // Opposes tangent motion
    }
};
```
**Estimated lines**: +60

Create `include/physics/hertz_mindlin.hpp`:
```cpp
#pragma once
#include "contact_model.hpp"

// Hertz-Mindlin contact forces for deformable spheres
// References: Makse et al. 1999, Science; Johnson 1985
class HertzMindlinModel : public ContactModel {
public:
    HertzMindlinModel(float E, float nu, float radius)
        : E_(E), nu_(nu), R_(radius) {
        // Effective modulus: E* = E / (2(1 - ν²))
        E_star_ = E / (2.0f * (1.0f - nu * nu));
    }
    
    glm::vec3 computeNormalForce(const ContactPoint& cp,
                                  float kn, float gamma_n) const override {
        // Hertz normal force: F_n = (4/3) E* √R δ^(3/2)
        // kn parameter is ignored (Hertz uses E* and R)
        float delta = cp.depth;
        if (delta <= 0) return glm::vec3(0);
        
        float Fn_mag = (4.0f / 3.0f) * E_star_ * std::sqrt(R_) * 
                       std::pow(delta, 1.5f);
        
        // Add damping: F_n += γ_n √(m_eff) √R √δ v_n
        float vn = glm::dot(cp.relVel, cp.normal);
        Fn_mag -= gamma_n * std::sqrt(R_ * delta) * vn;
        
        return Fn_mag * cp.normal;
    }
    
    glm::vec3 computeTangentForce(const ContactPoint& cp,
                                    float mu, float Fn_mag) const override {
        // Mindlin tangential force with incremental slip
        // For now, use simple Coulomb: F_t = -μ F_n v_t / |v_t|
        glm::vec3 vt = cp.relVel - glm::dot(cp.relVel, cp.normal) * cp.normal;
        float vt_mag = glm::length(vt);
        if (vt_mag < 1e-9f) return glm::vec3(0);
        
        return -(mu * Fn_mag / vt_mag) * vt;
    }
    
private:
    float E_;       // Young's modulus
    float nu_;      // Poisson ratio
    float R_;       // Effective radius
    float E_star_;  // Effective modulus
};
```
**Estimated lines**: +50

### 5. Scene Configuration
Update JSON schema to support spheres:
```json
{
  "scene": {
    "populate": {
      "mode": "grid",  // or "nonoverlap"
      "shape": "sphere",  // NEW: "capsule" or "sphere"
      "count": 1000,
      "radius": 0.1,
      "density": 2500
    }
  }
}
```

Update scene loader to handle sphere creation.
**Estimated lines**: +20

### 6. Visualization (Optional)
If you have rendering code, add sphere drawing (icosahedron or UV sphere mesh).
**Estimated lines**: +30-50 (if needed)

## Summary of Changes

| File | Change | Lines |
|------|--------|-------|
| `include/physics/shape.hpp` | Add Sphere struct | +5 |
| `include/physics/rigid_body.hpp` | Add sphere member, factory | +40 |
| `include/physics/contact_model.hpp` | Abstract contact model | +60 |
| `include/physics/hertz_mindlin.hpp` | Hertz-Mindlin implementation | +50 |
| Collision detection | Sphere-sphere, sphere-capsule | +75 |
| Scene loading | JSON support for spheres | +20 |
| **Total (core)** | | **~250 lines** |
| Visualization (optional) | Sphere rendering | +30-50 |

## Testing Plan

1. **Unit tests**:
   - Sphere-sphere overlap detection
   - Sphere-capsule overlap detection
   - Hertz-Mindlin force computation

2. **Integration tests**:
   - Load scene with 10 spheres, verify no crashes
   - Compare Hertz-Mindlin with existing soft contact (energy conservation)

3. **Physics validation**:
   - Reproduce Makse et al. 1999 Fig 1: σ ~ (φ - φ_c)^β
   - Check Z_c ≈ 6 for frictionless, Z_c ≈ 4 for frictional

## Implementation Priority

### Phase 1 (Minimal Viable Product)
- [ ] Add Sphere shape definition
- [ ] Add RigidBody::makeSphere()
- [ ] Implement sphere-sphere collision detection
- [ ] Test with small sphere-only scene (10-100 particles)

### Phase 2 (Mixed Systems)
- [ ] Implement sphere-capsule collision detection
- [ ] Test mixed rod-sphere scenes

### Phase 3 (Hertz-Mindlin)
- [ ] Abstract contact model interface
- [ ] Implement Hertz-Mindlin model
- [ ] Compare with current soft contact
- [ ] Reproduce Makse et al. results

### Phase 4 (Jammed Packing Studies)
- [ ] Implement isotropic compression protocol
- [ ] Measure φ_c, Z_c, force distributions
- [ ] Generate phase diagram (φ vs friction μ)

## Rollback Strategy

If issues arise:
1. **Easy rollback**: `git checkout main` to return to rod-only version
2. **Branch preservation**: Keep `feature/sphere-shape-support` branch
3. **Minimal risk**: Sphere code is additive, won't affect existing rod simulations

## Questions to Resolve

1. **Contact model**: Should we keep existing soft contact for rods and use Hertz-Mindlin only for spheres, or unify?
   - **Recommendation**: Keep both, make it a runtime choice via scene config

2. **Sphere-capsule interactions**: Needed for mixed systems?
   - **Recommendation**: Yes, implement for completeness

3. **Visualization**: Required for debugging?
   - **Recommendation**: Optional, but useful for small test cases

4. **Scene generation**: Random sphere packing algorithm?
   - **Recommendation**: Start with grid, then nonoverlap mode (reuse capsule logic)

## References

- Makse et al., "Packing of Compressible Granular Materials", Science, 1999
- Song et al., "A phase diagram for jammed matter", Nature, 2008
- Johnson, "Contact Mechanics", Cambridge, 1985
