# Development Checkpoint: Before Sphere Implementation

**Date**: November 23, 2025  
**Branch**: `feature/sphere-shape-support` (created from `main`)  
**Commit**: d886c26

## What We've Completed and Protected

### 1. Dynamic Rod Count Formula ✅
- **Implementation**: `parametric_study/submit_aspect_ratio.py`
- **Formula**: N = 6 × L³ × AR / l³
- **Results**: AR=50 → 399 rods, AR=100 → 799 rods, etc.
- **Status**: Tested and working

### 2. Circular/Ring COM Method ✅
- **Implementation**: `src/app/main.cpp::computeCOM()`
- **Method**: Maps positions to angles, averages via complex numbers
- **Advantage**: No wrapping discontinuities, works for particles spanning box
- **Status**: Implemented and tested, produces smoother trajectories

### 3. Analysis Tools ✅
- **Script**: `parametric_study/quick_analyze_run.py`
- **Features**: KE dynamics, COM movement, automated plot generation
- **Status**: Fully functional

### 4. Algorithm Verification ✅
- **Files**: `test_nonoverlap.cpp`, `verify_nonoverlap.cpp`
- **Result**: Nonoverlap algorithm confirmed correct
- **Status**: Documented and tested

## Git Strategy: Easy Rollback

### Current State
```
main (safe, stable)
  ├─ feat: Dynamic rod counts, circular COM, analysis tools [committed]
  └─ feature/sphere-shape-support (current branch)
       └─ docs: Sphere implementation design [committed]
```

### Rollback Options

#### Option 1: Abandon Sphere Work
```bash
git checkout main
git branch -D feature/sphere-shape-support
```
**Effect**: Returns to rod-only codebase, all recent improvements preserved

#### Option 2: Temporarily Switch Back
```bash
git checkout main
# Work on rod simulations
git checkout feature/sphere-shape-support
# Resume sphere work
```
**Effect**: Can easily switch contexts

#### Option 3: Merge Improvements, Defer Spheres
```bash
git checkout main
# Current main already has rod improvements committed
# If you want to keep design doc:
git checkout feature/sphere-shape-support -- docs/sphere_implementation_design.md
git add docs/sphere_implementation_design.md
git commit -m "docs: Add sphere design (future work)"
git branch -D feature/sphere-shape-support
```
**Effect**: Keep design doc, delete implementation branch

## What Happens Next: Sphere Implementation

### Phase 1: Minimal Viable Product (MVP)
**Goal**: Get spheres working in simplest possible way

**Files to modify**:
1. `include/physics/shape.hpp` (+5 lines)
   - Add `Sphere` struct, extend `ShapeType` enum

2. `include/physics/rigid_body.hpp` (+40 lines)
   - Add `Sphere sphere{}` member
   - Add `makeSphere()` factory method

3. Collision detection file (~+75 lines)
   - Implement `detectSphereSphere()`
   - Add sphere case to collision dispatcher

4. Scene loading (~+20 lines)
   - Parse `"shape": "sphere"` from JSON
   - Call `makeSphere()` during population

**Test**: Load scene with 100 spheres, verify they bounce and don't overlap

### Phase 2: Contact Model Separation (Optional)
**Goal**: Keep rod and sphere physics independent

Create `include/physics/contact_model.hpp` abstraction:
- Base class: `ContactModel`
- Derived: `SoftContactModel` (current, for rods)
- Derived: `HertzMindlinModel` (new, for spheres)

**Advantage**: Can compare rod behavior with both models

### Phase 3: Jammed Packing Studies
**Goal**: Reproduce Makse et al. 1999, 2004 results

**Workflow**:
1. Generate sphere packing at low φ
2. Isotropically compress until φ_target
3. Measure σ, Z, force distributions
4. Repeat for multiple friction coefficients μ

**Expected Results**:
- φ_c ≈ 0.634 (frictionless), φ_c ≈ 0.6284 (frictional)
- σ ~ (φ - φ_c)^β with β ≈ 1.6-2.0
- Force distributions: exponential → Gaussian transition

## Design Decisions to Make

### 1. Contact Model Strategy
**Options**:
- **A**: Keep existing soft contact for everything (rods + spheres)
  - Pro: Simple, no architectural changes
  - Con: Not physically accurate for Hertz-Mindlin comparison
  
- **B**: Add Hertz-Mindlin alongside soft contact
  - Pro: Can compare models, reproduce literature
  - Con: More code, need abstraction layer

- **C**: Replace soft contact with Hertz-Mindlin globally
  - Pro: Single consistent model
  - Con: May change rod behavior unexpectedly

**Recommendation**: **Option B** (add Hertz-Mindlin, keep soft contact)

### 2. Sphere-Capsule Interactions
**Question**: Should spheres and rods interact in same simulation?

**Options**:
- **A**: Sphere-only simulations (skip sphere-capsule detection)
  - Pro: Simpler initial implementation
  - Con: Limits future mixed-system studies
  
- **B**: Implement sphere-capsule detection
  - Pro: Complete, enables interesting heterogeneous systems
  - Con: +30 lines of geometry code

**Recommendation**: **Option A initially**, add B later if needed

### 3. Scene Configuration
**Current**:
```json
{
  "populate": {
    "mode": "nonoverlap",
    "shape": "capsule",
    "count": 200,
    "length": 1.0,
    "radius": 0.05,
    "aspect_ratio": 20.0
  }
}
```

**Proposed for spheres**:
```json
{
  "populate": {
    "mode": "nonoverlap",
    "shape": "sphere",
    "count": 1000,
    "radius": 0.1,
    "density": 2500.0
  }
}
```

**Recommendation**: Add shape-specific fields, validate at load time

## Risk Assessment

### Low Risk ✅
- Adding `Sphere` struct (pure data, no logic)
- Adding `makeSphere()` factory (isolated function)
- Sphere-sphere collision (self-contained geometry)

### Medium Risk ⚠️
- Modifying collision dispatcher (affects all shapes)
- Scene loading changes (could break existing scenes)

### High Risk 🚨
- Changing contact model globally (affects rod physics)
- Modifying broadphase acceleration structure

### Mitigation
1. **Unit tests**: Test each new function independently
2. **Regression tests**: Run existing rod scenes after changes
3. **Branch isolation**: Keep sphere work separate until validated
4. **Incremental commits**: Small, revertable changes

## Communication Plan

### If Things Go Wrong
**Scenario**: Sphere implementation breaks rod simulations

**Action**:
```bash
# Immediately roll back
git checkout main

# Investigate on branch
git checkout feature/sphere-shape-support
git log  # Find breaking commit
git revert <commit-hash>
# Or git reset --hard <last-good-commit>
```

**Prevention**: Run rod regression test after each sphere commit:
```bash
# Quick smoke test
./build-debug/rigidbody_viewer_3d --scene assets/scenes/confined_n2.json --headless --steps 100
echo $?  # Should exit 0
```

## Separation Strategy: Contact Models

### Why Separate?
1. **Rods (capsules)**: Current soft contact works well
   - k_n, γ_n parameters tuned
   - Stable for large AR rods
   
2. **Spheres**: Hertz-Mindlin is standard in literature
   - Makse et al. used (2/3) k_n R^{1/2} δ^{3/2}
   - Direct comparison with experiments

3. **Different physics**: 
   - Hertz: δ^{3/2} nonlinearity
   - Soft: Linear penalty

### Implementation Approach

**Step 1**: Abstract base class
```cpp
// include/physics/contact_model.hpp
class ContactModel {
public:
    virtual ~ContactModel() = default;
    virtual glm::vec3 computeNormalForce(
        const ContactPoint& cp,
        float kn, float gamma_n
    ) const = 0;
};
```

**Step 2**: Wrap existing soft contact
```cpp
// include/physics/soft_contact_model.hpp
class SoftContactModel : public ContactModel {
public:
    glm::vec3 computeNormalForce(
        const ContactPoint& cp,
        float kn, float gamma_n
    ) const override {
        // Existing code from main.cpp
        return (kn * cp.depth - gamma_n * vn) * cp.normal;
    }
};
```

**Step 3**: Add Hertz-Mindlin for spheres
```cpp
// include/physics/hertz_mindlin_model.hpp
class HertzMindlinModel : public ContactModel {
public:
    HertzMindlinModel(float E, float nu, float R)
        : E_(E), nu_(nu), R_(R) {}
    
    glm::vec3 computeNormalForce(
        const ContactPoint& cp,
        float kn, float gamma_n
    ) const override {
        // F_n = (4/3) E* √R δ^(3/2)
        float Fn = (4.0f/3.0f) * E_ * std::sqrt(R_) * 
                   std::pow(cp.depth, 1.5f);
        return Fn * cp.normal;
    }
    
private:
    float E_, nu_, R_;
};
```

**Step 4**: Choose model per shape type
```cpp
// In physics solver:
const ContactModel* model = nullptr;
if (rb.type == ShapeType::Capsule) {
    model = &softContactModel;  // Existing for rods
} else if (rb.type == ShapeType::Sphere) {
    model = &hertzMindlinModel;  // New for spheres
}

glm::vec3 F_normal = model->computeNormalForce(contact, kn, gamma_n);
```

**Advantage**: Rods and spheres never interfere with each other's physics!

## Next Steps (Your Choice)

### Option A: Start Sphere Implementation Now
1. Create `include/physics/shape.hpp` changes
2. Add `RigidBody::makeSphere()`
3. Implement sphere-sphere collision
4. Test with small scene (10-100 spheres)

**Time estimate**: 2-3 hours for MVP

### Option B: Run Full Rod Parametric Study First
1. Ensure current rod code is production-ready
2. Submit aspect ratio sweep to cluster:
   ```bash
   cd parametric_study
   python submit_aspect_ratio.py --job-name ar_sweep_full
   ```
3. Analyze results with `quick_analyze_run.py`
4. Return to spheres later

**Time estimate**: 1-2 weeks (mostly cluster time)

### Option C: Defer Spheres, Focus on Rod Analysis
1. Keep `feature/sphere-shape-support` branch for later
2. Checkout `main`, continue rod studies
3. Revisit spheres when rod work is published

**Time estimate**: Indefinite (come back to spheres as needed)

## What to Tell Collaborators

### If asked: "What's the current status?"
> "We've implemented dynamic rod counts based on the N = 6L³AR/l³ formula, 
> fixed the COM calculation to use circular statistics for periodic boundaries,
> and created analysis tools for KE and COM trajectories. All changes are 
> committed to `main` branch and tested.
>
> We're exploring adding sphere support for jammed packing studies (Makse et al.),
> but that's on a separate feature branch. We can easily roll back if needed."

### If asked: "Should we add spheres?"
> "It's optional. The design is documented, implementation is ~250 lines, 
> and we have a clean rollback strategy. Spheres would enable reproducing 
> classic jammed packing results (RCP/RLP transitions, force chains, etc.).
> But it's not required for rod-only studies."

### If asked: "Will spheres break rod code?"
> "No. The architecture is additive—we're adding new shape types and collision
> detectors, not modifying existing rod physics. We can even use different 
> contact models for rods vs spheres. The changes are isolated."

## Summary

**What's protected**: ✅
- Dynamic rod count formula
- Circular COM calculation
- Analysis tools
- All tested and committed

**What's planned**: 📋
- Sphere shape support (optional)
- ~250 lines of code
- 4-phase implementation
- Easy rollback strategy

**Decision needed**: ❓
- Start sphere implementation?
- Run full rod study first?
- Defer spheres indefinitely?

**Recommendation**: **Your choice!** All infrastructure is in place for either path.
