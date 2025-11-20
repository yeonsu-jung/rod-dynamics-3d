# Rigid Motion Decomposition for Multiple Rods

Summary of Theory, Derivations, and Simulation Framework

## 1. Rigid Motions and Screw Motions

Any rigid motion in ℝ³ can be written as:

**x'** = **R****x** + **t**, where **R** ∈ SO(3), **t** ∈ ℝ³

By Chasles' theorem, every rigid motion is a **screw motion**: rotation by θ about some axis ℓ, plus translation h along that axis.

Given (**R**, **t**), the screw parameters are:

- **Axis direction** **n** = unit eigenvector of **R** with eigenvalue 1
- **Rotation angle** cos θ = ½(tr(**R**) - 1)
- **Pitch** (translation along axis) h = **n** · **t**
- **Point on axis**: Solve (**I** - **R**)**p** = **t** - (**n** · **t**)**n**

## 2. Skew Lines & Invariance Under Screw Motion

Distance between skew lines L₁ and L₂:

d(L₁, L₂) = |(**p₂** - **p₁**) · (**a** × **b**)| / ‖**a** × **b**‖

Rigid motion is an isometry, so:

- Cross products are preserved: **R**(**a** × **b**) = (**R****a**) × (**R****b**)
- Dot products are preserved
- Translation cancels in differences

Thus: **d(L₁', L₂') = d(L₁, L₂)**

## 3. When Does a Global Rigid Motion Exist?

Given many rods or points **rᵢ**(s, t), a global rigid motion exists iff:

‖**rᵢ**(s₁, t) - **rⱼ**(s₂, t)‖ is constant in time

Equivalent instantaneous condition:

d/dt ‖**rᵢ** - **rⱼ**‖² = 0 for all i, j

**Reconstruction method:**

1. Fit best rigid motion using Kabsch algorithm
2. Test residuals = zero → perfect rigid motion
3. Residuals > 0 → internal deformations

## 4. Why Two Endpoints per Rod Is Not Enough Alone

A single rod gives 2 points **p₀**, **p₁**.

Mapping these two points does not determine a unique rigid motion: we can freely rotate around the line **p₀****p₁**.

However:

- Several rods → endpoints are not collinear
- Any three non-collinear points determine a unique rigid motion

Thus, with many rods:

**Two endpoints per rod = plenty of constraints for a unique SE(3).**

## 5. Twist Representation (se(3))

Rigid body instantaneous velocity field:

**v**(**x**) = **ω** × **x** + **v₀**

The pair (**ω**, **v₀**) is a **twist**:

**ξ** = [**ω**; **v₀**] ∈ ℝ⁶

Rigid transform from twist:

**T** = exp(**ξ̂**), where **ξ̂** = [[**ω**]ₓ **v₀**; **0** 0]

## 6. Best-Fit Global Rigid Motion of Many Rods

Given marker points **xᵢ**(t) and velocities **vᵢ**(t):

Fit the closest rigid motion:

**vᵢ**_rigid = **ω** × **xᵢ** + **v₀**

by minimizing:

Σᵢ mᵢ ‖**vᵢ** - (**ω** × **xᵢ** + **v₀**)‖²

This is a projection of the velocity field onto the 6D subspace of rigid motions.

Then:

**vᵢ**_def = **vᵢ** - **vᵢ**_rigid

This gives a clean decomposition:

- **Global motion** = bulk drift + rotation
- **Local motion** = bending, shearing, entangling, rod rearrangement

## 7. Energy Decomposition

Total kinetic energy:

T = ½ Σᵢ mᵢ ‖**vᵢ**‖²

Insert **vᵢ** = **vᵢ**_rigid + **vᵢ**_def:

**T = T_global + T_def**

because the projection ensures:

Σᵢ mᵢ **vᵢ**_rigid · **vᵢ**_def = 0

Meaning:

- **Global kinetic energy**: T_global = ½ Σ mᵢ ‖**vᵢ**_rigid‖²
- **Deformational kinetic energy**: T_def = ½ Σ mᵢ ‖**vᵢ**_def‖²

This is exactly the energy split used in:

- Continuum mechanics (rigid + strain energy)
- Multibody dynamics
- Granular flows with "fluctuation" kinetic energy
- Turbulence (Reynolds decomposition)

## 8. Python Pipeline Summary

1. Sample markers on rods (e.g. endpoints)
2. Compute velocities by finite difference
3. Fit twist (**ω**, **v₀**) by least squares
4. Compute:
   - Global velocity **vᵢ**_rigid
   - Deformational velocity **vᵢ**_def
   - Global + deformational kinetic energies
5. Visualize rods and vectors
6. (Optional) Fit full SE(3) using Kabsch to track positions across frames

## 9. Visualization Insights

We showed that:

- **One rod alone** → infinite rigid motions (free rotation about rod axis)
- **Two rods** → endpoints give enough constraints → unique rigid motion

The two-rod visualization makes this intuitively clear:

Rotate around rod A → rod B no longer aligns → only one SE(3) fits both rods simultaneously.

## 10. What We Can Package Next

A downloadable .zip file containing:

- `README.md` (this summary)
- `rigid_motion_tutorial.py` (all code: rods, twist fit, Kabsch, energy decomposition)
- Visualization scripts
- Examples + synthetic data generation
- Optional JAX version
- Optional screw-motion animation

---

## Key Equations Reference

### Twist Fitting (Least Squares)

Given positions **xᵢ** and velocities **vᵢ**, find (**ω**, **v₀**) minimizing:

E = Σᵢ mᵢ ‖**vᵢ** - (**ω** × **xᵢ** + **v₀**)‖²

Solution:

- **v₀** = (Σᵢ mᵢ **vᵢ**) / (Σᵢ mᵢ) - **ω** × **x̄**
- **ω** solves: **J****ω** = **b**
  - **J** = Σᵢ mᵢ [**r̃ᵢ**]ₓ²
  - **b** = Σᵢ mᵢ [**r̃ᵢ**]ₓ **ṽᵢ**
  - **r̃ᵢ** = **xᵢ** - **x̄**
  - **ṽᵢ** = **vᵢ** - **v̄**

### Kabsch Algorithm (Position Alignment)

Given corresponding point sets {**pᵢ**} and {**qᵢ**}, find optimal rotation **R**:

1. Center both sets: **p̃ᵢ** = **pᵢ** - **p̄**, **q̃ᵢ** = **qᵢ** - **q̄**
2. Compute covariance: **H** = Σᵢ **p̃ᵢ** **q̃ᵢ**ᵀ
3. SVD: **H** = **U****Σ****Vᵀ**
4. **R** = **V** diag(1, 1, det(**V****Uᵀ**)) **Uᵀ**
5. **t** = **q̄** - **R****p̄**

### Energy Orthogonality

‖**v**‖² = ‖**v**_rigid‖² + ‖**v**_def‖² + 2(**v**_rigid · **v**_def)

The projection ensures the cross term vanishes:

Σᵢ mᵢ (**v**_rigid · **v**_def) = 0

Therefore:

T_total = T_rigid + T_def (exact decomposition, no cross terms)
