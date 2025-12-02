# Feature Report: Sphere-Capsule Collision Support

## 1. Overview
We have successfully implemented the missing **Sphere-Capsule** collision detection in the soft contact solver. This completes the collision matrix for the supported shapes (Sphere, Capsule).

## 2. Changes Implemented

### A. Collision Logic (`src/physics/soft_contact.cpp`)
*   **New Function**: `detectSphereCapsule`
    *   Calculates the closest point on the capsule's central segment to the sphere's center.
    *   Computes distance and checks against the sum of radii + margin.
    *   Generates contact points and normals.
    *   Handles both "Edge-to-Point" (hitting the cylinder side) and "Point-to-Point" (hitting the hemispherical end-caps) cases.

### B. Dispatch Logic
*   Updated `detectContactsNaive` and `detectContactsSpatialHash` to handle mixed shape pairs:
    *   `Sphere` vs `Capsule`
    *   `Capsule` vs `Sphere` (swapped arguments)

### C. Optimization
*   **AABB Caching**: Extended the AABB optimization (previously only for Capsule-Capsule) to **all** shape pairs in the Spatial Hash broadphase.
    *   This ensures that Sphere-Sphere and Sphere-Capsule checks also benefit from the fast rejection of non-overlapping bounding boxes.

## 3. Verification
*   **Test Scene**: Created a test scene with a sphere falling onto a static horizontal capsule.
*   **Result**: Contacts were successfully detected at the expected time (frame ~474), and the physics engine responded by generating forces (implied by the contact persistence).
*   **Profiling**: Confirmed that the AABB check is active and working for these new pairs.

## 4. Next Steps
*   The `feature/sphere-shape-support` branch goal appears to be met regarding collision detection.
*   Further testing with complex granular scenes (e.g., `mixed_spheres_capsules.json`) is recommended to tune friction and stiffness parameters for mixed collisions.
