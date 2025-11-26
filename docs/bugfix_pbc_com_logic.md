# Fix: Center of Mass Relative Displacement in Periodic Boundary Conditions

## Overview

This document details a critical fix applied to the analysis logic regarding the Center of Mass (CoM) and relative particle displacements in a Periodic Boundary Condition (PBC) environment.

## The Problem

In a periodic domain (e.g., a torus), the "center" of a cluster of particles is calculated using the **Circular Mean** method (mapping positions to angles, averaging vectors, and mapping back). This part of the code was correct.

However, two critical bugs existed in how this CoM was **used** to calculate distances and displacements.

### Bug 1: Naive Euclidean Distance in `logRelDispFrame`

When logging the displacement of each rod relative to the CoM (`d = rod.x - com`), the code performed a simple vector subtraction.

**Scenario:**

- Domain Size $L = 10$
- CoM is at $x = 9.9$
- Rod is at $x = 0.1$

**Incorrect Calculation:**

$$ dx = 0.1 - 9.9 = -9.8 $$

The analysis interprets this as the rod being far away on the other side of the box.

**Correct Calculation (Minimum Image):**

The rod is actually only $0.2$ units away (wrapping around the edge).

$$ dx = 0.2 $$

### Bug 2: Inflated Dispersion in `relDispAvgL2`

The function `relDispAvgL2` calculates the "spread" of the rods (Average $L^2$ norm). Because it also used naive subtraction, any time a cluster of rods straddled the periodic boundary, the dispersion metric would spike massively, as rods on one side of the boundary were treated as being $L$ distance away from rods on the other side.

## The Fix: Minimum Image Convention

We applied the **Minimum Image Convention** to both functions. This ensures that the vector $d$ always points to the *nearest* image of the CoM.

### Code Change

**Before:**

```cpp
glm::vec3 d = rods[i].x - rc;
```

**After:**

```cpp
glm::vec3 d = rods[i].x - rc;
if (usePBC) {
    for (int k = 0; k < 3; ++k) {
        if (boxSize[k] > 0.0f) {
            // Wrap delta to range [-L/2, L/2]
            d[k] -= boxSize[k] * std::floor(d[k] / boxSize[k] + 0.5f);
        }
    }
}
```

## Impact

- **Artifacts Removed:** The "jumps" seen in time-series plots of relative displacement are eliminated.
- **Continuous Metrics:** Dispersion and relative positions now vary smoothly even when the entire cluster moves across the periodic boundary.
