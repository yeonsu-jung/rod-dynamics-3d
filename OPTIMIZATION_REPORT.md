# Optimization Report: Soft Contact Broadphase

## 1. Executive Summary
The primary objective was to improve the performance of the soft contact collision detection system in `rod-dynamics-3d`. By enabling Spatial Hashing and implementing a specific optimization for Axis-Aligned Bounding Box (AABB) caching, we achieved a **~2.4x speedup** in the broadphase collision detection stage (from ~86ms to ~36ms per frame for ~750 rods).

## 2. Problem Identification
Initial profiling revealed two main issues:
1.  **Naive Complexity**: The solver defaulted to a "Naive" O(N²) all-pairs check in some configurations, which is computationally expensive for large numbers of rods ($N > 100$).
2.  **Redundant Calculations**: Even when Spatial Hashing was enabled, the `detectContactsSpatialHash` function was inefficient. Inside the inner loop (where pairs of potential collisions are checked), the code was re-calculating the AABB for every body involved in a check.
    *   *Scenario*: If Rod A is in a grid cell with 50 other rods, Rod A's bounding box was being re-calculated 50 times.

## 3. Optimization Strategy

### A. Spatial Hashing (Algorithmic Improvement)
We enforced the use of a **Spatial Hash** grid. This divides the 3D space into cells. Instead of checking every rod against every other rod, we only check rods that share the same grid cell. This reduces complexity from $O(N^2)$ to approximately $O(N)$ in sparse scenes.

### B. AABB Caching (Micro-Optimization)
We modified `src/physics/soft_contact.cpp` to pre-calculate AABBs.

*   **Before**:
    ```cpp
    // Inside the nested loop over cell neighbors...
    if (checkAABBOverlap(rods[i], rods[j])) { ... }
    // checkAABBOverlap would re-compute the min/max extent of rods[i] and rods[j]
    ```
*   **After**:
    ```cpp
    // Pre-pass: Compute AABBs for all bodies once
    static std::vector<glm::vec3> aabb_min, aabb_max;
    // ... resize and fill ...

    // Inside the nested loop...
    // Direct float comparisons using cached values
    if (aabb_max[i].x < aabb_min[j].x || aabb_min[i].x > aabb_max[j].x) continue;
    // ... (repeat for Y and Z)
    ```
This removed the overhead of matrix-vector multiplications and geometry calculations from the "hot loop" of the collision detector.

## 4. Benchmark Results

Tests were conducted using `rigidbody_viewer_3d --headless` with a population of 1000 rods (resulting in ~758 placed rods).

| Metric | Initial (Spatial Hash) | Optimized (AABB Caching) | Improvement |
| :--- | :--- | :--- | :--- |
| **Broadphase Time** | ~86.0 ms | **~36.7 ms** | **2.35x Faster** |
| **Total Frame Time** | ~87.5 ms | ~38.0 ms | **2.30x Faster** |

*Note: The "Naive" O(N²) approach (not shown) was significantly slower than both.*

## 5. Configuration & Code Changes

### CLI Improvements (`src/app/main.cpp`)
Added command-line flags to allow rapid tuning without recompilation:
*   `--use-spatial-hash` / `--no-spatial-hash`
*   `--use-aabb` / `--no-aabb`
*   `--cell-size <float>`
*   `--verbose-soft` / `--no-verbose-soft` (Controls log spam)

### Default Configuration (`assets/scenes/default.json`)
Updated the default scene settings to ensure users get the best performance out of the box:
```json
"soft_contact": {
  "verbose": false,
  "use_spatial_hash": true,
  "use_aabb": true,
  "cell_size": -1.0  // Auto-tuned
}
```

## 6. Conclusion
The physics engine is now significantly more efficient for large-scale simulations. The bottleneck has been effectively widened, allowing for real-time or near-real-time simulation of dense rod configurations that were previously too slow to simulate practically.
