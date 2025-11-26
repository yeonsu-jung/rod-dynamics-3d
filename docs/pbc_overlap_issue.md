# Investigation: Periodic Boundary Condition Overlaps in Rod Generation

**Date:** November 26, 2025  
**Status:** Resolved  
**Component:** Configuration Generator (`rods_pbc_bruteforce.cpp`)

## 1. The Issue

Despite the configuration generator reporting zero overlaps ("gap-checked"), the simulation diagnostics (`minPairGap`) consistently reported deep penetrations (negative gaps) in the generated initial configurations.

- **Symptom:** "Valid" CSV files (e.g., `v5.csv`) caused immediate explosions or high energy in the simulation.
- **Diagnostic:** The simulator reported `minPairGap` values around `-0.005` (overlap), while the generator claimed `minGap > 0`.

## 2. Root Cause Analysis

### The "Nearest Image" Fallacy

The original generator used a standard "Nearest Image" convention for collision detection. For a pair of objects $A$ and $B$ in a periodic box of size $L$, it calculated the distance between $A$ and the image of $B$ closest to $A$.

$$ \vec{r}_{AB} = \vec{r}_B - \vec{r}_A $$
$$ \vec{r}_{AB}' = \vec{r}_{AB} - L \cdot \text{round}(\vec{r}_{AB} / L) $$

This approach is valid for **point particles** or objects significantly smaller than the box ($d \ll L$).

### The Failure Case

In our setup:

- **Box Size ($L$):** 1.1
- **Rod Length ($l$):** 1.0

Because the rod length is nearly equal to the box size, a rod can effectively interact with multiple periodic images of a neighbor simultaneously, or interact with a periodic image that is *not* the "nearest" one based on center-of-mass distance.

**Example Scenario:**

1. Rod A is near the left boundary.
2. Rod B is near the right boundary.
3. The "nearest image" check considers $B$'s image shifted by $-L$ (to the left of A).
4. However, due to their orientation and length, Rod A might actually be colliding with $B$'s image shifted by $+L$ (or another neighbor), or the collision might occur at the tips which extend beyond the "nearest" zone.

We verified this by manually calculating the Euclidean distance between colliding rods (IDs 11 and 428) in Python. The generator reported a large distance (safe), while the actual Euclidean distance in 3D space showed a deep overlap ($0.0003 < \text{diameter}$).

## 3. The Solution: 27-Image Brute Force

We replaced the heuristic "nearest image" check with a robust brute-force check. For every pair of rods, we check collisions against **all 27 possible periodic images** of the neighbor.

**Old Logic:**

```cpp
// Only checked the single nearest image
Vector3 neighbor = nearest_image(B, relative_to=A);
if (dist(A, neighbor) < threshold) return collision;
```

**New Logic:**

```cpp
// Checks all 27 images (3x3x3 grid)
for (int i = -1; i <= 1; ++i) {
    for (int j = -1; j <= 1; ++j) {
        for (int k = -1; k <= 1; ++k) {
            Vector3 offset(i*L, j*L, k*L);
            Vector3 B_image = B.pos + offset;
            if (dist(A, B_image) < threshold) return collision;
        }
    }
}
```

This ensures that no matter how the rods are oriented or wrapped, if *any* part of them overlaps in the periodic domain, it is detected.

## 4. Verification

After updating `rods_pbc_bruteforce.cpp` and generating `v6.csv`:

1. **Generator Output:** `All-pairs PBC min gap: 4.29e-06` (Positive, valid).
2. **Simulator Check:**

   ```bash
   ./rigidbody_viewer_3d ... --check-init-nonpenetration
   ```

   **Result:** `[init-check] minPairGap = 4.26453e-06`.

The simulator and generator now agree, and the system starts with zero contacts and zero potential energy.
