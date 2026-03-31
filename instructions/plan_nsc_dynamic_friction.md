# Plan: NSC Dynamic Friction via Maximum Dissipation

## 1. Why PSOR Failed (The Cause of the KE Explosion)
The standard NSC solver uses Projected Gauss-Seidel (PGS) with "Box Friction". In box friction, the two tangential directions ($t_1$ and $t_2$) are strictly solved **one after the other**. 
1. Solve for $t_1$, clamp to $\pm \mu \lambda_n$, and **immediately update the bodies' relative velocities**.
2. Solve for $t_2$ using the *newly updated* velocities, clamp, and update again.

Because the velocities are updated sequentially after every single 1D axis, the Gauss-Seidel algorithm guarantees that the system's kinetic energy monotonically decreases (or stays stable). 

In my previous attempt to enforce a perfect "circular cone" (Maximum Dissipation Principle), I computed the required impulses for $t_1$ and $t_2$ **simultaneously** using the same initial $v_{rel}$, combined them into a 2D vector, clamped the vector's length, and applied them together. This broke the sequential Gauss-Seidel guarantee! Treating coupled variables simultaneously without doing a full $2 \times 2$ matrix inversion represents a Jacobi-style update. When friction limits were hit, the overlapping impulses overshot the roots, pumping artificial kinetic energy into the rods and causing the explosion.

## 2. How Multiple Contacts are Implemented
Multiple contacts are organically resolved by the **outer PSOR loop**. 
The solver maintains a list of all contact points (`manifolds_`). It loops over every contact mathematically:
1. Contact A applies an impulse to Rod 1 and Rod 2.
2. Rod 1 and Rod 2 instantly have their linear velocity ($v$) and angular velocity ($w$) updated in memory.
3. If Contact B also involves Rod 2, when the loop reaches Contact B, it computes $v_{rel}$ using the *new* velocities of Rod 2 that were just altered by Contact A.

By repeating this loop many times (`velocity_iters = 40`), the forces propagate through the entire densely packed rod bundle until the whole network converges to a state where no rod is penetrating and friction bounds are satisfied everywhere.

## 3. Emulating `f ~ -mu * N * (v_t / |v_t|)` Safely
To implement your explicit maximum dissipation heuristic without breaking the PSOR stability, we will replace the box-friction solver with a direct, stable 2D velocity-drain step inside the Gauss-Seidel loop.

Instead of keeping track of accumulated tangential constraint multipliers ($\lambda_{t1}$, $\lambda_{t2}$) which is complex for a 2D circle, we will simply treat friction as an explicit damping impulse strictly aligned against instantaneous sliding:

**For each contact (after solving the Normal impulse $\lambda_n$):**
1. Compute the instantaneous 2D tangential sliding velocity:  
   $v_t = v_{rel} - (v_{rel} \cdot n) n$
2. If $|v_t|$ is negligibly small, we are sticking and do nothing.
3. Otherwise, determine the sliding direction:  
   $dir = \frac{v_t}{|v_t|}$
4. Compute the effective mass along that specific sliding direction:  
   $g_t = \text{computeDiag}(\dots, dir)$
5. Calculate the exact impulse needed to completely stop the sliding:  
   $j_{\text{stop}} = \frac{|v_t|}{g_t}$
6. Bound this impulse by your Coulomb friction law limit:  
   $j_{\text{apply}} = \min(j_{\text{stop}}, \, \mu \cdot \lambda_n)$
7. Apply the frictional impulse precisely opposing the slide:  
   $\Delta V = - j_{\text{apply}} \cdot dir$
8. Apply to bodies and instantly update relative velocities for the next contact in the loop.

This achieves exactly what you requested: applying a friction force parameterized by $\mu N$ exactly opposite to the instantaneous $v_t$, while the $j_{\text{stop}}$ bound unconditionally guarantees the force will never overshoot and cause reverse swimming (which avoids the explosion!).
