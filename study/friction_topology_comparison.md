# Friction Effects on Topological Evolution

## Comparison: mu=1.0 vs mu=0.05

### Dataset Information

| Parameter | mu=1.0 (high friction) | mu=0.05 (low friction) |
|-----------|------------------------|------------------------|
| Total frames | 107 | 107 |
| Rods (N) | 200 | 200 |
| Aspect ratio | 300 | 300 |

## Topological Evolution Results

### mu=1.0 (High Friction)

Analysis of frames 0 to 100 (step 10):

| Frame | Total Chirality C | Change from frame 0 |
|-------|-------------------|---------------------|
| 0     | -8,704           | —                   |
| 10    | -9,088           | -384                |
| 50    | -7,344           | +1,360              |
| 100   | -7,420           | +1,284              |

**Behavior**: Stable. The system remains in the same topological basin ($C \approx -8000$).
**Variance**: $3.1 \times 10^5$

### mu=0.05 (Low Friction)

Analysis of frames 0 to 100 (step 10):

| Frame | Total Chirality C | Change from frame 0 |
|-------|-------------------|---------------------|
| 0     | -8,704           | —                   |
| 10    | -11,976          | -3,272              |
| 50    | -9,788           | +1,084              |
| 100   | +1,584           | +10,288             |

**Behavior**: Unstable. Large fluctuations and **sign reversal** ($C > 0$) at the end.
**Variance**: $1.9 \times 10^7$ (60× higher than mu=1.0)

## Key Findings

### 1. Friction Stabilizes Topology
The new data confirms the hypothesis that high friction acts as a topological constraint. The magnitude of topological change decreases significantly with friction.

![Topological Change vs Friction](delta_C_vs_mu.png)

*   **mu=1.0**: Topology is approximately preserved. The rods stay entangled in a similar knot type (indicated by the sign and magnitude of $C$).
*   **mu=0.05**: Topology is not preserved. Lower friction allows rods to slide past each other more easily, leading to global rearrangement.

### 2. Identifying Stable Rods
We identified the **Stable Core**: the largest subset of rods that maintain their relative topological relationships (constant $v_{ijk}$) throughout the simulation.

![Stable Core Size vs Friction](stable_core_vs_mu.png)

There is a **Phase Transition** between $\mu=0.2$ and $\mu=0.4$:
*   **Unstable Regime ($\mu \le 0.2$)**: The system behaves like a liquid. Only ~10-15% of rods maintain their topology.
    *   $\mu=0.05$: Stable core = 17 rods (8.5%)
*   **Stable Regime ($\mu \ge 0.4$)**: The system behaves like a glass/solid. Most rods are locked in place.
    *   $\mu=1.0$: Stable core = 165 rods (82.5%)

### 3. Packing Density of Stable Core
We also analyzed the **Average Pairwise Distance** between rods within the stable core.

![Avg Pair Distance vs Friction](stable_core_avg_dist_vs_mu.png)

*   **Low Friction ($\mu \le 0.2$)**: The few stable rods are far apart (Avg Dist > 1.0). This suggests the "stable core" is just a set of distant, non-interacting rods that happen not to move much relative to each other.
*   **High Friction ($\mu \ge 0.4$)**: The stable core is highly compact (Avg Dist < 0.1). This indicates a **tightly interlocked** structure where valid topological constraints are actively preserving the configuration.

## Comparison with New Simulator (N=1000, T=200k)

We performed the same analysis on a new, larger dataset (N=1000) with much longer simulation time (200,000 frames).

### Results Summary

| $\mu$ | Stable Core Size | % Stable | Avg Pair Dist |
|-------|------------------|----------|---------------|
| 0.0   | 36               | 3.6%     | 12.8          |
| 0.05  | 95               | 9.5%     | 1.46          |
| 0.1   | 406              | 40.6%    | 0.32          |
| 0.15  | 582              | 58.2%    | 0.21          |
| 0.2   | 636              | 63.6%    | 0.19          |
| 0.4   | 679              | 67.9%    | 0.18          |
| 1.0   | 693              | 69.3%    | 0.18          |

![Stable Core N=1000](stable_core_vs_mu_new.png)
![Avg Dist N=1000](stable_core_avg_dist_vs_mu_new.png)

### Key Findings (New Simulator)

1.  **Earlier Phase Transition**: The transition from liquid-like to solid-like behavior happens earlier, between **$\mu=0.05$ and $\mu=0.15$**.
    *   At $\mu=0.1$, the core is already 40% stable.
    *   At $\mu=0.2$, the core is 64% stable (compared to only 15% in the older N=200 dataset).

2.  **Saturation**: The stable core saturates at ~70% of the rods, slightly lower than the 82% observed in the N=200 case. This might be due to the much longer simulation time allowing more "edge" rods to eventually escape or fluctuate.

3.  **Density Saturation**: The average pair distance saturates at ~0.18 for $\mu \ge 0.15$, indicating a very dense, interlocked core forms even at moderate friction.

### Effect of Filtering "Noisy" Triples
We further refined the analysis by **filtering out nearly parallel rods** (angle < 5°). Topological invariants involving parallel rods are sensitive to small fluctuations (noise). Eliminating these "dubious" constraints consistently increases the size of the identified stable core.

![Stable Core Filtered](stable_core_vs_mu_filtered.png)

| $\mu$ | Standard Core | Filtered Core (<5°) | Increase |
|-------|---------------|---------------------|----------|
| 0.0   | 36            | 37                  | +1       |
| 0.05  | 95            | 103                 | +8       |
| 0.1   | 406           | 428                 | +22      |
| 0.15  | 582           | 607                 | +25      |
| 0.2   | 636           | 657                 | +21      |
| 0.4   | 679           | 700                 | +21      |
| 1.0   | 693           | 714                 | +21      |

**Conclusion**: The phase transition is robust. Filtering "almost coplanar" interactions clarifies that the stable core is even larger (~71.4% at max) than initially estimated, as some "instabilities" were merely geometric noise.

### 2. Sign Reversal
The low friction case undergoes a dramatic transition where the total chirality flips sign (from -8704 to +1584). This implies a complete inversion or restructuring of the chiral nature of the packing, which is physically impossible without rods passing through each other or significant boundary movement.

### 3. Quantifying stability
The variance of the topological invariant $C$ is **60 times higher** in the low friction case.

## Physical Interpretation

*   **Energy Barriers**: High friction creates effective energy barriers against sliding, trapping the system in a local topological minimum.
*   **Timescale**: The "topological relaxation time" for mu=1.0 is clearly much longer than the simulation window, whereas for mu=0.05 it is comparable to the simulation time.

## Conclusion

Friction plays a critical role in preserving the topology of rod packings. High friction (`mu=1.0`) successfully maintains the topological state, while low friction (`mu=0.05`) permits rapid and extensive topological evolution.
