# Topological Evolution vs Friction Coefficient

## Complete Analysis: |ΔC| as a Function of μ

Analysis of 6 datasets with friction coefficients μ = 0.05, 0.1, 0.15, 0.2, 0.4, 1.0

### Results Summary

| μ | C_initial | C_final | ΔC | |ΔC| | % Change |
|---|-----------|---------|-----|------|----------|
| 0.05 | +9,324 | +1,840 | -7,484 | **7,484** | -80.3% |
| 0.1 | +5,678 | +8,296 | +2,618 | **2,618** | +46.1% |
| 0.15 | +4,570 | -5,084 | -9,654 | **9,654** | -211.2% |
| 0.2 | +5,680 | -8,120 | -13,800 | **13,800** | -242.9% |
| 0.4 | +6,004 | -8,840 | -14,844 | **14,844** | -247.2% |
| 1.0 | +13,970 | -3,148 | -17,118 | **17,118** | -122.5% |

### Key Findings

#### 1. **Unexpected Non-Monotonic Relationship**

Contrary to initial expectations, |ΔC| does **NOT** decrease monotonically with increasing μ:

- **Minimum at μ = 0.1**: |ΔC| = 2,618 (least topological change)
- **Maximum at μ = 1.0**: |ΔC| = 17,118 (most topological change!)
- **Ratio**: 6.5× difference between extremes

#### 2. **Two Distinct Regimes**

**Low friction regime (μ ≤ 0.1):**
- Moderate topological change
- μ = 0.05: |ΔC| = 7,484
- μ = 0.1: |ΔC| = 2,618 (minimum!)

**Intermediate-to-high friction regime (μ ≥ 0.15):**
- Large topological change
- Increases with μ
- μ = 1.0: |ΔC| = 17,118 (maximum!)

#### 3. **Sign Reversals**

Most systems undergo **chirality sign reversal** (C crosses zero):
- μ = 0.15: +4,570 → -5,084 ✓
- μ = 0.2: +5,680 → -8,120 ✓
- μ = 0.4: +6,004 → -8,840 ✓
- μ = 1.0: +13,970 → -3,148 ✓

Only μ = 0.05 and μ = 0.1 maintain their initial chirality sign.

#### 4. **High Friction Paradox**

**Surprising result**: The highest friction (μ = 1.0) shows the **MOST** topological change!

This contradicts the simple hypothesis that "friction stabilizes topology." Instead:

**Possible explanations:**
1. **Different initial conditions**: Each μ starts from a different C_initial
2. **Timescale effects**: High friction may lead to longer trajectories or different dynamics
3. **Metastable states**: Different μ values may trap the system in different topological basins
4. **Simulation artifacts**: High friction might cause numerical issues leading to unphysical overlaps

### Physical Interpretation

#### Why is μ = 0.1 Special?

The **minimum topological change at μ = 0.1** suggests this friction coefficient represents an optimal balance:

- **Too low friction (μ < 0.1)**: System has enough kinetic energy to overcome topological barriers
- **Optimal friction (μ ≈ 0.1)**: System settles into topologically stable configuration quickly
- **Too high friction (μ > 0.1)**: System may be driven by external forces or boundary effects that cause large rearrangements

#### Recommendations for Future Analysis

1. **Check initial conditions**: Are all simulations starting from the same configuration?
2. **Examine trajectories**: Plot C(t) for each μ to see temporal evolution
3. **Compare timescales**: Do different μ values have different simulation durations?
4. **Verify physics**: Check for numerical artifacts at high μ

### Scalar Measure Validation

**Total chirality C** successfully serves as a scalar measure of topological constraint:
- Clear quantitative differences between friction coefficients
- Sensitive to topological rearrangements
- Easy to compute and interpret

However, the **non-monotonic relationship** suggests that friction's role in topology is more complex than simple "stabilization."

## Files Generated

- `topology_vs_friction.csv` - Complete numerical results
- `topology_vs_friction.png` - Visualization of |ΔC| vs μ

![Topology vs Friction](file:///Users/yeonsu/GitHub/rod-dynamics-3d/study/topology_vs_friction.png)

## Next Steps

To understand the unexpected μ = 1.0 behavior:
1. Analyze full C(t) trajectory for each μ
2. Compare simulation parameters (duration, forces, etc.)
3. Check if initial configurations are truly identical
4. Investigate whether high-μ simulations have different physics
