# Topological Evolution Analysis Results

## Key Finding: **Topology is NOT Preserved**

Analysis of frames 0, 1, and 106 (out of 107 total frames) reveals that the topological invariants **change over time**, indicating that the rod configuration is **not isotopic** across the trajectory.

## Results Summary

| Frame | Total Chirality C | Per-Rod Range | Change in C |
|-------|-------------------|---------------|-------------|
| 0     | -8,704           | 3,420         | —           |
| 1     | -8,504           | 3,324         | **+200**    |
| 106   | -7,676           | 3,500         | **+828**    |

### Total Change
- **ΔC (frame 0 → 106)**: +1,028 (11.8% change)
- **Variance in C**: 198,041
- **Max single-step change**: +828 (between frames 1 and 106)

## Interpretation

### What This Means

The changing total chirality C indicates that **rods are passing through each other** during the simulation. This is because:

1. **Total chirality is a topological invariant**: For configurations related by continuous deformation (isotopy) without rod intersections, C must remain constant.

2. **C changes significantly**: The +1,028 change represents a fundamental alteration in the topological structure of the packing.

3. **Possible causes**:
   - **Soft-core interactions**: If the simulation uses soft repulsion instead of hard-core exclusion, rods can overlap temporarily
   - **Numerical integration errors**: Time-stepping may allow small overlaps that accumulate
   - **Periodic boundary conditions**: Rods crossing periodic boundaries can change topology
   - **Intentional dynamics**: The simulation may be designed to allow topological rearrangements

### Scalar Measures of Topological Constraint

Two key scalar measures track topological evolution:

1. **Total Chirality C(t)**
   - **Most sensitive** topological invariant
   - Detects any change in triple-wise rod relationships
   - Variance: 198,041 (high variability)

2. **Per-Rod Chirality Range**
   - Measures heterogeneity: max(c_i) - min(c_i)
   - Less sensitive but still shows variation (3,324 → 3,500)
   - Variance: 5,177 (moderate variability)

### Recommendation

If you want to **enforce topological constraints** (prevent rods from passing through each other):

1. Use hard-core exclusion in the simulation
2. Reduce time step to prevent numerical overlap
3. Add explicit overlap detection and rejection
4. Monitor C(t) in real-time as a constraint violation detector

If the current behavior is **intentional** (e.g., studying topological rearrangements):

1. C(t) serves as an excellent **order parameter** for topological transitions
2. Track dC/dt to identify when and where topology changes occur
3. Correlate C changes with energy, stress, or other physical quantities

## Files Generated

- `topology_evolution.csv` - Time series data for all invariants
- `analyze_topology_evolution.py` - Analysis script for arbitrary frame sets

## Usage

To analyze more frames:
```bash
cd /Users/yeonsu/GitHub/rod-dynamics-3d/study
python analyze_topology_evolution.py exampledata_endpoints_formatted_n100_ar300.csv 0 10 20 30 40 50 60 70 80 90 100 106
```

Or analyze all frames (will take time):
```bash
python analyze_topology_evolution.py exampledata_endpoints_formatted_n100_ar300.csv $(seq 0 106)
```
