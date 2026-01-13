# Reproducing Topological Analysis Plots

This document lists the Python scripts used to generate the figures for the friction topology study.

## 1. Topological Stability vs Friction (N=200)

**Script**: `study/plot_delta_C_vs_mu.py`
*   **Input**: `study/topology_analysis_data/endpoints_formatted_n200_ar300_mu*.csv`
*   **Output**: `study/delta_C_vs_mu.png`
*   **Description**: Plots the magnitude of Total Chirality Change ($|\Delta C|$) vs Friction Coefficient ($\mu$). Shows that friction stabilizes topology.

## 2. Stable Core & Pair Distances (N=200)

**Script**: `study/plot_stable_core_vs_mu.py`
*   **Input**: `study/topology_analysis_data/endpoints_formatted_n200_ar300_mu*.csv`
*   **Outputs**:
    *   `study/stable_core_vs_mu.png`: Number of rods in the stable core vs $\mu$.
    *   `study/stable_core_avg_dist_vs_mu.png`: Average pairwise distance of rods in the stable core vs $\mu$.
    *   `study/vorticity_changes_vs_mu.png`: Total number of changed vorticity triples vs $\mu$.
*   **Description**: Identifies the transition from liquid-like (unstable) to solid-like (stable) behavior around $\mu=0.2$.

## 3. Large Scale Analysis (N=1000, New Simulator)

**Script**: `study/plot_stable_core_vs_mu_new.py`
*   **Input**: `scripts/relax3rd_N1000_sweep/*`
*   **Outputs**:
    *   `study/stable_core_vs_mu_new.png`: Stable Core Size vs $\mu$ for N=1000.
    *   `study/stable_core_avg_dist_vs_mu_new.png`: Packing density of stable core for N=1000.
*   **Description**: Confirms the phase transition occurs earlier ($\mu \approx 0.1$) in the larger, longer simulation.

## 4. Filtered Analysis (N=1000, < 5° Filter)

**Script**: `study/plot_stable_core_vs_mu_filtered.py`
*   **Input**: `scripts/relax3rd_N1000_sweep/*`
*   **Outputs**:
    *   `study/stable_core_vs_mu_filtered.png`: Overlay of standard and filtered stable core sizes.
*   **Description**: Filters out triples containing nearly parallel rods (angle < 5°) to remove geometric noise. Shows that the stable core is larger (~71%) than estimated by the raw analysis.

## Summary of Analysis Flow
1.  **Compute Invariants**: Core logic in `study/compute_topology.py` and `study/find_stable_core.py`.
2.  **Batch Processing**: The scripts above batch process all friction directories.
3.  **Visualization**: Matplotlib is used to generate the final comparision figures.
