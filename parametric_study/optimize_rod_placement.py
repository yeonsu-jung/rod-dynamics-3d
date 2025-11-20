#!/usr/bin/env python3
"""
Optimize placement of N tumbling rods to minimize spread while avoiding collisions.

Objective: Minimize the maximum pairwise centroid distance
Constraint: Minimum rod-to-rod distance >= safety_margin

For capsules (rods), the distance between two rods is computed as the
segment-segment distance between their centerlines.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import json

def segment_segment_distance(p1, p2, q1, q2):
    """
    Compute minimum distance between two line segments in 3D.
    p1, p2: endpoints of segment 1
    q1, q2: endpoints of segment 2
    Returns: minimum distance
    """
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    
    D = a * c - b * b
    sc, tc = 0.0, 0.0
    
    # Compute line parameters of closest approach
    if D < 1e-8:  # parallel
        sc = 0.0
        tc = e / c if c > 1e-8 else 0.0
    else:
        sc = (b * e - c * d) / D
        tc = (a * e - b * d) / D
    
    # Clamp to [0, 1]
    sc = np.clip(sc, 0.0, 1.0)
    tc = np.clip(tc, 0.0, 1.0)
    
    # Recompute tc for clamped sc (and vice versa if needed)
    if sc == 0.0 or sc == 1.0:
        tc = np.clip((e + b * sc) / c if c > 1e-8 else 0.0, 0.0, 1.0)
    if tc == 0.0 or tc == 1.0:
        sc = np.clip((-d + b * tc) / a if a > 1e-8 else 0.0, 0.0, 1.0)
    
    closest_p = p1 + sc * u
    closest_q = q1 + tc * v
    
    return np.linalg.norm(closest_p - closest_q)


def rod_endpoints(center, orientation, length):
    """
    Compute rod endpoints given center, orientation (unit vector along rod), and length.
    """
    half_len = length / 2.0
    p1 = center - half_len * orientation
    p2 = center + half_len * orientation
    return p1, p2


def pairwise_rod_distances(centers, orientations, length):
    """
    Compute all pairwise rod-to-rod distances.
    centers: (N, 3) array of rod centers
    orientations: (N, 3) array of unit vectors
    length: scalar rod length
    Returns: (N*(N-1)/2,) array of distances
    """
    N = len(centers)
    distances = []
    for i in range(N):
        p1, p2 = rod_endpoints(centers[i], orientations[i], length)
        for j in range(i + 1, N):
            q1, q2 = rod_endpoints(centers[j], orientations[j], length)
            d = segment_segment_distance(p1, p2, q1, q2)
            distances.append(d)
    return np.array(distances)


def objective_and_constraint(x, N, orientations, length, safety_margin):
    """
    x: flattened [x1,y1,z1, x2,y2,z2, ...] for N rods
    Returns: (objective_value, min_rod_distance - safety_margin)
    """
    centers = x.reshape(N, 3)
    
    # Objective: minimize maximum centroid-to-centroid distance
    # (compact arrangement)
    centroid_dists = cdist(centers, centers)
    max_centroid_dist = np.max(centroid_dists)
    
    # Constraint: minimum rod-to-rod distance >= safety_margin
    rod_dists = pairwise_rod_distances(centers, orientations, length)
    min_rod_dist = np.min(rod_dists) if len(rod_dists) > 0 else np.inf
    
    # Penalty constraint (for penalty method or feasibility check)
    constraint_violation = safety_margin - min_rod_dist
    
    return max_centroid_dist, constraint_violation, min_rod_dist


def optimize_placement(N, orientations, length, diameter, safety_factor=1.5, method='differential_evolution'):
    """
    Optimize rod placement.
    
    Args:
        N: number of rods
        orientations: (N, 3) array of unit orientation vectors
        length: rod length
        diameter: rod diameter
        safety_factor: multiplier on diameter for safety margin
        method: 'differential_evolution' or 'slsqp'
    
    Returns:
        optimized_centers: (N, 3) array
        metrics: dict with objective and constraint values
    """
    safety_margin = diameter * safety_factor
    
    # Normalize orientations
    orientations = np.array(orientations)
    orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)
    
    def objective_func(x):
        obj, violation, min_dist = objective_and_constraint(x, N, orientations, length, safety_margin)
        # Penalize constraint violations heavily
        penalty = 1000.0 * max(0, violation)**2
        return obj + penalty
    
    def constraint_func(x):
        """Constraint: must be >= 0 for feasibility"""
        _, violation, _ = objective_and_constraint(x, N, orientations, length, safety_margin)
        return -violation  # scipy wants g(x) >= 0
    
    # Initial guess: place rods in a rough circle
    initial_radius = length + safety_margin
    x0 = []
    for i in range(N):
        angle = 2 * np.pi * i / N
        x0.extend([initial_radius * np.cos(angle), 0.0, initial_radius * np.sin(angle)])
    x0 = np.array(x0)
    
    if method == 'differential_evolution':
        # Global optimization
        bounds = [(-2.0, 2.0)] * (3 * N)
        result = differential_evolution(
            objective_func,
            bounds,
            seed=42,
            maxiter=1000,
            atol=1e-6,
            tol=1e-6,
            workers=1
        )
        optimized_centers = result.x.reshape(N, 3)
        final_obj, final_violation, final_min_dist = objective_and_constraint(
            result.x, N, orientations, length, safety_margin
        )
    else:
        # Local optimization with constraint
        from scipy.optimize import NonlinearConstraint
        constraint = NonlinearConstraint(constraint_func, 0, np.inf)
        result = minimize(
            objective_func,
            x0,
            method='SLSQP',
            constraints=constraint,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        optimized_centers = result.x.reshape(N, 3)
        final_obj, final_violation, final_min_dist = objective_and_constraint(
            result.x, N, orientations, length, safety_margin
        )
    
    metrics = {
        'objective': final_obj,
        'min_rod_distance': final_min_dist,
        'safety_margin': safety_margin,
        'constraint_violation': final_violation,
        'feasible': final_violation <= 0
    }
    
    return optimized_centers, metrics


def main():
    # Rod parameters from three_rods_rotating.json
    length = 1.0
    diameter = 0.05
    
    # Orientations from rot_axis and rot_deg in the scene
    # Rod 1: rotated 30° around X
    # Rod 2: rotated 45° around Z
    # Rod 3: rotated 60° around (1,1,0) normalized
    
    def rotation_matrix(axis, deg):
        """Compute rotation matrix from axis and angle in degrees."""
        axis = np.array(axis) / np.linalg.norm(axis)
        angle = np.radians(deg)
        c, s = np.cos(angle), np.sin(angle)
        C = 1 - c
        x, y, z = axis
        return np.array([
            [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
        ])
    
    # Initial rod orientation is along Y-axis [0,1,0]
    base_orientation = np.array([0.0, 1.0, 0.0])
    
    orientations = [
        rotation_matrix([1, 0, 0], 30.0) @ base_orientation,
        rotation_matrix([0, 0, 1], 45.0) @ base_orientation,
        rotation_matrix([1, 1, 0], 60.0) @ base_orientation
    ]
    
    N = 3
    
    print("=" * 60)
    print("Rod Placement Optimization")
    print("=" * 60)
    print(f"Number of rods: {N}")
    print(f"Rod length: {length}")
    print(f"Rod diameter: {diameter}")
    print(f"Safety factor: 2.0 (min distance = {2.0 * diameter:.4f})")
    print()
    
    # Run optimization
    print("Running differential evolution optimization...")
    centers, metrics = optimize_placement(N, orientations, length, diameter, safety_factor=2.0)
    
    print("\nOptimization Results:")
    print("-" * 60)
    print(f"Feasible: {metrics['feasible']}")
    print(f"Max centroid-to-centroid distance: {metrics['objective']:.4f}")
    print(f"Min rod-to-rod distance: {metrics['min_rod_distance']:.4f}")
    print(f"Required safety margin: {metrics['safety_margin']:.4f}")
    print(f"Constraint violation: {metrics['constraint_violation']:.4f}")
    print()
    
    print("Optimized positions:")
    for i, center in enumerate(centers):
        print(f"  Rod {i+1}: [{center[0]:7.4f}, {center[1]:7.4f}, {center[2]:7.4f}]")
    print()
    
    # Compute all pairwise centroid distances
    centroid_dists = cdist(centers, centers)
    print("Centroid-to-centroid distances:")
    for i in range(N):
        for j in range(i+1, N):
            print(f"  Rod {i+1} ↔ Rod {j+1}: {centroid_dists[i,j]:.4f}")
    print()
    
    # Compute all pairwise rod distances
    rod_dists = pairwise_rod_distances(centers, np.array(orientations), length)
    idx = 0
    print("Rod-to-rod distances (segment-segment):")
    for i in range(N):
        for j in range(i+1, N):
            print(f"  Rod {i+1} ↔ Rod {j+1}: {rod_dists[idx]:.4f}")
            idx += 1
    print()
    
    # Export updated scene JSON snippet
    print("JSON snippet for three_rods_rotating.json:")
    print("-" * 60)
    for i, center in enumerate(centers):
        print(f'  {{ "pos": [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}], ... }},')
    print()


if __name__ == "__main__":
    main()
