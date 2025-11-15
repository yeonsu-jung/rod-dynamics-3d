#!/usr/bin/env python3
"""
Analyze swept cone overlap for choreographed rod rotations.

Key insight: If rotation axis ω is fixed and misaligned with rod axis u,
the rod sweeps a cone. Multiple rods can share overlapping cone volumes
if they're phase-shifted in rotation angle (temporal separation).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cone_angle(rod_axis, rotation_axis):
    """
    Compute half-angle of swept cone when rod rotates around rotation_axis.
    
    Args:
        rod_axis: (3,) unit vector along rod
        rotation_axis: (3,) unit vector for angular velocity direction
    
    Returns:
        theta: half-angle of cone in radians
    """
    cos_theta = np.dot(rod_axis, rotation_axis)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return theta


def rotation_matrix_axis_angle(axis, angle):
    """Rodrigues' rotation formula"""
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    x, y, z = axis
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ])


def rod_endpoints_rotating(center, initial_orientation, rotation_axis, angle_phase, length):
    """
    Compute rod endpoints at a specific phase angle.
    
    Args:
        center: (3,) centroid position
        initial_orientation: (3,) initial rod direction (at phase=0)
        rotation_axis: (3,) axis of rotation
        angle_phase: current rotation phase in radians
        length: rod length
    
    Returns:
        p1, p2: endpoints
    """
    R = rotation_matrix_axis_angle(rotation_axis, angle_phase)
    current_orientation = R @ initial_orientation
    half_len = length / 2.0
    p1 = center - half_len * current_orientation
    p2 = center + half_len * current_orientation
    return p1, p2


def check_collision_at_phases(centers, initial_orientations, rotation_axis, phases, length, diameter):
    """
    Check if rods collide at given phase angles.
    
    Args:
        centers: (N, 3) centroid positions
        initial_orientations: (N, 3) initial rod directions
        rotation_axis: (3,) common rotation axis
        phases: (N,) array of current phase angles for each rod
        length, diameter: rod geometry
    
    Returns:
        min_distance: minimum rod-to-rod distance across all pairs
    """
    N = len(centers)
    
    # Compute current orientations
    current_orientations = []
    for i in range(N):
        R = rotation_matrix_axis_angle(rotation_axis, phases[i])
        current_orientations.append(R @ initial_orientations[i])
    
    # Compute pairwise distances (simplified: endpoint distances)
    min_dist = np.inf
    for i in range(N):
        p1_i, p2_i = rod_endpoints_rotating(centers[i], initial_orientations[i], 
                                            rotation_axis, phases[i], length)
        for j in range(i+1, N):
            p1_j, p2_j = rod_endpoints_rotating(centers[j], initial_orientations[j],
                                                rotation_axis, phases[j], length)
            # Quick check: minimum distance between any two endpoints
            for pi in [p1_i, p2_i]:
                for pj in [p1_j, p2_j]:
                    d = np.linalg.norm(pi - pj)
                    min_dist = min(min_dist, d)
    
    return min_dist


def design_choreography(N=3, length=1.0, diameter=0.05, cone_half_angle_deg=30.0):
    """
    Design a choreographed rotation with N rods sharing overlapping cone volumes.
    
    Strategy:
    - All rods rotate around the same axis (e.g., Y-axis)
    - All have the same cone half-angle
    - Phase-shift them by 2π/N to avoid collisions
    - Choose initial orientations to maximize spatial overlap while maintaining safety
    
    Returns:
        scene_config: dict with positions, orientations, angular velocities
    """
    rotation_axis = np.array([0.0, 1.0, 0.0])  # All rotate around Y
    
    # Design initial orientations to create cone_half_angle with Y-axis
    # Rod initially tilted in X-Z plane
    theta = np.radians(cone_half_angle_deg)
    
    # Place rods at same centroid (or very close) for maximum cone overlap
    centers = np.zeros((N, 3))
    
    # Initial orientations: all start at same angle from Y, but rotated around Y by phase offset
    # Base orientation: tilted theta from Y in the X-Y plane
    base_orientation = np.array([np.sin(theta), np.cos(theta), 0.0])
    
    initial_orientations = []
    phase_offsets = []
    for i in range(N):
        # Phase offset: rotate around Y-axis
        phase = 2 * np.pi * i / N
        phase_offsets.append(phase)
        R = rotation_matrix_axis_angle(rotation_axis, phase)
        u = R @ base_orientation
        initial_orientations.append(u / np.linalg.norm(u))
    
    initial_orientations = np.array(initial_orientations)
    phase_offsets = np.array(phase_offsets)
    
    # Angular velocities: same magnitude, same axis, different initial phases
    omega_mag = 2.0  # rad/s
    angular_velocities = np.tile(rotation_axis * omega_mag, (N, 1))
    
    # Verify no collisions at initial phase configuration (t=0)
    # At t=0, phase angles are just the phase_offsets
    min_dist = check_collision_at_phases(centers, initial_orientations, rotation_axis,
                                        np.zeros(N), length, diameter)  # phases are baked into initial orientations
    
    print(f"Minimum distance at initial phases: {min_dist:.4f}")
    print(f"Rod diameter: {diameter}")
    print(f"Safety margin: {min_dist - diameter:.4f}")
    
    # Sample throughout one full rotation to find worst-case separation
    print("\nSampling collision distances over full rotation...")
    t_samples = np.linspace(0, 2*np.pi/omega_mag, 100)
    min_distances = []
    for t in t_samples:
        # All rods rotate with same omega, so relative phase stays constant
        phases_t = omega_mag * t * np.ones(N)  # same rotation for all
        min_dist_t = check_collision_at_phases(centers, initial_orientations, rotation_axis,
                                               phases_t, length, diameter)
        min_distances.append(min_dist_t)
    
    worst_case_dist = np.min(min_distances)
    print(f"Worst-case minimum distance: {worst_case_dist:.4f}")
    print(f"Clearance: {worst_case_dist - diameter:.4f}")
    
    # Plot minimum distance over time
    plt.figure(figsize=(10, 5))
    plt.plot(t_samples, min_distances, 'b-', linewidth=2)
    plt.axhline(y=diameter, color='r', linestyle='--', label=f'Diameter = {diameter}')
    plt.xlabel('Time (s)')
    plt.ylabel('Min Rod-to-Rod Distance')
    plt.title('Minimum Distance Between Rods Over One Rotation Period')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/Users/yeonsu/GitHub/rod-dynamics-3d/parametric_study/cone_choreography_distances.png', dpi=150)
    print(f"\nSaved plot: cone_choreography_distances.png")
    
    # Return scene configuration
    scene_config = {
        'centers': centers,
        'initial_orientations': initial_orientations,
        'angular_velocities': angular_velocities,
        'phase_offsets': phase_offsets,
        'rotation_axis': rotation_axis,
        'cone_half_angle_deg': cone_half_angle_deg
    }
    
    return scene_config


def generate_scene_json(config, length=1.0, diameter=0.05):
    """Generate JSON snippet for three_rods_rotating.json"""
    N = len(config['centers'])
    
    print("\nJSON bodies for three_rods_rotating.json:")
    print("-" * 60)
    
    for i in range(N):
        center = config['centers'][i]
        orientation = config['initial_orientations'][i]
        ang_vel = config['angular_velocities'][i]
        
        # Convert orientation to rot_axis and rot_deg
        # Initial base is [0, 1, 0], need rotation to map it to orientation
        base = np.array([0, 1, 0])
        
        # Rotation axis is perpendicular to both base and orientation
        rot_axis = np.cross(base, orientation)
        rot_axis_norm = np.linalg.norm(rot_axis)
        
        if rot_axis_norm < 1e-6:
            # Already aligned or opposite
            if np.dot(base, orientation) > 0:
                rot_axis = [1, 0, 0]
                rot_deg = 0.0
            else:
                rot_axis = [1, 0, 0]
                rot_deg = 180.0
        else:
            rot_axis = rot_axis / rot_axis_norm
            cos_angle = np.dot(base, orientation)
            rot_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        print(f"  {{")
        print(f'    "pos": [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}],')
        print(f'    "rot_axis": [{rot_axis[0]:.6f}, {rot_axis[1]:.6f}, {rot_axis[2]:.6f}],')
        print(f'    "rot_deg": {rot_deg:.6f},')
        print(f'    "length": {length},')
        print(f'    "diameter": {diameter},')
        print(f'    "density": 1000.0,')
        print(f'    "restitution": 1.0,')
        print(f'    "friction": 0.0,')
        print(f'    "friction_s": 0.0,')
        print(f'    "friction_d": 0.0,')
        print(f'    "v_lin": [0.0, 0.0, 0.0],')
        print(f'    "v_ang": [{ang_vel[0]:.6f}, {ang_vel[1]:.6f}, {ang_vel[2]:.6f}]')
        print(f"  }}{'' if i == N-1 else ','}")


def main():
    print("=" * 60)
    print("Choreographed Rod Rotation with Cone Overlap")
    print("=" * 60)
    print()
    
    config = design_choreography(N=3, length=1.0, diameter=0.05, cone_half_angle_deg=30.0)
    
    print(f"\nCone half-angle: {config['cone_half_angle_deg']}°")
    print(f"Rotation axis: {config['rotation_axis']}")
    print(f"Number of rods: {len(config['centers'])}")
    print(f"Phase offsets (deg): {np.degrees(config['phase_offsets'])}")
    
    generate_scene_json(config, length=1.0, diameter=0.05)


if __name__ == "__main__":
    main()
