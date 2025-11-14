#!/usr/bin/env python3
"""
Generate scene JSON files for pairwise metrics study.
Sweep: n ∈ {6, 7, 8}, AR ∈ {50, 150, 500}
"""
import json
import numpy as np
import os


def create_pairwise_scene(n_rods, aspect_ratio, output_path):
    """
    Create a scene with n_rods of given aspect ratio in periodic box.
    
    Args:
        n_rods: Number of rods
        aspect_ratio: Length/diameter ratio
        output_path: Path to save JSON
    """
    # Rod dimensions
    diameter = 0.05
    length = aspect_ratio * diameter
    half_length = length / 2.0
    
    # Periodic box size - scale with number of rods and aspect ratio
    # Ensure box is large enough to avoid crowding
    # Volume per rod: ~L^3 / n_rods
    # Rod volume: π(D/2)^2 * L
    # Use packing fraction ~0.1-0.2 for dilute system
    
    rod_volume = np.pi * (diameter/2)**2 * length
    total_rod_volume = n_rods * rod_volume
    packing_fraction = 0.15  # Dilute system
    box_volume = total_rod_volume / packing_fraction
    box_size = box_volume**(1/3)
    
    # Make box cubic for simplicity
    box_min = [-box_size/2, -box_size/2, -box_size/2]
    box_max = [box_size/2, box_size/2, box_size/2]
    
    # Generate random positions and orientations
    np.random.seed(42 + n_rods + int(aspect_ratio))  # Reproducible but different per config
    
    bodies = []
    for i in range(n_rods):
        # Random position in box
        pos = np.random.uniform(box_min, box_max).tolist()
        
        # Random orientation (uniform on sphere)
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        axis = [
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ]
        
        # Random initial velocity (small)
        vel = np.random.uniform(-0.1, 0.1, 3).tolist()
        
        # Random angular velocity (small)
        ang_vel = np.random.uniform(-0.5, 0.5, 3).tolist()
        
        body = {
            "type": "capsule",
            "radius": diameter / 2,
            "halfLength": half_length,
            "mass": 1.0,
            "position": pos,
            "rotAxis": axis,
            "rotAngle": 0.0,
            "velocity": vel,
            "angularVelocity": ang_vel
        }
        bodies.append(body)
    
    # Scene configuration
    scene = {
        "bodies": bodies,
        "physics": {
            "gravity": [0, 0, 0],  # No gravity for cleaner dynamics
            "dt": 1.0/600.0,
            "solverIterations": 20,
            "linDamp": 0.0,  # No damping
            "angDamp": 0.0,
            "friction": 0.0,  # No friction for initial study
            "restitution": 1.0,  # Elastic collisions
            "floor": False,
            "randomInit": False,
            "useRandomForce": False  # Start without noise
        },
        "periodic": {
            "enabled": True,
            "min": box_min,
            "max": box_max
        }
    }
    
    # Write JSON
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(scene, f, indent=2)
    
    print(f"Created {output_path}: n={n_rods}, AR={aspect_ratio}, box={box_size:.2f}")


def main():
    n_values = [6, 7, 8]
    ar_values = [50, 150, 500]
    
    scene_dir = "../assets/scenes/pairwise_sweep"
    os.makedirs(scene_dir, exist_ok=True)
    
    for n in n_values:
        for ar in ar_values:
            filename = f"pairwise_n{n}_ar{ar}.json"
            path = os.path.join(scene_dir, filename)
            create_pairwise_scene(n, ar, path)
    
    print(f"\nGenerated {len(n_values) * len(ar_values)} scene files in {scene_dir}/")
    print("\nTo run simulations:")
    print("  cd build")
    print("  for scene in ../assets/scenes/pairwise_sweep/*.json; do")
    print("    name=$(basename $scene .json)")
    print("    ./rigidbody_viewer_3d --scene $scene --headless 5000 --perrod ../parametric_study/pairwise_sweep_csvs/${name}.csv --perrod-max 1000")
    print("  done")


if __name__ == "__main__":
    main()
