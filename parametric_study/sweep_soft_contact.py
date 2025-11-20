#!/usr/bin/env python3
"""
Parametric sweep for soft contact model with varying friction and noise amplitudes.
"""

import json
import subprocess
import os
import sys
from pathlib import Path
import itertools
import numpy as np

# Parametric sweep parameters
FRICTION_COEFFS = [0.0, 0.05, 0.1, 0.2, 0.4]
NOISE_AMPLITUDES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
N_RODS = 200
BOX_SIZE = 1.0  # periodic box [-1, 1]^3

# Simulation parameters
DT = 0.001
N_STEPS = 100000
OUTPUT_INTERVAL = 100
SEED_BASE = 12345

# Paths
REPO_ROOT = Path(__file__).parent.parent
EXECUTABLE = REPO_ROOT / "build" / "rigidbody_viewer_3d"
SCENE_DIR = REPO_ROOT / "parametric_study" / "scenes_soft_contact"
OUTPUT_DIR = REPO_ROOT / "parametric_study" / "analysis_soft_contact"

def create_scene_file(friction, noise_amp, output_path):
    """Create a scene JSON file with specified parameters."""
    
    scene = {
        "scene": {
            "bodies": [
                {
                    "length": 1.5,
                    "diameter": 0.05,
                    "density": 1000.0,
                    "restitution": 1.0,
                    "friction": friction,
                    "friction_s": friction,
                    "friction_d": friction
                }
            ],
            "populate": {
                "count": N_RODS,
                "mode": "nonoverlap",
                "spacingMul": 2.0,
                "seed": SEED_BASE,
                "maxAttempts": 100000
            },
            "periodic": {
                "enabled": True,
                "min": [-BOX_SIZE, -BOX_SIZE, -BOX_SIZE],
                "max": [BOX_SIZE, BOX_SIZE, BOX_SIZE],
                "cellSize": 2.0
            },
            "randomInit": {
                "enabled": False,
                "vSigma": 0.1,
                "wSpeed": 0.1,
                "seed": 42
            },
            "randomForce": {
                "enabled": True,
                "fSigma": noise_amp,
                "tauMag": noise_amp,
                "seed": 123
            }
        },
        "physics": {
            "dt": DT,
            "gravity": [0.0, 0.0, 0.0],
            "lin_damp": 0.0,
            "ang_damp": 0.0,
            "soft_contact": {
                "enabled": True,
                "k_scaler": 1e1,
                "delta": 0.0002,
                "nu": 0.1,
                "mu": friction,
                "enable_friction": True,
                "verbose": False
            },
            "solver": {
                "baumgarte": 0.0,
                "allowed_pen": 0.0,
                "velIters": 10,
                "split_impulse": False,
                "split_orient": True,
                "ngsNormalSweeps": 0,
                "ngsHighVThresh": 0.5
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(scene, f, indent=2)
    
    return scene

def run_simulation(scene_path, output_prefix):
    """Run a single simulation with the given scene file."""
    
    cmd = [
        str(EXECUTABLE),
        "--scene", str(scene_path),
        "--steps", str(N_STEPS),
        "--output-interval", str(OUTPUT_INTERVAL),
        "--output-prefix", str(output_prefix),
        "--headless"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with error code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    # Create output directories
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if executable exists
    if not EXECUTABLE.exists():
        print(f"ERROR: Executable not found at {EXECUTABLE}")
        print("Please build the project first: cd build && cmake -DBUILD_HEADLESS=ON .. && make")
        sys.exit(1)
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(FRICTION_COEFFS, NOISE_AMPLITUDES))
    total_runs = len(param_combinations)
    
    print(f"Starting parametric sweep with {total_runs} runs")
    print(f"Friction coefficients: {FRICTION_COEFFS}")
    print(f"Noise amplitudes: {NOISE_AMPLITUDES}")
    print(f"Number of rods: {N_RODS}")
    print(f"Steps: {N_STEPS}, dt: {DT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 80)
    
    successful_runs = 0
    failed_runs = 0
    
    for i, (friction, noise_amp) in enumerate(param_combinations, 1):
        print(f"\n[{i}/{total_runs}] Running simulation:")
        print(f"  Friction coefficient μ = {friction}")
        print(f"  Noise amplitude σ = {noise_amp:.1e}")
        
        # Create unique identifier for this run
        run_id = f"mu{friction:.2f}_noise{noise_amp:.1e}"
        
        # Create scene file
        scene_path = SCENE_DIR / f"soft_contact_{run_id}.json"
        create_scene_file(friction, noise_amp, scene_path)
        print(f"  Scene file: {scene_path}")
        
        # Set up output prefix
        output_prefix = OUTPUT_DIR / run_id
        print(f"  Output prefix: {output_prefix}")
        
        # Run simulation
        success = run_simulation(scene_path, output_prefix)
        
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("PARAMETRIC SWEEP COMPLETE")
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Results directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    # Create summary file
    summary_path = OUTPUT_DIR / "sweep_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Soft Contact Model Parametric Sweep Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of rods: {N_RODS}\n")
        f.write(f"Periodic box: [{-BOX_SIZE}, {BOX_SIZE}]^3\n")
        f.write(f"Time step: {DT}\n")
        f.write(f"Total steps: {N_STEPS}\n")
        f.write(f"Output interval: {OUTPUT_INTERVAL}\n\n")
        f.write(f"Friction coefficients: {FRICTION_COEFFS}\n")
        f.write(f"Noise amplitudes: {NOISE_AMPLITUDES}\n\n")
        f.write(f"Total runs: {total_runs}\n")
        f.write(f"Successful: {successful_runs}\n")
        f.write(f"Failed: {failed_runs}\n")
    
    print(f"\nSummary written to: {summary_path}")
    
    return 0 if failed_runs == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
