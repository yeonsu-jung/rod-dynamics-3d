import argparse
import subprocess
import json
import shutil
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import csv

# Configuration
MU_VALUES = [0.0, 0.1, 0.2, 0.4]
BASE_SCENE_PATH = Path("assets/scenes/experiment_bigger_mu_0.2.json")
BINARY_PATH = Path("build/rigidbody_viewer_3d")  # Default, can be overridden
OUTPUT_DIR = Path("sweep_mu_results")

def find_binary():
    """Locate the rigidbody_viewer_3d executable."""
    candidates = [
        Path("build/rigidbody_viewer_3d"),
        Path("build-headless/rigidbody_viewer_3d"),
        Path("./rigidbody_viewer_3d"),
        Path("../build/rigidbody_viewer_3d"),
        Path("../build-headless/rigidbody_viewer_3d")
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c.resolve()
    return None

def load_csv(path: Path):
    """Load the output CSV into a dictionary of lists."""
    data = {
        "frame": [], "KE": [], "contacts": [], "max_overlap": [], 
        "reldisp_sq": [], "gyration_sq": []
    }
    if not path.exists():
        print(f"Warning: {path} not found.")
        return data

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["frame"].append(int(row["frame"]))
            data["KE"].append(float(row["KE"]))
            data["contacts"].append(float(row["contacts"]))
            data["max_overlap"].append(float(row["max_overlap"]))
            
            # Handle variations in column names
            if "reldisp_sq" in row:
                data["reldisp_sq"].append(float(row["reldisp_sq"]))
            elif "reldisp" in row:
                 data["reldisp_sq"].append(float(row["reldisp"]))
            else:
                 data["reldisp_sq"].append(0.0)
                 
            if "gyration_sq" in row:
                data["gyration_sq"].append(float(row["gyration_sq"]))
            elif "reldisp_norm" in row: # Old compat
                data["gyration_sq"].append(float(row["reldisp_norm"]))
            else:
                data["gyration_sq"].append(0.0)
    return data

def run_sweep(binary_path, steps):
    """Run the simulation for each mu value."""
    if not binary_path.exists():
        print(f"Error: Binary {binary_path} not found.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load base scene
    if not BASE_SCENE_PATH.exists():
        print(f"Error: Base scene {BASE_SCENE_PATH} not found. Run from project root.")
        sys.exit(1)
        
    with BASE_SCENE_PATH.open() as f:
        base_scene = json.load(f)

    results = {}

    for mu in MU_VALUES:
        print(f"--- Running for mu={mu} ---")
        
        # Modify scene
        scene = base_scene.copy()
        scene["physics"]["soft_contact"]["mu"] = mu
        
        # Save temp scene
        scene_path = OUTPUT_DIR / f"scene_mu_{mu}.json"
        with scene_path.open("w") as f:
            json.dump(scene, f, indent=4)
            
        # Define output paths
        output_csv = OUTPUT_DIR / f"output_mu_{mu}.csv"
        perrod_csv = OUTPUT_DIR / f"perrod_mu_{mu}.csv"
        
        # Run simulation
        cmd = [
            str(binary_path),
            "--scene", str(scene_path),
            "--headless",
            "--steps", str(steps),
            "--output", str(output_csv),
            "--perrod", str(perrod_csv)
            # "--profile" # Optional
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running simulation for mu={mu}: {e}")
            continue
            
        results[mu] = load_csv(output_csv)

    return results

def plot_results(results):
    """Generate overlaid plots."""
    print("Generating plots...")
    
    metrics = [
        ("KE", "Kinetic Energy"),
        ("contacts", "Contact Count"),
        ("max_overlap", "Max Overlap"),
        ("reldisp_sq", "Relative Displacement Sq"),
        ("gyration_sq", "Gyration Sq")
    ]
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3*len(metrics)), sharex=True)
    if len(metrics) == 1: axes = [axes]
    
    for i, (key, label) in enumerate(metrics):
        ax = axes[i]
        for mu, data in results.items():
            if not data["frame"]: continue
            ax.plot(data["frame"], data[key], label=f"mu={mu}")
        
        ax.set_ylabel(label)
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "sweep_comparison.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep mu parameter")
    parser.add_argument("--steps", type=int, default=10000, help="Simulation steps")
    parser.add_argument("--binary", type=str, help="Path to executable")
    args = parser.parse_args()
    
    # Locate binary
    bin_path = Path(args.binary) if args.binary else find_binary()
    if not bin_path:
        # Fallback check relative to script location if run from inside study/
        script_dir = Path(__file__).parent
        candidates = [
            script_dir.parent / "build/rigidbody_viewer_3d",
            script_dir.parent / "build-headless/rigidbody_viewer_3d"
        ]
        for c in candidates:
             if c.is_file() and os.access(c, os.X_OK):
                bin_path = c.resolve()
                break
                
    if not bin_path or not bin_path.exists():
        print("Error: Could not find rigidbody_viewer_3d binary.")
        print("Build it first or provide path with --binary")
        sys.exit(1)
        
    print(f"Using binary: {bin_path}")
    
    results = run_sweep(bin_path, args.steps)
    plot_results(results)
