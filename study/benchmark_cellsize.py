import json
import subprocess
import time
import shutil
import csv
from pathlib import Path

# Configuration
SCENE_PATH = Path("20251209-003150_ar100_N27000_fSig1.0e-04/scene.json")
BINARY_PATH = Path("build-headless/rigidbody_viewer_3d")
STEPS = 100
OUTPUT_DIR = Path("benchmark_results_cellsize")
CELL_SIZES = [0.8, 1.0, 1.2, 1.5, 3.0]

def run_sim(scene_path, name):
    output_csv = OUTPUT_DIR / f"output_{name}.csv"
    perrod_csv = OUTPUT_DIR / f"perrod_{name}.csv"
    
    cmd = [
        str(BINARY_PATH),
        "--scene", str(scene_path),
        "--headless",
        "--steps", str(STEPS),
        "--output", str(output_csv),
        # disable heavy logging for speed
        # "--perrod", str(perrod_csv) 
    ]
    
    print(f"Running {name}...")
    start_t = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end_t = time.time()
    duration = end_t - start_t
    print(f"  Duration: {duration:.4f}s")
    return duration

def main():
    if not BINARY_PATH.exists():
        print(f"Error: Binary {BINARY_PATH} not found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load base scene
    with open(SCENE_PATH) as f:
        base_scene = json.load(f)
        
    results = {}
    
    for cs in CELL_SIZES:
        name = f"cell_{cs}"
        scene = base_scene.copy()
        scene["physics"]["soft_contact"]["use_spatial_hash"] = True
        scene["physics"]["soft_contact"]["cell_size"] = cs
        
        path = OUTPUT_DIR / f"scene_{name}.json"
        with open(path, "w") as f:
            json.dump(scene, f, indent=4)
            
        dur = run_sim(path, name)
        results[cs] = dur
        
    print("\n--- CellSize Benchmark Summary (N=5400, 100 steps) ---")
    best_cs = None
    best_time = float('inf')
    
    print(f"{'CellSize':<10} | {'Time (s)':<10}")
    print("-" * 25)
    for cs in CELL_SIZES:
        t = results[cs]
        print(f"{cs:<10.1f} | {t:<10.4f}")
        if t < best_time:
            best_time = t
            best_cs = cs
            
    print(f"\nBest CellSize: {best_cs} (Time: {best_time:.4f}s)")

if __name__ == "__main__":
    main()
