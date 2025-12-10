import json
import subprocess
import time
import shutil
import csv
import math
from pathlib import Path

# Configuration
SCENE_PATH = Path("20251209-003150_ar100_N27000_fSig1.0e-04/scene.json")
BINARY_PATH = Path("build-headless/rigidbody_viewer_3d")
STEPS = 100
OUTPUT_DIR = Path("benchmark_results")
RODS_COUNT = 2700 # Explicitly checking for this N

def run_sim(scene_path, name):
    output_csv = OUTPUT_DIR / f"output_{name}.csv"
    perrod_csv = OUTPUT_DIR / f"perrod_{name}.csv"
    
    cmd = [
        str(BINARY_PATH),
        "--scene", str(scene_path),
        "--headless",
        "--steps", str(STEPS),
        "--output", str(output_csv),
        "--perrod", str(perrod_csv),
        "--perrod-max", str(STEPS) # Limit frames
    ]
    
    print(f"Running {name}...")
    start_t = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end_t = time.time()
    duration = end_t - start_t
    print(f"  Duration: {duration:.4f}s")
    return duration, perrod_csv

def compare_results(file1, file2):
    print("Comparing results...")
    # Load last frame positions
    def load_last_frame(path):
        data = {} # rod_id -> (x, y, z)
        last_frame = -1
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row['frame'])
                if frame > last_frame:
                    last_frame = frame
                # Always update to latest available position for this rod
                rid = int(row['rod'])
                data[rid] = (float(row['px']), float(row['py']), float(row['pz']))
        return data, last_frame

    data1, frame1 = load_last_frame(file1)
    data2, frame2 = load_last_frame(file2)
    
    if frame1 != frame2:
        print(f"Warning: Frames differ (Hash={frame1}, Naive={frame2})")
    
    diffs = []
    
    for rid, pos1 in data1.items():
        if rid in data2:
            pos2 = data2[rid]
            dist_sq = sum((p1 - p2)**2 for p1, p2 in zip(pos1, pos2))
            diffs.append(dist_sq)
            
    if not diffs:
        return 0.0
        
    rmse = math.sqrt(sum(diffs) / len(diffs))
    return rmse

def main():
    if not BINARY_PATH.exists():
        print(f"Error: Binary {BINARY_PATH} not found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load base scene
    with open(SCENE_PATH) as f:
        base_scene = json.load(f)
        
    # Ensure correct N if possible by not changing anything else
    # Setup HASH scene
    scene_hash = base_scene.copy()
    scene_hash["physics"]["soft_contact"]["use_spatial_hash"] = True
    scene_hash["physics"]["soft_contact"]["cell_size"] = 1.2
    
    path_hash = OUTPUT_DIR / "scene_hash.json"
    with open(path_hash, "w") as f:
        json.dump(scene_hash, f, indent=4)
        
    # Setup NAIVE scene
    scene_naive = base_scene.copy()
    scene_naive["physics"]["soft_contact"]["use_spatial_hash"] = False
    
    path_naive = OUTPUT_DIR / "scene_naive.json"
    with open(path_naive, "w") as f:
        json.dump(scene_naive, f, indent=4)
        
    # Run
    t_hash, out_hash = run_sim(path_hash, "hash")
    t_naive, out_naive = run_sim(path_naive, "naive")
    
    # Compare
    rmse = compare_results(out_hash, out_naive)
    
    print("\n--- Benchmark Summary ---")
    print(f"Steps: {STEPS}")
    print(f"Spatial Hash Time: {t_hash:.4f}s")
    print(f"Naive Time:        {t_naive:.4f}s")
    print(f"Speedup:           {t_naive / t_hash:.2f}x")
    print(f"RMSE (Pos diff):   {rmse:.6e}")
    
    if rmse < 1e-4:
        print("Result: MATCH (within tolerance)")
    else:
        print("Result: DIFFERENCE DETECTED (check logic)")

if __name__ == "__main__":
    main()
