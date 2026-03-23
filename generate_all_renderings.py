import os
import subprocess
import shutil

N_values = [10, 15, 20, 30, 50, 100, 200, 500, 1000]
scales = ["1x", "1.5x", "2x"]
base_path = "/Users/yeonsu/Downloads/entangled_packings_2nd"
renderer = "./build/rigidbody_viewer_3d"
output_base = "/Users/yeonsu/GitHub/rod-dynamics-3d/renderings"

os.makedirs(output_base, exist_ok=True)

# Arbitrary keys
keys = {
    10: "278,868,121",
    15: "122,53,87",
    20: "140,437,328",
    30: "250,122,163",
    50: "312,174,828",
    100: "180,271,742",
    200: "199,97,131",
    500: "20,909,910",
    1000: "117,696,524"
}

for scale in scales:
    scale_dir = os.path.join(output_base, scale)
    os.makedirs(scale_dir, exist_ok=True)
    scene = f"assets/scenes/default_still_{scale}.json"
    
    print(f"--- Rendering for Scale: {scale} ---")
    for N in N_values:
        key = keys[N]
        csv_path = os.path.join(base_path, f"N{N}", key, "x_entangled.txt")
        if not os.path.exists(csv_path):
            print(f"Skipping N={N}, file not found: {csv_path}")
            continue
        
        tmp_dump = f"tmp_dump_N{N}_{scale}"
        if os.path.exists(tmp_dump):
            shutil.rmtree(tmp_dump)
        os.makedirs(tmp_dump)
        
        cmd = [
            renderer,
            "--scene", scene,
            "--playback", csv_path,
            "--export", tmp_dump,
            "--no-label",
            "--white-bg",
            "--auto-exit",
            "--cam-pos", "-1", "1", "-1",
            "--no-floor"
        ]
        
        subprocess.run(cmd, capture_output=True)
        
        src = os.path.join(tmp_dump, "frame_00000.png")
        dst = os.path.join(scale_dir, f"N{N}.png")
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  N={N} -> {dst}")
        else:
            print(f"  Failed N={N} (Scale={scale})")
        
        shutil.rmtree(tmp_dump)

print("All multi-scale renderings complete.")
