import os
import subprocess
import shutil

N = 200
AR = 50
key = "199,97,131"
base_path = "/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization-cpp/examples/relaxation_3rd_multithreading"
csv_path = os.path.join(base_path, f"N{N}", key, f"x_relaxed_AR{AR}.txt")
renderer = "./build/rigidbody_viewer_3d"
visualizer = "visualize_entangled.py"
output_base = "/Users/yeonsu/GitHub/rod-dynamics-3d/renderings_relaxed"

# 1. Custom Renderings
scales = ["1x", "1.5x", "2x"]
for scale in scales:
    scale_dir = os.path.join(output_base, scale)
    os.makedirs(scale_dir, exist_ok=True)
    scene = f"assets/scenes/default_still_{scale}.json"
    
    tmp_dump = f"tmp_relaxed_N{N}_AR{AR}_{scale}"
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
        "--rod-diameter", "0.02", # Physical diameter for AR=50
        "--auto-exit",
        "--cam-pos", "-1", "1", "-1",
        "--no-floor"
    ]
    subprocess.run(cmd, capture_output=True)
    
    src = os.path.join(tmp_dump, "frame_00000.png")
    dst = os.path.join(scale_dir, f"N{N}_AR{AR}.png")
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Custom Scale {scale} -> {dst}")
    shutil.rmtree(tmp_dump)

# 2. Polyscope Rendering
ps_output = os.path.join(output_base, "polyscope")
os.makedirs(ps_output, exist_ok=True)
screenshot_path = os.path.join(ps_output, f"N{N}_AR{AR}.png")

cmd = [
    "python3", visualizer,
    csv_path,
    "--diameter", "0.02",
    "--screenshot", screenshot_path
]
subprocess.run(cmd, capture_output=True)
print(f"Polyscope -> {screenshot_path}")
