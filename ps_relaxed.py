import os
import subprocess
import shutil

N_values = [10, 15, 20, 30, 50, 100, 200, 500, 1000]
base_path = "/Users/yeonsu/GitHub/entanglement-optimization-combined/entanglement-optimization-cpp/examples/relaxation_3rd_multithreading"
visualizer = "visualize_entangled.py"
output_base = "/Users/yeonsu/GitHub/rod-dynamics-3d/renderings_relaxed/polyscope"

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

for N in N_values:
    key = keys[N]
    csv_path = os.path.join(base_path, f"N{N}", key, "x_relaxed_AR200.txt")
    if not os.path.exists(csv_path):
        print(f"Skipping N={N}, file not found: {csv_path}")
        continue
    
    screenshot_name = f"N{N}.png"
    screenshot_path = os.path.join(output_base, screenshot_name)
    
    cmd = [
        "python3", visualizer,
        csv_path,
        "--diameter", "0.01",
        "--screenshot", screenshot_path
    ]
    
    print(f"Polyscope rendering Relaxed N={N} ...")
    subprocess.run(cmd, capture_output=True)
    
    if os.path.exists(screenshot_path):
        print(f"  Saved to {screenshot_path}")
    else:
        print(f"  Failed Polyscope relaxed N={N}")

print("Relaxed Polyscope renderings complete.")
