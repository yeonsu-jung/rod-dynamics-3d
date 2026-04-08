from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
sys.path.insert(0, str(BUILD_DIR))

import rod_dynamics_py


scene_path = REPO_ROOT / "assets" / "scenes" / "default_entangled.json"
init_csv_path = REPO_ROOT / "initial-configs" / "relaxation_3rd_multithreading" / "N200" / "945,12,381" / "x_relaxed_AR200.txt"

sim = rod_dynamics_py.Simulator(str(scene_path), str(init_csv_path))

print("before", sim.diagnostics())
sim.step()
print("after", sim.diagnostics())

first_rod = sim.rods()[0]
print("first rod position", first_rod["position"])