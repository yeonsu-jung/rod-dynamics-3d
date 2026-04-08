import sys
sys.path.insert(0, "build")

import rod_dynamics_py

sim = rod_dynamics_py.Simulator(
    "assets/scenes/default_entangled.json",
    "initial-configs/relaxation_3rd_multithreading/N200/945,12,381/x_relaxed_AR200.txt",
)

print(sim.diagnostics())
sim.step()
print(sim.diagnostics())
print(sim.rods()[0]["position"])