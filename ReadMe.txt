rod-dynamics-3d/
├─ CMakeLists.txt
├─ external/
│  ├─ glfw/              # as you have now (add_subdirectory)
│  └─ glad/              # glad.c + headers
├─ include/
│  ├─ physics/
│  │  ├─ types.hpp           # Contact, SolverConfig, constants
│  │  ├─ shape.hpp           # Box, Capsule, helpers
│  │  ├─ rigid_body.hpp      # RigidBody API (no OpenGL here)
│  │  ├─ collision.hpp       # collideCapsuleCapsule, collideCapsuleFloor, utils
│  │  ├─ solver.hpp          # applyImpulse, positionalCorrection
│  │  └─ integrator.hpp      # integrate (damping, quaternion step)
│  └─ gfx/
│     ├─ shader.hpp          # compile/link, uniform helpers
│     ├─ mesh.hpp            # Mesh, primitive builders
│     ├─ camera.hpp          # orbit camera
│     └─ renderer.hpp        # draw capsule, floor; grid uniforms, lights
├─ src/
│  ├─ physics/
│  │  ├─ rigid_body.cpp
│  │  ├─ collision.cpp
│  │  ├─ solver.cpp
│  │  └─ integrator.cpp
│  ├─ gfx/
│  │  ├─ shader.cpp
│  │  ├─ mesh.cpp            # cube + capped-cylinder generator
│  │  └─ renderer.cpp        # shaders (inline or loaded), draw calls
│  └─ app/
│     └─ main.cpp            # GLFW window, input, fixed-step loop, wiring
└─ assets/
   └─ shaders/
      ├─ basic.vert
      └─ basic.frag



mkdir -p external/nlohmann_json/include/nlohmann
curl -L https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp \
  -o external/nlohmann_json/include/nlohmann/json.hpp






./rigidbody_viewer_3d --scene /Users/yeonsu/GitHub/rod-dynamics-3d/assets/scenes/red_bg_one_rod.json

## Parametric Dissipation Study

The `parametric_study/` folder contains scripts for running parametric studies on rod dissipation rates.

### Setup Python Environment

1. Create a virtual environment (if not already done):
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install required packages:
   ```
   pip install scipy numpy matplotlib pandas
   ```

### Running the Study

Navigate to the `parametric_study/` directory and run the script:
```
cd parametric_study
../.venv/bin/python parametric_dissipation_study.py
```

This will:
- Generate scene files for different aspect ratios.
- Run headless simulations (skips if CSVs already exist).
- Analyze KE decay by fitting exponential functions.
- Produce plots: `ke_decay_with_fits.png` and `decay_exponents.png`.

Note: Simulations require the built executable `../build/rigidbody_viewer_3d`.

## Submitting parametric runs on a SLURM cluster

A helper script is provided: `parametric_study/submit_parametric_runs.py`

- It discovers the repo root `rod-dynamics-3d` automatically.
- For each aspect ratio, it creates a run folder under `<root>/runs/`.
- It copies `build/rigidbody_viewer_3d` into each run folder.
- It generates a per-run `scene.json` from `assets/scenes/dissipation_study_sample.json` and sets populate.count and rod diameter.
- It writes an `Sbatch.sh` that runs the simulation headlessly and generates a quick KE plot.
- It submits via `sbatch`.

Steps:
1) Build headless (once):
   module load cmake
   mkdir -p build && cd build
   cmake .. -DBUILD_HEADLESS=ON
   cmake --build . -j
2) Submit jobs:
   cd parametric_study
   python3 submit_parametric_runs.py

Customize SLURM defaults by editing `SLURM = {...}` in the script.

## Building the Project

### Full Build (with Graphics)
```
mkdir build
cd build
cmake ..
make
```

### Headless Build (without OpenGL, for clusters)
In environments without OpenGL support (e.g., Linux clusters), build with the headless option:
```
mkdir build
cd build
cmake .. -DBUILD_HEADLESS=ON
make
```

The headless build excludes graphics dependencies and can run simulations without rendering.

## Cluster/Headless Notes

- On clusters without OpenGL/display (no X/Wayland), use the headless build. No GLFW/GLAD are required or linked.
- CMake will auto-fetch missing dependencies (glm, nlohmann_json). If your cluster blocks outbound internet, vendor them:
  - glm: set CMAKE_PREFIX_PATH to an installed glm or provide headers in external/glm/include
  - nlohmann_json: place header at external/nlohmann_json/include/nlohmann/json.hpp

### Build on a cluster
1) Load cmake (and a C++17 compiler if needed):
   module load cmake
2) Configure and build headless:
   mkdir -p build
   cd build
   cmake .. -DBUILD_HEADLESS=ON
   cmake --build . -j

### Run headless
- You can pass --headless explicitly, but the HEADLESS build runs headless by default.
  Examples:
  ./rigidbody_viewer_3d --scene ../assets/scenes/pbc_100_rods.json --steps 5000 --csv out.csv
  ./rigidbody_viewer_3d --headless --scene ../assets/scenes/single_rod.json --steps 2000

The parametric study scripts already invoke the executable in headless mode and will work on clusters.