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