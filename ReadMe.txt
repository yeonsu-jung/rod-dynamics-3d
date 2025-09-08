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