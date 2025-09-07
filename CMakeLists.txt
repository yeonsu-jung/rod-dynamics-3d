cmake_minimum_required(VERSION 3.16)
project(rod_dynamics_3d LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional: nice warnings and sanitizers in Debug
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()
option(BUILD_SANITIZERS "Enable ASAN/UBSAN in Debug builds" ON)
if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND BUILD_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---------- Dependencies ----------
# GLFW (submodule / vendored)
add_subdirectory(external/glfw)

# GLAD (single C file)
add_library(glad STATIC external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)

# GLM (try config package, then headers path fallback)
find_package(glm CONFIG QUIET)       # provides glm::glm if available
if (TARGET glm::glm)
  set(GLM_TARGET glm::glm)
else()
  # Fallback: locate headers (Homebrew path common on Apple Silicon)
  find_path(GLM_INCLUDE_DIR glm/glm.hpp
    HINTS
      ${CMAKE_PREFIX_PATH}
      /opt/homebrew/include
      /usr/local/include
      /usr/include
  )
  if (NOT GLM_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find glm headers. Install glm (brew install glm) or set CMAKE_PREFIX_PATH.")
  endif()
endif()

# Define assets path for shader loading
add_compile_definitions(ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets")

# ---------- Physics library ----------
add_library(physics STATIC
  src/physics/rigid_body.cpp
  src/physics/collision.cpp
  src/physics/solver.cpp
  src/physics/integrator.cpp
)
target_include_directories(physics PUBLIC include)
if (GLM_TARGET)
  target_link_libraries(physics PUBLIC ${GLM_TARGET})
else()
  target_include_directories(physics PUBLIC ${GLM_INCLUDE_DIR})
endif()

# ---------- Gfx library ----------
add_library(gfx STATIC
  src/gfx/shader.cpp
  src/gfx/mesh.cpp
  src/gfx/renderer.cpp
  src/gfx/camera.cpp
)
target_include_directories(gfx PUBLIC include)
target_link_libraries(gfx PUBLIC glad glfw)
if (GLM_TARGET)
  target_link_libraries(gfx PUBLIC ${GLM_TARGET})
else()
  target_include_directories(gfx PUBLIC ${GLM_INCLUDE_DIR})
endif()

# ---------- App ----------
add_executable(rigidbody_viewer_3d
  src/app/main.cpp
)
target_include_directories(rigidbody_viewer_3d PRIVATE include)
target_link_libraries(rigidbody_viewer_3d PRIVATE physics gfx glad glfw)

# macOS frameworks for GLFW
if (APPLE)
  target_link_libraries(rigidbody_viewer_3d PRIVATE
    "-framework Cocoa" "-framework IOKit" "-framework CoreVideo"
  )
endif()
