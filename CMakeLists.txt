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

# ---------- Build options ----------
option(BUILD_HEADLESS "Build headless version without OpenGL dependencies" OFF)

# ---------- Tracy (optional) ----------
option(TRACY_ENABLE "Enable Tracy profiler zones" OFF)
if (TRACY_ENABLE)
  add_compile_definitions(TRACY_ENABLE)
  # Try to locate Tracy headers (expected to be available on include path)
  # Users can set CMAKE_PREFIX_PATH or TRACY_INCLUDE_DIR explicitly.
  find_path(TRACY_INCLUDE_DIR Tracy.hpp
    HINTS ${CMAKE_PREFIX_PATH} /usr/local/include /opt/homebrew/include
          ${CMAKE_SOURCE_DIR}/external/tracy
  )
  if (TRACY_INCLUDE_DIR)
    message(STATUS "Tracy found: ${TRACY_INCLUDE_DIR}")
    include_directories(${TRACY_INCLUDE_DIR})
  else()
    message(WARNING "Tracy headers not found; ensure tracy is installable or provide -DTRACY_INCLUDE_DIR.")
  endif()
endif()

# ---------- Dependencies ----------
# GLFW and GLAD only if not headless
if (NOT BUILD_HEADLESS)
  # GLFW (submodule / vendored)
  add_subdirectory(external/glfw)

  # GLAD (single C file)
  add_library(glad STATIC external/glad/src/glad.c)
  target_include_directories(glad PUBLIC external/glad/include)
endif()

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
    # As a last resort, fetch glm via CMake's FetchContent
    include(FetchContent)
    set(GLM_TEST_ENABLE OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      glm
      GIT_REPOSITORY https://github.com/g-truc/glm.git
      GIT_TAG 1.0.1
    )
    FetchContent_MakeAvailable(glm)
    if (TARGET glm::glm)
      set(GLM_TARGET glm::glm)
    else()
      message(FATAL_ERROR "Failed to fetch glm via FetchContent.")
    endif()
  endif()
endif()

# Define assets path for shader loading
if (NOT BUILD_HEADLESS)
  add_compile_definitions(ASSETS_DIR="${CMAKE_SOURCE_DIR}/assets")
endif()

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

# ---------- Gfx library (only if not headless) ----------
if (NOT BUILD_HEADLESS)
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
endif()

# ---------- Config library ----------
# nlohmann_json: prefer config package; else header-only fallback or FetchContent
find_package(nlohmann_json CONFIG QUIET)
if (TARGET nlohmann_json::nlohmann_json)
  set(JSON_TARGET nlohmann_json::nlohmann_json)
else()
  # If vendored header exists, use it
  if (EXISTS ${CMAKE_SOURCE_DIR}/external/nlohmann_json/include/nlohmann/json.hpp)
    add_library(nlohmann_json INTERFACE)
    target_include_directories(nlohmann_json INTERFACE ${CMAKE_SOURCE_DIR}/external/nlohmann_json/include)
    add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
    set(JSON_TARGET nlohmann_json::nlohmann_json)
  else()
    include(FetchContent)
    FetchContent_Declare(
      nlohmann_json
      GIT_REPOSITORY https://github.com/nlohmann/json.git
      GIT_TAG v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
    if (TARGET nlohmann_json::nlohmann_json)
      set(JSON_TARGET nlohmann_json::nlohmann_json)
    else()
      message(FATAL_ERROR "Failed to provide nlohmann_json dependency.")
    endif()
  endif()
endif()

add_library(config STATIC
  src/config/config.cpp
)
target_include_directories(config PUBLIC include)
target_link_libraries(config PUBLIC ${JSON_TARGET})
# ⬇️ add this so config gets GLM headers (and glm::glm if available)
if (GLM_TARGET)
  target_link_libraries(config PUBLIC ${GLM_TARGET})
else()
  target_include_directories(config PUBLIC ${GLM_INCLUDE_DIR})
endif()

# ---------- App ----------
if (BUILD_HEADLESS)
  add_compile_definitions(HEADLESS_BUILD)
  add_executable(rigidbody_viewer_3d
    src/app/main.cpp
  )
  target_include_directories(rigidbody_viewer_3d PRIVATE include)
  target_link_libraries(rigidbody_viewer_3d PRIVATE physics config)
  if (UNIX AND NOT APPLE)
    target_link_libraries(rigidbody_viewer_3d PRIVATE pthread)
  endif()
else()
  add_executable(rigidbody_viewer_3d
    src/app/main.cpp
  )
  target_include_directories(rigidbody_viewer_3d PRIVATE include)
  target_link_libraries(rigidbody_viewer_3d PRIVATE physics gfx glad glfw config)
  if (UNIX AND NOT APPLE)
    target_link_libraries(rigidbody_viewer_3d PRIVATE pthread)
  endif()

  # macOS frameworks for GLFW
  if (APPLE)
    target_link_libraries(rigidbody_viewer_3d PRIVATE
      "-framework Cocoa" "-framework IOKit" "-framework CoreVideo"
    )
  endif()
endif()
