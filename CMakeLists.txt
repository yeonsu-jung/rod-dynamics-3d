cmake_minimum_required(VERSION 3.18)
project(rod_dynamics_3d LANGUAGES C CXX)

# ---------- CUDA (optional) ----------
# Build with:  cmake -DENABLE_CUDA=ON ..
# Requires nvcc on PATH (e.g. `module load cuda` on HPC clusters).
option(ENABLE_CUDA "Enable CUDA-accelerated capsule broadphase" OFF)
if (ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  # Default to common HPC/datacenter architectures.  Override via
  # -DCMAKE_CUDA_ARCHITECTURES="XX;YY" on the cmake command line.
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;80;86;90" CACHE STRING
        "CUDA SM architectures to compile for")
  endif()
  message(STATUS "CUDA broadphase enabled (archs: ${CMAKE_CUDA_ARCHITECTURES})")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Link stdc++fs for C++17 filesystem on older GCC/Clang
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
  link_libraries(stdc++fs)
endif()

# MSVC: define M_PI and related constants
if (MSVC)
  add_compile_definitions(_USE_MATH_DEFINES)
endif()

# Optional: nice warnings and sanitizers in Debug
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  add_compile_options(-Wall -Wextra -Wno-unused-parameter)
  # Enable native architecture optimizations for Release builds
  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-march=native)
  endif()
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
  # GLFW: vendored submodule preferred, FetchContent fallback
  if (EXISTS ${CMAKE_SOURCE_DIR}/external/glfw/CMakeLists.txt)
    add_subdirectory(external/glfw)
  else()
    message(STATUS "external/glfw not found; fetching GLFW via FetchContent")
    include(FetchContent)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
      glfw
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG 3.4
    )
    FetchContent_MakeAvailable(glfw)
  endif()

  # GLAD: vendored preferred, FetchContent fallback (pre-generated GL 4.6 core)
  if (EXISTS ${CMAKE_SOURCE_DIR}/external/glad/src/glad.c)
    add_library(glad STATIC external/glad/src/glad.c)
    target_include_directories(glad PUBLIC external/glad/include)
  else()
    message(STATUS "external/glad not found; fetching pre-generated glad via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
      glad
      GIT_REPOSITORY https://github.com/Dav1dde/glad.git
      GIT_TAG v0.1.36
      SOURCE_SUBDIR cmake
    )
    FetchContent_MakeAvailable(glad)
    if (NOT TARGET glad)
      glad_add_library(glad REPRODUCIBLE LOADER API gl:core=4.1)
    endif()
  endif()
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
# Build physics sources, conditionally adding entanglement-cpp TU if present
set(PHYSICS_SOURCES
  src/physics/rigid_body.cpp
  src/physics/collision.cpp
  src/physics/contact_geometry.cpp
  src/physics/contact_geometry_adapters.cpp
  src/physics/contact_detector.cpp

  src/physics/integrator.cpp
  src/physics/soft_contact.cpp
  src/physics/mujoco_contact.cpp
  src/physics/hertz_mindlin.cpp
  src/physics/nsc_contact.cpp
  src/physics/entanglement_adapter.cpp
)
if (EXISTS ${CMAKE_SOURCE_DIR}/external/entanglement-cpp/linking_number.cpp)
  list(APPEND PHYSICS_SOURCES external/entanglement-cpp/linking_number.cpp)
  set(HAVE_ENT_CPP_TU ON)
else()
  message(STATUS "external/entanglement-cpp/linking_number.cpp not found; assuming header-only interface.")
  set(HAVE_ENT_CPP_TU OFF)
endif()

if (ENABLE_CUDA)
  list(APPEND PHYSICS_SOURCES src/physics/cuda_broadphase.cu)
endif()

# Create physics library
add_library(physics STATIC ${PHYSICS_SOURCES})
# Ensure the entanglement-cpp translation unit does not compile its CLI main (only if compiled)
if (HAVE_ENT_CPP_TU)
  target_compile_definitions(physics PRIVATE ENT_NO_MAIN=1)
endif()
# Include entanglement headers
target_include_directories(physics PUBLIC include ${CMAKE_SOURCE_DIR}/external/entanglement-cpp)
if (GLM_TARGET)
  target_link_libraries(physics PUBLIC ${GLM_TARGET})
else()
  target_include_directories(physics PUBLIC ${GLM_INCLUDE_DIR})
endif()

# CUDA broadphase: expose USE_CUDA define and link cudart
if (ENABLE_CUDA)
  target_compile_definitions(physics PUBLIC USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(physics PUBLIC CUDA::cudart)
endif()

# ---------- Core library (World) ----------
add_library(core STATIC
  src/core/World.cpp
)
target_include_directories(core PUBLIC include)
target_link_libraries(core PUBLIC physics)
if (GLM_TARGET)
  target_link_libraries(core PUBLIC ${GLM_TARGET})
else()
  target_include_directories(core PUBLIC ${GLM_INCLUDE_DIR})
endif()

# ---------- Logging library ----------
add_library(logging STATIC
  src/logging/WorldLogger.cpp
)
target_include_directories(logging PUBLIC include)
target_link_libraries(logging PUBLIC core physics)
if (GLM_TARGET)
  target_link_libraries(logging PUBLIC ${GLM_TARGET})
else()
  target_include_directories(logging PUBLIC ${GLM_INCLUDE_DIR})
endif()

# ---------- Sim (Environments + Observers) ----------
add_library(sim STATIC
  src/sim/HeadlessEnvironment.cpp
  src/sim/EntanglementObserver.cpp
)
target_include_directories(sim PUBLIC include external/entanglement-cpp)
target_link_libraries(sim PUBLIC core logging physics)
if (UNIX AND NOT APPLE)
  target_link_libraries(sim PRIVATE pthread)
endif()

# Optional: Entanglement-cpp (parallel/OpenMP-enabled) - prefer vendored submodule, else FetchContent
set(ENTANGLEMENT_TARGET_NAME "")
if (EXISTS ${CMAKE_SOURCE_DIR}/external/entanglement-cpp/CMakeLists.txt)
  message(STATUS "Using local entanglement-cpp at external/entanglement-cpp")
  add_subdirectory(external/entanglement-cpp)
elseif(NOT BUILD_HEADLESS)
  include(FetchContent)
  message(STATUS "Fetching entanglement-cpp via FetchContent")
  FetchContent_Declare(
    entanglement_cpp
    GIT_REPOSITORY https://github.com/yeonsu-jung/entanglement-cpp.git
    GIT_TAG master
  )
  FetchContent_MakeAvailable(entanglement_cpp)
endif()

# Try to detect common target names exported by entanglement-cpp
if (TARGET entanglement_cpp)
  set(ENTANGLEMENT_TARGET_NAME entanglement_cpp)
elseif (TARGET entanglement)
  set(ENTANGLEMENT_TARGET_NAME entanglement)
elseif (TARGET entanglement::entanglement)
  set(ENTANGLEMENT_TARGET_NAME entanglement::entanglement)
endif()

if (ENTANGLEMENT_TARGET_NAME)
  message(STATUS "Linking physics against entanglement target: ${ENTANGLEMENT_TARGET_NAME}")
  target_link_libraries(physics PUBLIC ${ENTANGLEMENT_TARGET_NAME})
else()
  message(STATUS "entanglement-cpp not found or did not provide a known CMake target. If you added the submodule, ensure it defines an exportable target or adjust CMakeLists accordingly.")
endif()

# ---------- OpenMP
find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: enabling OpenMP for physics and executable")
  target_link_libraries(physics PUBLIC OpenMP::OpenMP_CXX)
else()
  # On macOS with clang, try linking to libomp if available (Homebrew)
  if (APPLE)
    find_library(HOMEBREW_OMP_LIBRARY NAMES omp libomp PATHS /opt/homebrew/lib /usr/local/lib /opt/homebrew/opt/libomp/lib)
    if (HOMEBREW_OMP_LIBRARY)
      message(STATUS "Linking against Homebrew libomp: ${HOMEBREW_OMP_LIBRARY}")
      target_link_libraries(physics PUBLIC ${HOMEBREW_OMP_LIBRARY})
      
      # Try to find the include directory
      find_path(HOMEBREW_OMP_INCLUDE_DIR omp.h PATHS /opt/homebrew/include /usr/local/include /opt/homebrew/opt/libomp/include)
      if (HOMEBREW_OMP_INCLUDE_DIR)
        message(STATUS "Found OpenMP headers: ${HOMEBREW_OMP_INCLUDE_DIR}")
        target_include_directories(physics PUBLIC ${HOMEBREW_OMP_INCLUDE_DIR})
        target_compile_options(physics PUBLIC -Xpreprocessor -fopenmp)
      endif()

      set(OpenMP_CXX_FOUND TRUE)
    endif()
  endif()
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

# ---------- LodePNG ----------
if (EXISTS ${CMAKE_SOURCE_DIR}/external/lodepng/lodepng.cpp)
  message(STATUS "Using local lodepng")
  set(LODEPNG_SOURCES external/lodepng/lodepng.cpp)
  target_include_directories(config PUBLIC external/lodepng) # Expose to others if needed
else()
  message(STATUS "Fetching lodepng via FetchContent")
  include(FetchContent)
  FetchContent_Declare(
    lodepng
    GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
    GIT_TAG master
  )
  FetchContent_MakeAvailable(lodepng)
  if(TARGET lodepng)
    set(LODEPNG_TARGET lodepng)
  else()
    set(LODEPNG_SOURCES ${lodepng_SOURCE_DIR}/lodepng.cpp)
    include_directories(${lodepng_SOURCE_DIR})
  endif()
endif()

# ---------- ZLIB (system first, FetchContent fallback for Windows/portable builds) ----------
find_package(ZLIB QUIET)
if (NOT ZLIB_FOUND)
  message(STATUS "ZLIB not found via system; fetching zlib via FetchContent")
  include(FetchContent)
  set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    zlib_fc
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.3.1
  )
  FetchContent_MakeAvailable(zlib_fc)
  if (NOT TARGET ZLIB::ZLIB)
    if (TARGET zlibstatic)
      add_library(ZLIB::ZLIB ALIAS zlibstatic)
      target_include_directories(zlibstatic PUBLIC
        ${zlib_fc_SOURCE_DIR} ${zlib_fc_BINARY_DIR}
      )
    elseif (TARGET zlib)
      add_library(ZLIB::ZLIB ALIAS zlib)
    endif()
  endif()
endif()

# ---------- App ----------
if (BUILD_HEADLESS)
  add_compile_definitions(HEADLESS_BUILD)
  add_executable(rigidbody_viewer_3d
    src/app/main.cpp
    ${LODEPNG_SOURCES}
  )
  target_include_directories(rigidbody_viewer_3d PRIVATE include external/entanglement-cpp)
  target_link_libraries(rigidbody_viewer_3d PRIVATE sim logging core physics config ZLIB::ZLIB)
  if (LODEPNG_TARGET)
    target_link_libraries(rigidbody_viewer_3d PRIVATE ${LODEPNG_TARGET})
  endif()
  if (UNIX AND NOT APPLE)
    target_link_libraries(rigidbody_viewer_3d PRIVATE pthread)
  endif()
else()
  add_executable(rigidbody_viewer_3d
    src/app/main.cpp
    ${LODEPNG_SOURCES}
  )
  target_include_directories(rigidbody_viewer_3d PRIVATE include external/entanglement-cpp)
  target_compile_definitions(rigidbody_viewer_3d PRIVATE GLFW_INCLUDE_NONE)
  target_link_libraries(rigidbody_viewer_3d PRIVATE sim logging core physics gfx glad glfw config ZLIB::ZLIB)
  if (LODEPNG_TARGET)
    target_link_libraries(rigidbody_viewer_3d PRIVATE ${LODEPNG_TARGET})
  endif()
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

# Link OpenMP to the final executable if available (will be safe even if not present)
if (TARGET rigidbody_viewer_3d)
  if (TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(rigidbody_viewer_3d PRIVATE OpenMP::OpenMP_CXX)
  elseif (APPLE AND HOMEBREW_OMP_LIBRARY)
      target_link_libraries(rigidbody_viewer_3d PRIVATE ${HOMEBREW_OMP_LIBRARY})
      if (HOMEBREW_OMP_INCLUDE_DIR)
        target_include_directories(rigidbody_viewer_3d PRIVATE ${HOMEBREW_OMP_INCLUDE_DIR})
        target_compile_options(rigidbody_viewer_3d PRIVATE -Xpreprocessor -fopenmp)
      endif()
  endif()
endif()

# ---------- Tools ----------
add_executable(convert_perrod_endpoints
  src/tools/convert_perrod_endpoints.cpp
)
if (GLM_TARGET)
  target_link_libraries(convert_perrod_endpoints PRIVATE ${GLM_TARGET})
else()
  target_include_directories(convert_perrod_endpoints PRIVATE ${GLM_INCLUDE_DIR})
endif()
if (TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(convert_perrod_endpoints PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------- Broadphase benchmark ----------
add_executable(benchmark_broadphase
  src/tools/benchmark_broadphase.cpp
)
target_include_directories(benchmark_broadphase PRIVATE include)
target_link_libraries(benchmark_broadphase PRIVATE physics)
if (TARGET OpenMP::OpenMP_CXX)
  target_link_libraries(benchmark_broadphase PRIVATE OpenMP::OpenMP_CXX)
endif()
if (UNIX AND NOT APPLE)
  target_link_libraries(benchmark_broadphase PRIVATE pthread)
endif()
