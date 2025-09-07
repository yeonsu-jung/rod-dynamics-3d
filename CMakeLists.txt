cmake_minimum_required(VERSION 3.12)
project(rigidbody3d_gl LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# GLFW (C)
add_subdirectory(external/glfw)

# GLAD (C)
add_library(glad external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)

# GLM (header-only)
find_package(glm CONFIG REQUIRED)   # brew install glm

add_executable(rigidbody_viewer_3d
  src/main.cpp
)

target_include_directories(rigidbody_viewer_3d PRIVATE
  external/glad/include
  external/glfw/include
)

target_link_libraries(rigidbody_viewer_3d PRIVATE
  glad
  glfw
  glm::glm
  ${CMAKE_DL_LIBS}
)

if(APPLE)
  target_link_libraries(rigidbody_viewer_3d PRIVATE
    "-framework Cocoa" "-framework OpenGL" "-framework IOKit" "-framework CoreVideo"
  )
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(rigidbody_viewer_3d PRIVATE -Wall -Wextra -Wpedantic)
elseif (MSVC)
  target_compile_options(rigidbody_viewer_3d PRIVATE /W4 /permissive-)
endif()
